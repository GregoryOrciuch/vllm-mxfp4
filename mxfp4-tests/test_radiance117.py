"""GPU correctness and graph replay for the optional Radiance 1.17 decode kernel."""
import torch
import radiance117_mxfp4 as kernel


def check(m, n, k, seed):
    torch.manual_seed(seed)
    device = "cuda"
    codes = torch.randint(0, 16, (n, k), device=device, dtype=torch.uint8)
    packed = (codes[:, ::2] | (codes[:, 1::2] << 4)).contiguous()
    packed = packed.view(n // 16, 16, k // 16, 2, 4).permute(0, 2, 3, 1, 4).contiguous()
    scales = torch.randint(119, 128, (k // 32, n), device=device, dtype=torch.uint8)
    refs = scales.max(dim=0).values.contiguous()
    x = torch.randn((m, k), device=device).to(torch.float8_e4m3fn).contiguous()
    xs = torch.rand((m,), device=device).float().contiguous() + 0.25
    out = torch.empty((m, n), device=device, dtype=torch.bfloat16)
    lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=device)
    weights = lut[(codes & 7).long()] * torch.where((codes & 8) != 0, -1., 1.)
    weights *= torch.exp2(scales.T.float() - 127).repeat_interleave(32, dim=1)
    expected = (x.float() * xs[:, None]) @ weights.T
    sk = 8 if m <= 16 else 4
    while sk > 1 and k % (sk * 32):
        sk //= 2

    def launch():
        kernel.launch(x.data_ptr(), xs.data_ptr(), packed.data_ptr(), scales.data_ptr(),
                      refs.data_ptr(), out.data_ptr(), m, k, n, 2, sk,
                      min(4, (m + 15) // 16), 2, torch.cuda.current_stream().cuda_stream)

    launch()
    torch.cuda.synchronize()
    error = ((out.float() - expected).norm() / expected.norm()).item()
    assert error < 0.02, (m, n, k, error)
    eager = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(eager, out), (m, n, k, "graph replay differs")
    print(f"PASS M={m} N={n} K={k} relative_error={error:.6f}")


for case in [(1, 48, 512), (5, 512, 1024), (17, 80, 1024), (64, 512, 1024)]:
    check(*case, seed=sum(case))


def check_vllm_dispatch():
    import radiance_mxfp4 as dispatch

    assert dispatch._r4d is kernel, "opt-in did not select the Radiance 1.17 kernel"
    m, n, k = 5, 512, 1024
    torch.manual_seed(42)
    codes = torch.randint(0, 16, (n, k), device="cuda", dtype=torch.uint8)
    packed = (codes[:, ::2] | (codes[:, 1::2] << 4)).contiguous()
    packed = dispatch.permute_w(packed, n, k)
    scales = torch.full((k // 32, n), 127, device="cuda", dtype=torch.uint8)
    refs = torch.full((n,), 127, device="cuda", dtype=torch.uint8)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    out = dispatch.mxfp4_linear(x, packed, scales, refs)
    lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device="cuda")
    weights = lut[(codes & 7).long()] * torch.where((codes & 8) != 0, -1., 1.)
    expected = x.float() @ weights.T
    error = ((out.float() - expected).norm() / expected.norm()).item()
    assert error < 0.05, error
    print(f"PASS vLLM dispatch relative_error={error:.6f}")

    original = kernel.launch
    def unexpected(*args):
        raise AssertionError("M > 64 must fall back to the existing kernel")
    kernel.launch = unexpected
    try:
        large = torch.randn((65, k), device="cuda", dtype=torch.bfloat16)
        out = dispatch.mxfp4_linear(large, packed, scales, refs)
        expected = large.float() @ weights.T
        error = ((out.float() - expected).norm() / expected.norm()).item()
        assert error < 0.05, error
        print(f"PASS M=65 fallback relative_error={error:.6f}")
    finally:
        kernel.launch = original


check_vllm_dispatch()
