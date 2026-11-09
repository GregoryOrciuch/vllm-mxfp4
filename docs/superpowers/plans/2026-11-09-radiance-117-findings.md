# Radiance 1.17 MXFP4 compatibility findings

## Baseline

The baked fork pins the standalone Python R4D library to `b9e42ab` and applies `r4d_radiance_extras.patch`. That patch applies cleanly at this commit, but the resulting library contains no MXFP4 GEMM. `radiance_mxfp4.py` therefore normally calls the fork's own `radiance_mxfp4_fp8.hip` kernel; the optional R4D decode path defaults off and falls back if the library does not offer `mxfp4a8`.

## Upstream interface

Radiance v1.17.0 commit `19d4e8fe0e18071b9b6300f15fc4ccf38dc4fa3d` has an MXFP4 GEMM in `libr4d/r4d_gemm_mxfp4a8_nt_m64.hip`. Its plugin entry point takes `RadArgs` and `RadStream`; it is not the Python `r4d` extension consumed by this fork. The underlying C launcher accepts separate pointers for activation, activation scale, packed weights, E8M0 scales, per-row reference, and output, followed by dimensions, tuning, activation-scale layout, stream and leading dimensions.

The upstream decode kernel requires fragment-order weights, `N` divisible by 16, `M` in 1..64, and `K` divisible by `SK*32`. It supports per-row or per-128 activation scaling. The fork's optional decode path expects fragment-order weights and per-row activation scales, and the main path has additional fallback and debug behavior. The upstream kernel depends on Radiance headers and support code, and its launcher signature differs from the legacy binding expected by the fork. A safe port therefore needs an adapter, explicit shape checks, matching weight and scale layouts, and preservation of the fork's fallback.

## Verification status

The local host is macOS and has neither `hipcc` nor Python `torch`. On the remote R9700, the fork's `radiance_mxfp4_fp8.hip` compiled in `stilldeadcode/vllm-radiance:0.9.3` with two existing NWAVE redefinition warnings. One baseline case, M=5, N=512, K=1024 with constant E8M0 scales, matched a float32 dequantization oracle at relative error 0.001631. This is not a complete baseline: exponent-boundary tests, HIP graph replay, a model smoke test and a benchmark remain outstanding. Neither upstream benchmark claims nor source similarity establish correctness or a speedup in this fork's TP=1 workload.

## Implemented experimental path

`build_radiance117_mxfp4.py` generates a Python binding around the pinned upstream kernel without its standalone-server plugin entry point. The baked Dockerfiles build it, retain the upstream license and notice, and enable it by default. Set `RADIANCE_MXFP4_R4D_DECODE_MAX_M=0` to disable it; M above 64 and unsupported shapes continue to the existing path.

On `blackbox`, the generated extension compiled against `stilldeadcode/vllm-radiance:0.9.3`; `Dockerfile.ggz14.top` built successfully. `mxfp4-tests/test_radiance117.py` passed four synthetic GPU oracle cases (M=1, 5, 17 and 64; maximum relative error 0.001674), HIP graph replay, the actual vLLM dispatch (relative error 0.025483 against unquantized BF16 input), and M=65 fallback (0.026260). Explicitly disabling the dispatch was also checked. These are not model-output or performance measurements; full-model smoke tests and representative serving benchmarks remain outstanding, as do exhaustive exponent-boundary tests.
