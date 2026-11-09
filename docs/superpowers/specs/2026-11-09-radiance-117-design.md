# Radiance 1.17 integration design

Keep the existing vLLM server, model formats and launchers while bringing in compatible improvements from Radiance v1.17.0. The target is a single Radeon Pro R9700 (gfx1201), tensor parallelism 1, on Linux; the Ryzen 7 5700X is the host CPU. Preserve existing fallbacks and fork-specific behavior. Do not replace vLLM with Radiance's standalone server or assert speedups without GPU measurements.

## Approaches

- Selectively adapt compatible upstream kernels to the existing Python R4D integration: chosen because it preserves the fork's interface and permits correctness and performance checks per kernel.
- Replace the legacy R4D module wholesale with the new Radiance plugin: not chosen because the new RadArgs plugin interface is different from the Python module and would also need the fork's local extensions ported.

Start by establishing a precise compatibility and performance baseline for the MXFP4 path; port only an improvement whose interfaces, numerical behavior and testing can be established. Keep existing multi-GPU paths untouched. Record limitations and commands for hardware validation.
