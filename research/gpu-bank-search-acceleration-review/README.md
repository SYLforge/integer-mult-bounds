# GPU bank-search acceleration (review package)

This is an optional search accelerator, not a certificate verifier or a new κ bound. It makes no changes to upstream checks, the selected frontier, or other contributors’ packages.

The benchmark uses the rational WORD and LAYOUT from [Clear1nsight’s #7786 snapshot](https://github.com/CrocSwap/integer-mult-bounds/commit/172475c2ca321d3496fda7a2a71478ada5acb948). The exact compressed input digests are checked by the scripts. The legacy scripts are our previous local comparator, not the project’s official verifier. New work: reuse price-table preparation and generate/score combinations directly on the GPU. OpenAI Codex assisted with implementation and testing.

On one Windows 11 machine (Ryzen 9 7900X, RTX 5080, driver 621.14), fresh-process medians over six alternating runs were 9.676 s for legacy CPU, 21.807 s for legacy hybrid, and 1.278 s for the new hybrid. The legacy hybrid slowdown relative to its historical 8.268 s is unexplained; even against that faster earlier median the new run was about 6.47× faster. See `evidence/benchmark.json` (local path prefixes anonymized) and `evidence/audit-final.json`. The 27,397,025 enumerated assignments matched in order, chart IDs and integer scores; the single near-boundary case had zero exact cost change. The capped search is not exhaustive over all possible bank configurations. No κ improvement was found.

## Reproduce on Windows

Requires Python 3, NumPy, CUDA-enabled PyTorch for the legacy comparator, an NVIDIA driver, and CUDA NVRTC (install the CUDA Toolkit and expose its bin directory through CUDA_PATH if the bundled PyTorch DLLs are not found). The original run used Python 3.13.14, NumPy 2.5.4 and PyTorch 2.14.1+cu130. Run from this package directory:

```powershell
.\REPRODUCE.ps1 -PythonExe python
```

The exact differential audit runs separately from the timed trials. A local replay must not be taken as proof of portability, a guaranteed speedup, or a new mathematical result. Review open points: a second public input on this path, an optimized CPU comparator that shares preparation, and a positive end-to-end candidate control.
