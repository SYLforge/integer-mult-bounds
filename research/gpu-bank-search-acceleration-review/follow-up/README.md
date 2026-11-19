# Follow-up: optimized CPU versus optional CUDA

[Download the reproduction archive](gpu-bank-review-reproduction.zip). SHA-256: `70ce1aee333da0331c46d5a0f4f27051d6f6b90547efa62b7cc0c1592b9ab6b2`. The archive includes reproduction instructions, timing and differential receipts, a synthetic positive control and a full-buffer test. External inputs are fetched from pinned public sources; this is not a replacement verifier.

| #7786 snapshot | Original CPU | Optimized CPU (4 workers) | Optimized hybrid |
| --- | ---: | ---: | ---: |
| `172475c2` | 9.193 s | 1.260 s | 1.219 s |
| `787cddda` | 9.067 s | 1.270 s | 1.217 s |

Medians from six interleaved fresh processes per configuration per input, on one Windows 11 system (Ryzen 9 7900X, RTX 5080). The optional GPU reduced time by about 3–4% relative to the optimized CPU. A clean-directory replay on the same machine gave similar ratios. All 53,977,655 audited candidates agreed in order, chart tuples, integer scores, and boundary verdicts. The two real inputs yielded no improvement; a separately labeled synthetic graph exercised the positive path. Linux, other devices, and broader workloads were not tested. No new κ bound.

Credit to Clear1nsight and the #7786 contributors for the public source snapshots and verifier. OpenAI Codex assisted with the implementation and validation.
