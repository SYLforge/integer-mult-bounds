# P104 frame-basis regression and candidate-contract handoff

**Draft audit, not a new kappa record, supplier admission, or ready-to-merge construction.** This package publishes a small reproducible counterexample encountered while verifying a local successor of #7624/#7626. It does not replace or edit those PRs, selected-result.json, frozen sources, or shared verifiers. The original public predecessor PRs are not accused of this later local regression.

## Reproducible finding

For the retained 9-bit frame interpreter, the following basis replacements preserve GF(2) span but change the physical operator, including its signed Pauli action:

| Frame | Original basis | Changed local basis | Exact differing column |
|---|---|---|---|
| 1090 | 265,194,164,75 | 45,75,137,265 | 0 |
| 1091 | 11,21,51,469 | 11,21,38,448 | 16 |

Adjoint operators also differ. These are not merely global-phase differences. Restoring the literal original basis restores the original frame operator. An equality-of-span predicate is therefore insufficient to certify that these interfaces stayed unchanged. This finding does not prove every complete program using the changed basis is wrong; a different program may supply valid and fully priced corrections.

Run from any directory, Python 3.9+ standard library:

```
python3 -I -B research/p104-frame-basis-contract-audit/verify.py
```

The interpreter is vendored byte-for-byte with SHA256 authentication. On Python3.9 only, execution replaces its single `value.bit_count()` by `bin(value).count("1")`; the source file is not changed. The receipt discloses this adapter. The new wrapper checks span equality, forward/adjoint Pauli inequality, an exact Gaussian witness column, literal-basis restoration, wrong-phase rejection, and refuses optimized Python. It is not a full word or theorem verifier. No producer or Clifford compiler is imported. No external source cache/network is needed.

Observed fresh execution: macOS, CPython3.9.6, exit0. `python3 -O -I -B .../verify.py` rejects. An initial wrapper run failed on tuple/list concatenation in the mutation control; changing the list to a tuple fixed the wrapper, not a scientific assertion. The final published wrapper was rerun successfully. Hosted CI and Python3.12 are not claimed. This draft has no registered VERIFY.json hook; use the direct command above.

## Local follow-up, explicitly not reproduced by this small package

Local N24 originally had canonical SHA256 `38e63f3ebc84b3f859de0bfcb0b0e27fc013de844ba4359d45d956c57613f42c`. Restoring only those two bases produces new canonical SHA256 `327c007aeb638af62e936c0b4afe74e04c6915424b38fd8245c9b03ec9d90600`, with the instruction program unchanged. A separate local run of candidate-adapted native/signed checkers reported conditional invoice kappa `.003670439815466`,191 lifetimes,peak49,68 labels. Its chart coverage was1160 pairs and signed catalogue3123 maps. That run was interrupted and resumed, not one cold invocation. The full candidate/source closure and its application receipts are **not included or independently reproducible from this small draft**, so these are local-only handoff facts, not this PR's demonstrated result or frontier request. Later hydrogen candidates inherited changed bases and are not claimed here.

## Contract blockers remain separate

74 is a historical candidate census, not an identified universal theorem limit. Replacing74 with191 still needs a valid candidate-specific ownership/erasure application, not merely a relaxed expectation. Exact rational CLEAR certificates establish algebraic relations, not an implementation of full-extent reset on reachable rounded residuals. Common U in #7627 explicitly remains an unimplemented stronger supplier premise; no implication from the original retained erasure contract is asserted here. Finite precompilation permits a finite paid inventory under its hypotheses, but does not alone implement/reset/price it.

Two honest paths remain: bind an actual reset provider and namespace/child-return construction to the complete domain and tariff; or restrict a new candidate to the original owned-event contract and verify every premise of the applicable transfer argument. Fresh banks, nonzero rank certificates, or span equality do not by themselves discharge these obligations. Nonzero CLEAR is not automatically illegal in the retained non-reversible model.

## Sources and overlap

Base main refreshed before publication: `e54523517d0f03e58352e647c403a5af3e525a57`. Scope is a user-assigned audit of a local successor, not a new task-picker result. The native interpreter comes from #7605's retained role902 independent engine, authenticated in the existing source closure; exact file pin in SOURCES.json. Frame labels and local-candidate hashes are extraction provenance, not a claim that the local full words are publicly available in this draft.

Nearest pending work refreshed: #7627 at `f9bdd00d3869ef6e82227a9acf4633f102540f54` (courier port and explicit unimplemented U), #7628 at `6db8d8351a2bb5a530500303fb6b848b50841926` (independent invoice recount), #7629 at `c2a2131e7dba35f9396b16588e125a7e8b5fab8e` (restricted shared-call geometry). None is claimed reproduced or superseded. This bounded check is not exhaustive novelty proof. Discussion-board overlap and scripts/find_overlap.py were not rerun for this draft.

## Open next questions

1. Independently review the frame counterexample and add an operator-level regression at dual-to-primal restoration. First run the small verifier; no data blocker.
2. Publish and cold-replay the larger corrected candidate closure with exact runner diffs, sources, and all scientific receipts. Blocked by publication packaging and premise audit, not solved by this draft.
3. Derive the required reset-domain/tariff implication or produce a contract-preserving candidate. First inspect the pinned original owned-erasure/finite-precompilation contracts and #7627 COMMON-U. An implementation/proof is missing.

## Credit and license

Prepared for SYLforge with substantial OpenAI assistance. Local candidate generation involved Antigravity and AI assistance as preserved in HISTORICAL-NOTICE.txt. The retained interpreter, P104 construction and source chain retain credit to Ali Romman, Clear1nsight, DreamingOfClouds, Rohan Arun, Hao Yan/FanPUYen, ToikY, SYLforge, dillona, CubeJerry, Nick Petalas and all nested-source authors. Apache-2.0. HISTORICAL-NOTICE.txt is preserved attribution/history; its earlier certification language is superseded by this README's audit scope and is not endorsed as proof.
