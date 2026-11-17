# Hydrogen research snapshot: UNVERIFIED archival backup

This is a public backup on SYLforge/integer-mult-bounds, not a pull request, scientific approval, new verified bound, or runnable full source closure. No selected record, existing research branch or active experiment is modified.

The archive preserves twelve selected files from the H5B/H6A/H6B research state: candidate word, checkpoints, invoice, CLEAR certificates, candidate-adapted native receipt, three research/verification scripts, and original NOTICE. It deliberately excludes temporary worktrees, live logs, unrelated files and credentials. These selected files were read twice, then all source hashes rechecked. Candidate canonical hash matches the H6A checkpoint. This is a stable captured set, not an atomic filesystem snapshot or claim that all downstream stages finished.

## Known limits

- H5B inherits changed physical bases at frames1090/1091. Equal binary spans do not establish equality of physical frame operators.
- H6B uses candidate-adapted/span-based conditions. Its native PASS is preserved as historical evidence, not original-contract acceptance.
- Reset/provider applicability, lifetime changes and namespace/child-completion obligations remain unresolved.
- No signed completion receipt is included. Later verification output is not covered by this fixed snapshot.
- Numerical kappa claims and certification wording inside preserved files are researcher-generated, unverified claims. They are not endorsed by this archival backup.
- Four text files had personal home prefixes replaced by `<LOCAL_HOME>`. Original and public hashes plus replacement counts are recorded in MANIFEST.json. Original local bytes remain unchanged. Sanitized scripts may require explicit path/dependency setup and are not claimed ready to run.
- Full dependencies and source archives are not included. This archive is useful for preservation and review, not a replacement for full reproduction.

## Integrity check and optional extraction

Python3 standard library:

```
python3 -I -B verify_archive.py
python3 -I -B verify_archive.py --extract NEW_DIRECTORY
```

The checker decodes snapshot.tar.gz.b64, authenticates the compressed archive and every member against MANIFEST.json, checks the candidate canonical identity, and refuses existing extraction destinations or unsafe paths/member types. It verifies integrity only, not mathematics or supplier admission. Base64 is used solely because this backup was transmitted through the UTF-8 contents API.

## Credit

Retained NOTICE and source attribution are preserved; its scientific claims remain historical/unverified as stated above. Prepared for SYLforge with OpenAI assistance. Apache-2.0 and all inherited source notices apply. No PR creation or selected-frontier change accompanies this backup.
