# Research closeout — 2026-11-17

Archived at the owner's request. Research is paused. This is not a pull request, selected record update, runtime benchmark, or proof of a complete multiplier.

## Subsequent clarification to frozen audit notes

The original package above is preserved byte-for-byte. Its statement that tariff conversion remains owed must be read alongside the later selected #7000 application review: docs/research/pr-intake-evidence/pr7000-application/PROOF-REVIEW.md at CrocSwap/integer-mult-bounds commit e54523517d0f03e58352e647c403a5af3e525a57, sections 4 and 6.

That review supplies conservative scalar-bit and monomial-record cost substitutions under retained assumptions, and expressly does not require the raw 10^9 pass-unit cap. It does not identify 4000/pair or 32/service with individual Tape.read/write actions. H5B-specific full service/compiler/precision/cost binding remains unestablished in this archive. Do not generalize an earlier raw-tariff gap into a claim that the selected #7000 application has no physical-cost argument.

## Direct same-caller comparison

#7605 canonical e20f3f9240ac3f5fa4bbd7ab9b5550d660886901bb209f335409c329ac45d9cd versus restored H5B canonical ad56c629ac587510aa59ad13fde7db3f38e08d4fc09beb70fbb839a8de17b41d, same signed 17-stage caller:
- Native lifetimes 74 -> 232, peak 50 -> 50.
- Productive fields 22461 unchanged; paid calls 110366 -> 109057.
- Historical normalized ordinary upper 782699212 -> 792444836 (+9745624).
- Scalar services 43923200 -> 44670656.
- Conditional raw crossing approximately .00365044424471688776 -> .00369858869752724585.
These are not measured execution times. More scalar/reset work trades against a more favorable recursive rank profile. No unconditional improvement is asserted.

## Bounded zero-extra-CLEAR alternative, not fully verified

A later read-only in-memory exploration found 20 adjacent public SCALE pairs in the original #7605 word. In each pair the first factor is -1, the second has magnitude 1/2, 1, or 2, and role and literal frame match. Replace each pair by one SCALE with the product factor. Retain even a resulting factor +1 for this exact candidate.

Zero-based original positions:
29059, 29078, 32062, 32494, 32947, 36285, 36414, 36808, 36828, 37205, 37288, 37299, 37402, 37441, 37716, 37832, 38759, 38761, 45637, 45642.

Construct by traversing the original program left-to-right; at each listed position emit ['SCALE', role, str(Fraction(first_factor)*Fraction(second_factor)), frame], skip the following instruction; otherwise copy the instruction. Preserve every other JSON field. Canonical JSON uses sort_keys=True,separators=(',',':') and UTF-8.

Candidate canonical SHA256 b515babfd6fb8b2401b37231fa1b081df2de6e7a2086434e2200fbf367737947.
The in-memory check confirmed 20 fewer SCALEs and identical ordered INIT/CLEAR/COPY/MOVE lists. Local exact scalar identities hold for arbitrary inputs. Frame and endpoint metadata are unchanged. The paid/target histogram and kappa do not improve. The inherited 32*17 macro census predicts 10880 fewer scalar services.

Full original-engine replay, complete precision/invoice replay and novelty review were NOT performed. This is an ordinary-only exploratory recipe, not an accepted contribution. No candidate source artifact was written over the original.

## Stop condition

No additional search, upload to upstream, or PR creation follows from this archive. Original research and earlier evidence remain untouched. Reproduction limitations in README and MANIFEST continue to apply, including same-Mac relocation and reused native/signed receipts.
