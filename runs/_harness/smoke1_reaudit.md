# Smoke run 1: re-audit with the fixed harness (main 69a66c75), 2026-10-02

Each task worktree merged main locally (BI 27332aa9, SH 3c4d7190, LD 5c54b643; none pushed). `sandbox finish` was re-run on every
episode without --transcript. The fixed discovery found exactly the test agent's transcript each time (prompt inside the
documented workflow wrapper). Episode ids and outcomes only, no slot answers.

| task | episode | valid | invalid reasons | pass | score |
|---|---|---|---|---|---|
| boolintermediates | ep0375aa5296 | True | - | None | 1.00 |
| boolintermediates | ep179d5f1a6a | True | - | None | 1.00 |
| boolintermediates | ep3bd6b501bc | True | - | None | 1.00 |
| boolintermediates | ep3ee487e95b | True | - | None | 1.00 |
| boolintermediates | ep50448dd69f | True | - | None | 1.00 |
| boolintermediates | ep5fd72cf457 | True | - | None | 1.00 |
| boolintermediates | ep767f85e4dd | True | - | None | 1.00 |
| boolintermediates | ep7ae06129b5 | True | - | None | 1.00 |
| boolintermediates | ep80b7dbdaef | True | - | None | 1.00 |
| boolintermediates | ep93703cac08 | False | transcript_audit | None | 1.00 |
| boolintermediates | epccc2b4c495 | True | - | None | 1.00 |
| boolintermediates | epdbfedf5bf6 | True | - | None | 1.00 |
| shifthunt | ep184bffd5b2 | True | - | None | 1.00 |
| shifthunt | ep19839c342d | True | - | None | 0.67 |
| shifthunt | ep4db95008f9 | True | - | None | 0.75 |
| shifthunt | epa8848eb63c | True | - | None | 1.00 |
| shifthunt | epc93bac5e43 | True | - | None | 1.00 |
| shifthunt | epdbe6a54a7e | True | - | None | 1.00 |
| latentdiff | ep06db39d74f | True | - | None | 1.00 |
| latentdiff | ep183dbd35dd | False | transcript_audit | None | 1.00 |
| latentdiff | ep49e0123d3a | True | - | None | 1.00 |
| latentdiff | ep6e9aa23b71 | True | - | None | 1.00 |
| latentdiff | ep86ba17bb92 | True | - | None | 1.00 |
| latentdiff | epc7a9376588 | False | transcript_audit | None | 1.00 |

Totals: valid 21/24 (BI 11/12, SH 6/6, LD 4/6). Pass among valid: BI 11/11, SH 4/6, LD 4/4.
Still INVALID: ep93703cac08 (real hits: exec chain with .replace, `pkill -f`, a blocked /tmp read); ep183dbd35dd and epc7a9376588
(latentdiff false positives not yet fixed: closing tags, `https://` and `docker` inside probe text; `HERE + "/tool"` path joins).
