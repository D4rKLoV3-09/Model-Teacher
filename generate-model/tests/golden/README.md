# Golden set

One folder per taught case: a small crop of a real drawing store + `case.json` (the rule and the expected reading).
`python generate-model/tests/golden_check.py` runs them all; every rule change must keep every case passing.
The curator's own cases are kept with the curator (cut from client drawings) and are rerun when packs are merged.
