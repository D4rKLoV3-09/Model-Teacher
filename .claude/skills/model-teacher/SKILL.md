---
name: model-teacher
description: Turn 2D steel drawings (PDF) into a 3D model and learn from the user's corrections (reading rules + tests + teaching log + zip pack). Use for any drawing, model, correction, ruling answer, "teach" or "pack" request in this repo.
---

Follow `TEACHING.md` in the repo root, in order:
0. `python teach.py setup` (must print SETUP OK; if BLOCKED, tell the user the setting it names and stop).
1. `python teach.py start <job> <pdf> --by "<name>"`.
2. Fill `jobs/<job>/job.json` from the drawings (facts only).
3. Loop: `python teach.py build <job>` -> show model page + ruling sheet + numbers -> user corrects -> answer
   (`teach.py answer`) or general rule in the reader + golden test (`generate-model/tests/golden_check.py cut`) -> rebuild ->
   `teach.py note`.
4. `python teach.py pack <job>` when the user is satisfied; hand them the zip.
