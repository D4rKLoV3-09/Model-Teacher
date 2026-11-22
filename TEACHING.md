# Teaching Claude to read steel drawings (2D PDF -> 3D model)

You (Claude) turn the user's 2D steel drawings into a 3D model, show it, and take their corrections. Each correction
becomes a **general reading rule + a permanent test**, so the next building is read right without being told. At the end
the user sends one zip (`python teach.py pack`) to the curator, who merges the best rules from every user.

The PDF is read once into a store; reading = rules run over the store, so a better rule reruns in seconds.

## 0. Before anything (every new chat)
1. Work in the kit folder (the repo root, where `teach.py` is). A claude.ai chat: unzip the attached kit, `cd` into it.
2. `python teach.py setup` must print `SETUP OK`. If it prints `SETUP BLOCKED`, stop and tell the user the one setting it
   names (claude.ai: Settings > Capabilities > Code execution > allow network egress to package managers, then a new chat).
   Never write a stand-in reader or skip a step to get around a missing package.
3. Ask the user two things only: their name (for the pack) and a short job name. Then go.

## 1. Start
`python teach.py start <job> <drawing.pdf ...> --by "<name>"`: copies the drawings into `jobs/<job>/Drawings/Original
Drawings/`, splits the set into buildings (steel + its own sheets; site plans, M&E and RC-only sheets go to unrelated) and
drafts `jobs/<job>/job.json` for the main building. One building per job; another building = another job.

## 2. Fill job.json from the drawings (facts only)
Look at the sheets (render pages to PNG with pymupdf at ~1000 px wide; crop for detail) and fill, from what is printed:
- `grids.sheet` (the plan showing every grid bubble) and `grids.box` if the sheet holds other views; `grids.override`
  only for a grid whose position is written as a dimension that the scaled reading misses.
- `datum_m` (the level taken as Z = 0, e.g. FFL or TOC), `level_prefix` as written on the levels (EL, SHD, FFL ...).
- `marks`: the regex of member marks on this set (B1, C2, BR3, TC1 ...); `named_members` for members named in words only.
- `columns`: the column layout plan (sheet, box, scale, mark regex, base level) if there is one.
- `first_arch_sheet`: the first architectural / RC-only sheet in the building PDF (not read for steel).
- `sizes`: a mark's size only if it is written outside a schedule table; `mark_alias` when a mark has no size but is
  drawn as another; `crane_marks`, `truss` chord regexes, `detail_sheets` as the set shows them.
- `title`, `drawing_ref`, `client`, `project` from the title block.
`job.json` never carries a reading rule ("lines on this sheet are dimensions"): that is a rule for the readers (section 3).

## 3. The loop (one correction at a time)
1. **Show the reading.** `python teach.py build <job>`. Give the user: the model page `jobs/<job>/Models/<name>_Rnn.html`
   (a new revision each build, so earlier ones stay to compare; opens in any browser), the ruling sheet `jobs/<job>/Models/_work/rulings/rulings.html` (unsure readings grouped by
   kind, a drawing crop beside each option), and the numbers: parts, tonnes, per-view drawn / modelled / lost
   (`Models/Models Data/count_check.md`), floating ends and overlaps. In a claude.ai chat, present both pages as files.
2. **The user corrects.** By letter on the ruling sheet ("C4 A", "R08 B"), or in words on one item ("this line is a
   dimension, not a brace", "B24 on grid 2 is one piece"). Ask what you need to place it exactly (sheet, view, member).
3. **A ruling answer** -> `python teach.py answer <job> <RULE> <LETTER> --said "<their words>"`, rebuild, report.
4. **A misreading** -> generalise. Find *why* the reader got it wrong: which drawing pattern it misjudged. Write the rule
   by that pattern, never by member name. Good: "a level line through several bays with labels along it is split at the
   grid lines; each bay takes its nearest label". Bad: "B24 is one piece".
5. **Put the rule where it belongs** (table below), with a rule id. A reading with two valid answers becomes a ruling
   (`python generate-model/rulings.py add-rule <ID> --q "..." --opt A=... --opt B=... --default A`), default = the cautious
   option (includes steel / the heavier reading). A new drawing style: one line in `rules/drawing-styles.md`.
6. **Pin it with a test.** Cut the sheet region that shows the case:
   `python generate-model/tests/golden_check.py cut jobs/<job>/Models/_work/store <KEY> <PAGE> --box x0 y0 x1 y1 --name <what_it_shows_pNN>`
   (KEY = the store's sheet key, e.g. `01_building`), then write its `case.json` (format in the `golden_check.py`
   docstring: the rule, what the user said, the expected reading). Run `golden_check.py`: the new case must fail before
   the rule and pass after it; every older case must still pass.
7. **Rebuild and report** in a few lines: what else the rule changed (members, tonnes), per-view lost, floating ends.
8. **Log it**: `python teach.py note <job> --said "<their words>" --rule <ID> --what "<the rule in one line>" --fixture <name> --files <changed files>`.

| What was wrong | Where the rule goes |
|---|---|
| a kind of line / symbol misjudged (member vs dimension, label box, hidden line, hatch, legend) | `generate-model/view_read_store.py` (or `drawing_styles.py` for a drafting style) |
| grids, levels, scale, view placement | `generate-model/grids.py`, `register.py`, `coverage.py` |
| which members join, ends, overhangs, trusses as components | `generate-model/assemble.py`, `frame.py`, `components.py` |
| member sizes, schedule tables, catalogue | `generate-model/catalogue.py` (Consteel 2025 sections) |
| columns on the layout plan | `generate-model/columns_read.py` |
| a reading that stays a choice | a ruling in `rules/rulings.json` (`rulings.py add-rule`) |
| a fact of this job only | `jobs/<job>/job.json` (not a rule, not packed as one) |

## 4. Standards
- **A rule must hold on every drawing that shows the pattern**, not only this sheet. Never weaken or delete an existing
  test to make a new rule pass; narrow the pattern instead.
- **One fact, one place:** never hard-code this job's numbers (grid values, sheet numbers, member lists) in shared code.
- **Cautious default:** when unsure, build the reading that includes steel / the heavier one, and make it a ruling.
- **Count, don't guess:** a lost member is a reading defect, not a model patch. **No repair after the build** (no move /
  trim / drop scripts on the output): fix the reader, rerun.
- Readers use `Sheet.segs` (pieces the PDF shows), never `Sheet.segs_all` (also lines hidden under white fills / clips).
- Do not edit `generate-model/viewer_template.html` or `build_model.py` (the page is the same for every user).
- Keep replies short: result first, then what you need from the user.

## 5. Finish
When the user is satisfied (or wants to stop), `python teach.py pack <job>` (add `--with-model` if they agree to send the
model page too). Give them the zip from `out/` and say: send it to the person who shared this kit. Report: rules added
(id + one line each), tests added, golden set N/N, first -> last build numbers, rulings still open.
The zip holds: the teaching log, every file changed or added since the kit was downloaded, the job settings, the count
check, and the test crops cut from their drawings (small regions only). Nothing is uploaded anywhere by the kit.
