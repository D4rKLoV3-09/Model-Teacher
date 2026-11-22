# Drawing styles the readers know (add each new one here + in the shared reader)

| Style | Seen | Reader |
|---|---|---|
| Members as filled thin quads | Job A | `store.py` quad centre lines (principal axis) |
| Members as stroked black lines | Job B | `view_read_store.lines` |
| Member or not (the curator: dims, details, notes are not members) | Job A p4 | member pen learnt per view from the marked outlines (Job A 0.3 vs dims 0.12); an outline with writing between its faces = a box; single lines in a view drawn in outlines = not members |
| Members as outlines (two thin lines, 0.1-0.45 pt) | Job A columns, trusses | `view_read_store.outline_lines` (closest pair, overlap >= 60 % / 30 %, duplicates dropped) |
| Page rotated | Job A | `store.py` removes rotation |
| Ink the page hides (under a later white mask, outside a viewport clip: blanked title blocks, grid lines under symbols) | Job A | `store.py` vis % per piece; readers see `Sheet.segs` (>= 50 % shown) only (HIDDEN-R01) |
| Section symbol / label box: a short thin box (<= 2.2 m, >= 2.5 x as long as deep) hanging on a member line, its mark by it | Job A elevations, trusses | `view_read_store.label_boxes`: not steel, the mark names the line (BOX-R01) |
| Each kind on its own CAD layer (STEEL, Purlin, DIM, grids, Handrail, Grating, Fdn, ...) | Job A | `store.layer_class` + `Sheet.pass_mask`: the structure pass hides purlin, annotation and other trades' layers; the purlin pass reads purlin layers only; a name that says nothing is kept (LAYER-R01) |
| One level line drawn through several bays, a label box + mark per bay | Job A (D39) | `view_read_store.split_marks` splits at the grid bubbles between different box marks (BAY-R01) |
| Truss / bracing section with no level marker, grid bubbles on top, hanging down to its title; written depth upright at its side; names the main trusses at its posts (ROOF TRUSS A1) | Job A p3 | `register.elev_views` cuts stacked views between extents (marks to own title); `assemble.place_no_level`: at the panel point where the named trusses are as deep as it is (VIEW-R05) |
| Column marks written level beside an upright single line (SC6 at a column foot, SC13 above a beam) | Job A p5-p9, p18 | `view_read_store.pair`: column marks (UPRIGHT_MARK) label the upright beside them after every along-line mark is placed (COL-LEVEL) |
| Section / detail marker: bubble with a bar through its centre (S3 over '-') | Job A p10-p27 | `view_read_store._read` sets the bar aside (SYMBOL-R01) |
| Sheet legend: a short line sample + words (magenta dash-dot = MS HANDRAIL) | Job A p10, 11, 14, 16 | `view_read_store.legend` / `legend_lines`: every line of that colour + pen on the sheet = that thing (LEGEND-R01) |
| Member named in words with a leader (MONORAIL & HOIST), size / tag stacked in the same note | Job A p5-p8, p17-p19, p24 | `view_read_store.named_mark` / `named_sizes` (NAME-R02); end-on in elevations: the run comes from the plan |
| Splice: small box with an X on a member line (the member's bar stops either side of it) | Job A p7-p27 | `view_read_store.splice_boxes` + `join`: one member through it, `splices` kept on the row (SPLICE-R01) |
| Grid bubble = text in a circle; staggered bubbles for close grids | Job A | `view_read_store.bubbles` (circle centre); `grids.py` (neighbour + written dimension) |
| Grid lines grey dash-dot, main + secondary styles | Job A | `grids.py` learns the styles from lines through bubbles |
| Section cut mark = bare name, big text; names reused per sheet group | Job A | `coverage.py` (nearest sheet; a reversed pair = another cut) |
| Level marker = text on a long thin line | Job A | `view_read_store.level_pairs` (notes with "=" / CD skipped) |
| Leader labels, (TYP), arrowheads | Job A | `view_read_store.leaders`, `typ_fill` |
| Short marks written level beside outlined webs | Job A trusses | `view_read_store.pair` (across-angle, small penalty) |
| Member named in words (CRANE RUNWAY BEAM) | Job A | NAME-R01 |
| Schedule table (MARKING / MEMBER SIZE) on every sheet | Job A | `catalogue.schedule`; excluded from members |
| Member size list: `MEMBER SIZE:` heading over lines `SC1 = UB 533x210x82.2 kg/m` (rods skipped) | Job C | `catalogue.schedule` (SCHED-LIST) |
| Column layout: section symbol + boxed label + leader | Job A | `columns_read.py` |
| Side-by-side / stacked views | Job A | `register.elev_views` (marks -> nearest title below) |
| Outlined text / scans | Job C | `vectorize_drawings.py` (OCR layer) first |
| Title block right column; drawing number under any NO. label | Job A, Job C | `split_drawings.py` |
