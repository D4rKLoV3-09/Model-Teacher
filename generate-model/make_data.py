#!/usr/bin/env python3
"""spec + parts (geometry.py) -> data.json for build_model.py, for any job (project text from job.json).

    python generate-model/make_data.py JOB_JSON SPEC PARTS LEFT_OUT GRIDS OUT

Tags: columns UC, beams + crane girders UB, bracing BR, posts UC, truss chords TC / BC, truss webs W; one tag per drawing mark
in natural order inside each family; the drawing mark is the Ref. Groups: building > level > assembly. Notes: Ruling sheet,
Cautious, Left out, Decisions, Assumptions. Decisions come from the job's teaching log (jobs/<job>/teaching/log.jsonl).
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

cfg_p = Path(sys.argv[1])
cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
spec, parts, left_out, grids = (json.load(open(a, encoding="utf-8")) for a in sys.argv[2:6])
M, PM = spec["members"], spec["_profmeta"]
assert len(M) == len(parts), (len(M), len(parts))
DWG = cfg.get("drawing_ref", "")
DATUM = float(cfg.get("datum_m", 0.0)) * 1000
LVL = cfg.get("level_prefix", "EL")
GRADE = cfg.get("grade", "S275")
TITLE = cfg.get("title") or cfg_p.parent.name
NAME = cfg.get("name") or re.sub(r"\W+", "-", cfg_p.parent.name.lower()).strip("-")
TRUSS_TOP = re.compile(cfg.get("truss", {}).get("top_chord", r"^TC\d*a?$"))
BRACE = tuple(cfg.get("brace_prefixes") or ["BR"])
FAM = {"col": "UC", "beam": "UB", "crane": "UB", "post": "UC", "brace": "BR", "web": "W"}
TRUSS_BOT = re.compile(cfg.get("truss", {}).get("bottom_chord", r"^BC\d*a?$"))
fam = lambda m: (("TC" if TRUSS_TOP.match(m["ref"]) else "BC" if TRUSS_BOT.match(m["ref"]) else "UB") if m["layer"] == "chord"
                 else "BR" if m["ref"].startswith(BRACE) else FAM[m["layer"]])
nat = lambda s: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", s)]
refs = defaultdict(set)
for m in M:
    refs[fam(m)].add(m["ref"])
tagmap = {(f, r): f"{f}{i + 1}" for f, rs in refs.items() for i, r in enumerate(sorted(rs, key=nat))}

LAYERS = [("col", "Columns", "#4f7cac"), ("beam", "Beams", "#7a9a4a"), ("post", "Posts", "#5c8a8a"), ("brace", "Bracing", "#b5523b"),
          ("chord", "Truss chords", "#c9822b"), ("web", "Truss webs", "#d9a441"), ("crane", "Crane runway girders", "#2f6f9f")]
LI = {k: i for i, (k, _, _) in enumerate(LAYERS)}
CATS = [("A", "COLUMNS", ["col"]), ("B", "BEAMS", ["beam"]), ("C", "POSTS + BRACING", ["post", "brace"]),
        ("D", "TRUSSES", ["chord", "web"]), ("E", "CRANE RUNWAY GIRDERS", ["crane"])]

profs = {}
for name, pm in PM.items():
    sp = spec["profiles"][name]
    if sp["type"] == "I":
        d, w, t = str(round(sp["h"])), str(round(sp["b"])), str(sp["kgm"])
        m2 = pm.get("_m2m") or (2 * sp["h"] + 4 * sp["b"] - 2 * sp["tw"]) / 1000
    elif sp["type"] == "L":
        d, w, t = str(round(sp["h"])), str(round(sp["b"])), str(sp["t"])
        m2 = (2 * sp["h"] + 2 * sp["b"] - 2 * sp["t"]) / 1000
    else:
        d, w, t = str(sp.get("h", "")), str(sp.get("b", "")), str(sp.get("t", sp.get("tw", "")))
        m2 = (2 * sp.get("h", 0) + 4 * sp.get("b", 0)) / 1000
    profs[name] = {"d": d, "w": w, "t": t, "cat": pm.get("_cat", ""), "kgm": sp["kgm"], "m2m": round(m2, 3),
                   "src": pm.get("_src", "") + ("" if pm.get("_m2m") else "; m2/m computed - ASSUMED")}
    if not pm.get("_m2m"):
        profs[name]["m2m_assumed"] = True

groups, gi = [], {}


def g(level, name, label=None):
    k = (level, name)
    if k not in gi:
        gi[k] = len(groups)
        groups.append({"level": level, "name": name, **({"label": label} if label else {})})
    return gi[k]


def level_name(m):
    if m["layer"] == "col":
        return "Columns"
    if m["layer"] in ("chord", "web"):
        return "Trusses"
    if m["layer"] in ("brace", "post"):
        return "Bracing + posts"
    if m["layer"] == "crane":
        return "Crane runway"
    z = (m["a"][2] + m["b"][2]) / 2 + DATUM
    return f"{LVL}{z / 1000:+.3f}"


tw = g("tower", TITLE, "Building")
out, count = [], Counter()
for m, p in zip(M, parts):
    f = fam(m)
    tag = tagmap[(f, m["ref"])]
    count[f] += 1
    q = {k: v for k, v in p.items() if k not in ("layer", "truss", "rank", "ends", "up", "reg", "key", "ids", "tview", "tat", "fixed", "comp")}
    q.update({"t": tag, "ref": m["ref"], "layer": LI[m["layer"]], "unit": 0, "tw": tw, "sub": -1, "desc": m["prof"], "grade": f"*{GRADE}",
              "src": m.get("src", ""), "grid": m.get("grid", "")})
    q["lu"] = g("lift", level_name(m), "Level")
    q["asm"] = g("asm", m["truss"] if m.get("truss") else f"{tag} ({m['id']})")
    out.append(q)

log_p = cfg_p.parent / "teaching" / "log.jsonl"
decisions = []
if log_p.exists():
    for ln in log_p.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(ln)
        except ValueError:
            continue
        if e.get("kind") in ("rule", "answer", "decision"):
            decisions.append(f"T{e.get('n', '?'):>02} {e.get('kind')}: {e.get('said', '')} -> {e.get('rule', '')} {e.get('what', '')}".strip())

cautions, cj = [], []
big = [i for i, m in enumerate(M) if m["layer"] == "crane"]
if big:
    cautions.append(f"Crane runway girders ({len(big)} pieces) bear on columns: bearing bracket not detailed in the reading.")
    cj.append({"n": "Crane girder on column", "p": big, "c": "Big member on smaller support"})

TXT = {"no section for the mark": lambda v: "Marks read with no section in the schedule: " + ", ".join(f"{k} x{n}" for k, n in sorted(v.items())) + " (not modelled).",
       "outside the building": lambda v: f"{v} read line(s) placed outside the building (a view placement off): not modelled.",
       "R02 unmarked line": lambda v: f"{v} unmarked lines on plans / elevations / sections (R02: no size given, or handrail / grating / equipment): not modelled; examples on the ruling sheet.",
       "R03 stub at a column": lambda v: f"{v} short unmarked pieces at columns (R03: drawing detail): nothing extra modelled.",
       "R07 stair stringer on a plan": lambda v: f"{v} stair stringer lines on the plans (R07): stairs not modelled.",
       "unmarked line in a truss view": lambda v: f"{v} unmarked lines in the truss views with no marked web beside them (TRUSS-R01): not modelled.",
       "views not placed": lambda v: f"{len(v)} views not placed (no two grid bubbles / no cut mark found), their members not read: " + "; ".join(v) + ".",
       "R08 members left out": lambda v: f"{v} read member(s) with an end meeting nothing read (R08): left out; on the ruling sheet."}
left = [TXT[k](v) for k, v in left_out.items() if v and k in TXT]
left += ["Connections: no plate, bolt or base plate modelled; all joints at their work points, zero gap (standing rule).",
         "Purlins, girts, sag rods, grating, handrails, stairs, ladders and RC are not read by this reader yet."]
assumptions = [
    "Read from the drawings only: every sheet read once into the drawing store; grids from the grid plan; members from every plan, "
    "grid-line elevation, section and truss view; sections from the member schedule tables.",
    f"Datum: Z = 0 at drawing level {DATUM / 1000:+.3f} m.",
    "A single line on a plan (or on an elevation at a level marker) is the member's top at that level; an outlined member is drawn to size.",
    "Joints decided from the drawing (generate-model/frame.py): an end runs along its own line to the member it is drawn against (at most 1.5 m).",
    "Section dimensions from the Consteel 2025 catalogue; web / flange thickness derived from the area for the 3D shape only.",
    f"Grade not stated: *{GRADE} (ASSUMED).",
]
gx = [[k, v] for k, v in sorted(grids["x"].items(), key=lambda t: t[1])]
gy = [[k, v] for k, v in sorted(grids["y"].items(), key=lambda t: t[1])]
kg = sum(q["kg"] for q in out)
data = {
    "meta": {"title": TITLE, "grand_total_label": TITLE.upper(), "currency": cfg.get("currency", "RM"), "export_prefix": NAME,
             "card": {"title": TITLE, "desc": "Modelled from the drawings by the store reader (rules taught by users)",
                      "ref": f"{DWG} · *{GRADE} assumed".strip(" ·")},
             "datum_z": 0, "grade": f"*{GRADE}", "unit_names": {"B1": TITLE}},
    "header": {"client": cfg.get("client", ""), "project": cfg.get("project", TITLE), "description": "STRUCTURAL STEEL (from the drawings; connections not modelled)",
               "contractor": cfg.get("contractor", ""), "date": cfg.get("date", "")},
    "parts": out, "groups": groups, "layers": [{"k": k, "n": n, "c": c} for k, n, c in LAYERS],
    "units": [TITLE], "unitKeys": ["B1"], "ghosts": [], "gx": gx, "gy": gy, "profs": profs,
    "cats": [{"code": c, "title": t, "layers": [LI[k] for k in ls], "dwg": DWG} for c, t, ls in CATS],
    "notes": assumptions, "left": left, "assumptions": assumptions, "decisions": decisions, "cautions": cautions, "caution_joints": cj,
    "noteSections": [
        {"title": "Basis", "items": [f"{DWG} {cfg.get('pdf', '')} read by script.".strip(),
                                     "Member tags: UC, UB, BR, TC / BC, W; the drawing mark is the Ref and is searchable."]},
        {"title": "Ruling sheet", "items": ["Unsure readings: Models/_work/rulings/rulings.html (each case drawn A / B / C, built with the "
                                            "answer or the default). Reply with letters (R01 A ...)."]},
        {"title": "Cautious", "items": cautions or ["None."]},
        {"title": "Left out of the model", "from": "left"},
        {"title": "Teaching log", "items": decisions or ["No correction yet."]},
        {"title": "Assumptions", "from": "assumptions"}],
}
json.dump(data, open(sys.argv[6], "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print(f"{len(out)} parts, steel {kg / 1000:,.1f} t, {len(groups)} groups; tags: " + ", ".join(f"{k} {v}" for k, v in sorted(count.items())))
