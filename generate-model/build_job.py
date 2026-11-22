#!/usr/bin/env python3
"""One building from its drawings -> 3D model page, with the store chain. Everything job-specific is in the job's job.json.

    python generate-model/build_job.py JOB            (jobs/JOB/job.json; template: generate-model/templates/job.json)

store (PDF read once) -> grids -> coverage -> register (every plan, elevation, section, truss view) -> schedule (MARKING /
MEMBER SIZE tables) -> columns (layout plan, if given) -> assemble (rulings) -> frame (joints) -> geometry -> data -> clash
-> page -> ruling sheet. Each unsure case is a ruling card; the job builds with the rule's answer (or its default).
Outputs (jobs/JOB/): Models/<name>_Rnn.html (a new revision each build), Models/Models Data/ (spec, parts, data, count check, left out), Models/_work/
(store, register, rulings/rulings.html). No reading rule lives here: rules go in the readers (see rules/teaching.md).
"""
import copy
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "generate-model"))
if len(sys.argv) > 1 and (ROOT / "jobs" / sys.argv[1] / "job.json").exists():     # the job's level word, for every reader (and
    os.environ["STEEL_LEVEL_PREFIX"] = json.loads((ROOT / "jobs" / sys.argv[1] / "job.json").read_text(encoding="utf-8")).get("level_prefix") or "SHD"
import assemble  # noqa: E402
import catalogue  # noqa: E402
import columns_read  # noqa: E402
import components  # noqa: E402
import frame  # noqa: E402
import reconcile  # noqa: E402
import revision  # noqa: E402
import rulings  # noqa: E402
import store  # noqa: E402
import view_read_store as view_read  # noqa: E402

if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
    sys.exit(__doc__)
JOB = sys.argv[1]
J = ROOT / "jobs" / JOB
CFG_P = J / "job.json"
if not CFG_P.exists():
    sys.exit(f"no {CFG_P.relative_to(ROOT)}: copy generate-model/templates/job.json there and fill it (rules/teaching.md step 1.6)")
CFG = json.loads(CFG_P.read_text(encoding="utf-8"))
NAME = CFG.get("name") or re.sub(r"\W+", "-", JOB.lower()).strip("-")
W = J / "Models/_work"
PDF = J / CFG["pdf"]
PAGE_DIR, DATA_DIR = J / "Models", J / "Models/Models Data"
for d in (W, PAGE_DIR, DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)
SPEC, PARTS, DATA = (DATA_DIR / f"{NAME}.{k}.json" for k in ("spec", "parts", "data"))
PAGE = revision.next_set(PAGE_DIR / f"{NAME}.html")         # never replaced: each build is a new <name>_Rnn.html
PY = sys.executable
MARK = CFG.get("marks") or r"([A-Z]{1,3}\d{1,3}[a-zA-Z]?)"
NAMED = CFG.get("named_members") or []                  # members named in words, not marked (CRANE RUNWAY BEAM)
if NAMED:
    _nm = "|".join(re.escape(n).replace(r"\ ", " ") for n in NAMED)     # the register matches the words as written
    MARK = MARK[:-1] + "|" + _nm + ")" if MARK.endswith(")") else f"({MARK}|{_nm})"
DATUM = float(CFG.get("datum_m", 0.0))                  # model Z = 0 at this drawing level (m)
FIRST_ARCH = int(CFG.get("first_arch_sheet") or 999)    # sheets from here on are architectural / RC: not read for steel
TRUSS_TOP = re.compile(CFG.get("truss", {}).get("top_chord", r"^TC\d*a?$"))
TRUSS_BOT = re.compile(CFG.get("truss", {}).get("bottom_chord", r"^BC\d*a?$"))
TRUSS_CHORD = re.compile(f"({TRUSS_TOP.pattern})|({TRUSS_BOT.pattern})")
CRANE = set(CFG.get("crane_marks") or [])
BRACE = tuple(CFG.get("brace_prefixes") or ["BR"])
OWN = ("SIZE-R01", "VIEW-R01", "VIEW-R02", "VIEW-R05", "VIEW-R03", "SCALE-R01", "SCHED-R01", "SCHED-R03", "R11", "NAME-R01", "R10",
       "R01", "R02", "R03", "R05", "R06", "R07", "R08", "R09", "COL-R01")


def run(*cmd, out=None, ok=(0,)):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, cwd=ROOT, encoding="utf-8", errors="replace")
    if out:
        Path(out).write_text(r.stdout, encoding="utf-8")
    print(f"{Path(str(cmd[1])).name:<16} {(r.stdout.strip().splitlines() or [''])[0][:150]}")
    if r.returncode not in ok and not out:
        sys.exit(f"FAILED: {' '.join(map(str, cmd))}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return r


if not PDF.exists():
    sys.exit(f"drawing not found: {PDF} (job.json 'pdf' is relative to jobs/{JOB}/)")
# ---------- reading (each step cached by input hash) ----------
run(PY, ROOT / "generate-model/store.py", "extract", PDF, "--out", W / "store")
rulings.reset_cases(W, rules=OWN)
SHEETS = store.sheets(W / "store")
GC = CFG.get("grids") or {}
g_sheet = int(GC.get("sheet", 1))
g_cmd = [PY, ROOT / "generate-model/grids.py", W / "store", "--sheet", g_sheet, "--out", W / "grids_read.json"]
if GC.get("box"):
    g_cmd += ["--box", *GC["box"]]
if GC.get("scale"):
    g_cmd += ["--scale", GC["scale"]]
if GC.get("origin"):
    g_cmd += ["--origin", *GC["origin"]]
run(*g_cmd)
G = json.loads((W / "grids_read.json").read_text(encoding="utf-8"))
for ax in ("x", "y"):                                       # positions written on the drawing that the reader cannot take (job.json)
    for k, v in (GC.get("override") or {}).get(ax, {}).items():
        G[ax][k] = v
        G.setdefault("src", {})[k] = "job.json override (written dimension)"
    G[ax] = dict(sorted(G[ax].items(), key=lambda t: t[1]))
(W / "grids.json").write_text(json.dumps(G, indent=1))
if len(G["x"]) < 2 or len(G["y"]) < 2:
    sys.exit(f"grids: only {len(G['x'])} x / {len(G['y'])} y read on p{g_sheet}: set grids.sheet / grids.box in job.json to the plan "
             "that shows every grid bubble (or add grids.override)")
run(PY, ROOT / "generate-model/coverage.py", W / "store", "--grids", W / "grids.json", "--datum", str(DATUM), "--first-arch", str(FIRST_ARCH),
    "--out", W / "coverage.json", "--md", W / "coverage.md")
run(PY, ROOT / "generate-model/register.py", W, "--mark", MARK, "--datum", str(DATUM), "--first-arch", str(FIRST_ARCH))
for v in json.loads((W / "register/elevations.json").read_text(encoding="utf-8"))["views"]:     # SCALE-R01 cards
    sf = v.get("fit", {}).get("scale_found", "")
    if "SCALE-R01" in sf:
        rulings.case(W, "SCALE-R01", readings={"A": f"placed at {sf.split(':')[0]}:{sf.split(':')[1]}", "B": "view refused (its members not read)"},
                     note=f"p{v['sheet']} {v['title'][:50]}: {sf}", key=f"SCALE-R01:{v['sheet']}:{v['title'][:30]}")

# ---------- schedule ----------
STEEL = SHEETS[:FIRST_ARCH - 1]
sched_raw, srep = catalogue.schedule(STEEL)
SCHED = {k: catalogue.designation(v) for k, v in sched_raw.items()}
SCHED.update({k: catalogue.designation(v) for k, v in (CFG.get("sizes") or {}).items()})   # sizes written outside a schedule table
PROF = {}


def profile(name):
    n = name.replace("UA", "L ").replace("X", "x")
    if n not in PROF:
        PROF[n] = catalogue.profile(n)
    return PROF[n]


def kgm(size):
    try:
        return profile(catalogue.designation(size)).get("kgm", 0)
    except Exception:
        return 0


rulings.rule("SCHED-R01", "A mark is listed twice in the schedule with two sizes. Which size?",
             {"A": "The heavier", "B": "The lighter"}, "A")
for mk in sorted({c[0] for c in srep["conflicts"]}):
    sizes = sorted({c[1] for c in srep["conflicts"] if c[0] == mk} | {c[2] for c in srep["conflicts"] if c[0] == mk}, key=kgm)
    hv, lt = catalogue.designation(sizes[-1]), catalogue.designation(sizes[0])
    rulings.case(W, "SCHED-R01", readings={"A": f"{mk} = {hv}", "B": f"{mk} = {lt}"}, note=f"schedule: {mk} listed as {' and '.join(sizes)}",
                 key=f"SCHED-R01:{mk}")
    SCHED[mk] = hv if rulings.choice("SCHED-R01") == "A" else lt
for mk, base in (CFG.get("mark_alias") or {}).items():      # a mark the schedule does not list, sized as another (job.json)
    if mk not in SCHED and base in SCHED:
        SCHED[mk] = SCHED[base]


def size_under(name):
    """A member named in words with its size written under the name (CRANE RUNWAY BEAM / UB 1016x305x487): that size."""
    for sh in STEEL:
        for w in sh.words:
            if w[4].strip() == name:
                for v in sh.words:
                    if 0 <= v[1] - w[3] <= 14 and abs(v[0] - w[0]) <= 10 and re.match(r"(UB|UC|PFC|C|L|UA)\s*\d+\s*[xX]\s*\d+", v[4]):
                        return catalogue.designation(v[4])
    return None


for nm in NAMED:
    if nm not in SCHED:
        s_ = size_under(nm)
        if s_:
            SCHED[nm] = s_
        else:
            print(f"named member  {nm!r}: no size written under it and none in job.json 'sizes' (its lines are left out, listed)")
for _nm, _found in view_read.named_sizes(STEEL, MARK).items():   # NAME-R02: a member named in words takes the size found with it
    _sz = [SCHED[f] if f in SCHED else catalogue.designation(f) for f in _found if f in SCHED or view_read.SECTION_TXT.match(f)]
    if _sz and _nm not in SCHED:
        SCHED[_nm] = Counter(_sz).most_common(1)[0][0]
        print(f"NAME-R02         {_nm} = {SCHED[_nm]} (found with the name: {_found})")
seen = set()
for f in ("plans.json", "elevations.json"):
    for v in json.loads((W / "register" / f).read_text(encoding="utf-8"))["views"]:
        seen.update(r["mark"] for r in v["rows"] if r["mark"])
rulings.rule("SCHED-R03", "A mark on the drawings is not in the schedule but its base is (BC1a, schedule has BC1). Its section?",
             {"A": "The base mark's section (BC1a = BC1)", "B": "Left out and listed"}, "A")
for mk in sorted(seen):
    base = re.sub(r"[a-z]$", "", mk)
    if mk not in SCHED and base != mk and base in SCHED:
        rulings.case(W, "SCHED-R03", readings={"A": f"{mk} = {SCHED[base]} (as {base})", "B": f"{mk} left out"}, note=f"mark {mk}", key=f"SCHED-R03:{mk}")
        if rulings.choice("SCHED-R03") == "A":
            SCHED[mk] = SCHED[base]
for mk, sz in list(SCHED.items()):                         # a size the catalogue cannot read: that mark is left out, listed
    try:
        profile(sz)
    except Exception as ex:
        print(f"schedule         {mk} = {sz!r}: no catalogue section ({str(ex)[:80]}): its members are left out (listed)")
        del SCHED[mk]
print(f"schedule         {len(SCHED)} marks with a size; marks read on the drawings with no size: "
      + (", ".join(sorted(m for m in seen if m not in SCHED)) or "none"))

# ---------- columns (layout plan, if the job has one) ----------
CC = CFG.get("columns") or {}
cols_l = []
z_bos = round((float(CC.get("base_level_m", DATUM)) - DATUM) * 1000)
if CC.get("sheet"):
    csh = SHEETS[int(CC["sheet"]) - 1]
    cols_l, crep = columns_read.read_columns(csh, CC.get("box") or [0, 0, *csh.size], G["x"], G["y"], float(CC.get("scale", 100)),
                                             CC.get("mark", r"SC\d+"), CC.get("base_plate", r"BP\d+"))
    print(f"columns_read     {crep['columns']} columns from {crep['symbols']} symbols; labels reaching none {len(crep['labels_reaching_no_symbol'])}, "
          f"symbols with no label {len(crep['symbols_with_no_label'])}")
else:
    print("columns_read     no column layout in job.json: columns come from the elevations only")

# ---------- assemble ----------
REG_P = json.loads((W / "register/plans.json").read_text(encoding="utf-8"))["views"]
REG_E = json.loads((W / "register/elevations.json").read_text(encoding="utf-8"))["views"]
rulings.rule("SIZE-R01", "A member drawn to size (outline) carries a mark whose section is far off its drawn depth (an angle's mark beside a "
             "deep chord). Which member does the mark name?", {"A": "The nearest unmarked line at the label (the web)", "B": "The outline (as paired)"}, "A")


def _seg_d(p, a, b):
    ax, ay = a; bx, by = b; px, py = p
    L2 = (bx - ax) ** 2 + (by - ay) ** 2 or 1e-9
    t = max(0, min(1, ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / L2))
    return math.hypot(px - ax - t * (bx - ax), py - ay - t * (by - ay))


n_size = 0
for v in REG_P + REG_E:
    if v["status"] != "read":
        continue
    for r in v["rows"]:
        if r["kind"] != "outline" or not r.get("mark") or r["mark"] not in SCHED or not r.get("depth_mm") or not r.get("label_pt"):
            continue
        pf = profile(SCHED[r["mark"]])
        dims = [d for d in (pf.get("h"), pf.get("b"), pf.get("d")) if d]
        h = max(dims or [0])
        if not h or any(0.6 <= r["depth_mm"] / d <= 1.6 for d in dims):
            continue
        cand_ = [q for q in v["rows"] if not q.get("mark") and not str(q["status"]).startswith("not a member")
                 and _seg_d(r["label_pt"], q["a_pt"], q["b_pt"]) < 30]
        q = min(cand_, key=lambda q: _seg_d(r["label_pt"], q["a_pt"], q["b_pt"]), default=None)
        rulings.case(W, "SIZE-R01", v["sheet"], [min(r["a_pt"][0], r["b_pt"][0]) - 60, min(r["a_pt"][1], r["b_pt"][1]) - 60,
                     max(r["a_pt"][0], r["b_pt"][0]) + 60, max(r["a_pt"][1], r["b_pt"][1]) + 60],
                     readings={"A": f"{r['mark']} -> {q['id'] if q else 'no line near the label (mark dropped)'}", "B": f"{r['mark']} = outline {r['id']} ({r['depth_mm']} mm deep)"},
                     note=f"{r['id']} drawn {r['depth_mm']} mm, {r['mark']} {SCHED[r['mark']]} is {round(h)} mm", key=f"SIZE-R01:{v['sheet']}:{v['title'][:20]}")
        if rulings.choice("SIZE-R01") == "A":
            if q:
                q["mark"], q["status"], q["label_pt"] = r["mark"], "SIZE-R01 A", r["label_pt"]
            r["mark"], r["status"] = None, "unlabelled (SIZE-R01: mark off its drawn size)"
            n_size += 1
print(f"SIZE-R01         {n_size} marks moved from an outline of another size to the line they name")
rulings.rule("VIEW-R01", "A section shows one grid bubble only: which way does it face (grid values growing left to right, or right to left)?",
             {"A": "Left to right (as the plan's grid order)", "B": "Right to left (mirrored about that grid)"}, "A")
rulings.rule("VIEW-R02", "A bracing truss view has no level marker: where is its top?",
             {"A": "Its top flush with the top of the main roof trusses' top chord where it crosses them",
              "B": "Its top line on the main top chord's centre line"}, "A")
rulings.rule("VIEW-R03", "A plan shows grid bubbles on one axis only; the sheet's elevation names two grids. Its column row is on which grid?",
             {"A": "Both (one platform on each)", "B": "The first named only", "C": "The second named only"}, "A")
for v in REG_E:
    f_ = v.get("fit", {})
    if v["status"] == "read" and "VIEW-R01" in str(f_.get("method")):
        rulings.case(W, "VIEW-R01", v["sheet"], v["box"], readings={"A": f"{v['title']} @ {v['where']}: as read", "B": "mirrored about its grid"},
                     note=f"p{v['sheet']} {v['title']} ({v['where']})", key=f"VIEW-R01:{v['sheet']}:{v['title']}:{v['where']}")
        if rulings.choice("VIEW-R01") == "B":
            k_ = 0 if v["axis"] == "x" else 1
            Gk = {**G["x"], **G["y"]}
            g0 = next((Gk[n] for n in f_.get("h_grids", []) if n in Gk), None)
            for r in v["rows"] if g0 is not None else []:
                for e in ("a", "b"):
                    r[e][k_] = round(2 * g0 - r[e][k_], 1)
anch = sorted({v["anchor"] for v in REG_P if v.get("anchor")})
if anch:
    rulings.case(W, "VIEW-R03", readings={"A": " and ".join(f"grid {g}" for g in anch), "B": f"grid {anch[0]}", "C": f"grid {anch[-1]}"},
                 note="plan with grid bubbles on one axis", key="VIEW-R03:" + ",".join(anch))
    keep = {"A": anch, "B": anch[:1], "C": anch[-1:]}[rulings.choice("VIEW-R03")]
    REG_P = [v for v in REG_P if not v.get("anchor") or v["anchor"] in keep]
for _t, _a, _s, _d, _hd in assemble.place_no_level(REG_E, lambda pg: SHEETS[pg - 1].words, TRUSS_TOP, TRUSS_BOT, rulings, W):
    print(f"VIEW-R05         {_t[:40]}: {_a} -> {_s} (depth {_d:.0f}, trusses {_hd})")
tc_rows = [(v, r) for v in REG_E if v["status"] == "read" and re.search("TRUSS", v["title"]) and not v["fit"].get("z_rel")
           for r in v["rows"] if r.get("mark") and TRUSS_TOP.match(r["mark"])]
for v in REG_E:                                             # VIEW-R02: a truss view with no level set on the main trusses it crosses
    if v["status"] != "read" or not v["fit"].get("z_rel"):
        continue
    fixed = 1 if v["axis"] == "x" else 0
    tops = []
    for tv, r in tc_rows:
        a_, b_ = r["a"], r["b"]
        if tv["axis"] == v["axis"] or not min(a_[fixed], b_[fixed]) <= v["at"] <= max(a_[fixed], b_[fixed]) or abs(a_[fixed] - b_[fixed]) < 1:
            continue
        t_ = (v["at"] - a_[fixed]) / (b_[fixed] - a_[fixed])
        zc = a_[2] + t_ * (b_[2] - a_[2])
        h_ = profile(SCHED[r["mark"]])["h"] if r["mark"] in SCHED else 0
        tops.append(zc + (h_ / 2 if rulings.choice("VIEW-R02") == "A" else 0))
    if not tops:
        v["status"] = "needs placement: no main truss crosses it (VIEW-R02)"
        continue
    dz = sorted(tops)[len(tops) // 2]
    for r in v["rows"]:
        for e in ("a", "b"):
            if r[e][2] is not None:
                r[e][2] = round(r[e][2] + dz, 1)
    rulings.case(W, "VIEW-R02", v["sheet"], v["box"], readings={"A": f"top at z {round(dz)} mm (main top chord top, {len(tops)} crossings)",
                 "B": "top on the main top chord centre line"}, note=f"p{v['sheet']} {v['title'][:40]} @ {v['where'][:40]}",
                 key=f"VIEW-R02:{v['sheet']}:{v['title']}:{v['where']}")
gx_, gy_ = list(G["x"].values()), list(G["y"].values())
mg = float(CFG.get("bound_margin_mm", 9000))
BOUND = tuple(CFG.get("bound_mm") or (min(gx_) - mg, max(gx_) + mg, min(gy_) - mg, max(gy_) + mg, -mg, float(CFG.get("max_height_mm", 60000))))
asm = assemble.Assembler(W, SCHED, lambda s: profile(s), rulings, [(c["x"], c["y"]) for c in cols_l], DATUM)
cand = asm.candidates(REG_P, "plan", BOUND) + asm.candidates(REG_E, "elev", BOUND)
asm.level_ranges(cand)
mem = asm.merge(cand)
(DATA_DIR / "trace_marks.json").write_text(json.dumps({"cand": Counter(c["mark"] for c in cand), "merged": Counter(m["mark"] for m in mem)}))
cols, rest, extra_cols = assemble.columns(cols_l, mem, z_bos, SCHED, profile, rulings, W)
print(f"assemble         {len(cand)} marked lines -> {len(mem)} members; {len(cols)} layout columns "
      f"({sum(c['z1'] is not None for c in cols)} with a top read), {len(extra_cols)} extra column lines; rules " +
      ", ".join(f"{k} {v}" for k, v in sorted(asm.n.items()) if not k.startswith("left")))

# ---------- spec members ----------
R08 = DATA_DIR / "r08_leave_out.json"
if "--keep-r08" not in sys.argv or not R08.exists():
    R08.write_text("[]")
DOC = CFG.get("drawing_ref", "")


def spec_of(e, n):
    vertical = abs(e["a"][0] - e["b"][0]) < 60 and abs(e["a"][1] - e["b"][1]) < 60
    inc = abs(e["a"][2] - e["b"][2]) > 300 and not vertical
    if e.get("truss"):
        layer = "chord" if TRUSS_CHORD.match(e["mark"]) or (abs(e["a"][2] - e["b"][2]) < 30 and not e["mark"].startswith(BRACE)) else "web"
    elif e["mark"] in CRANE:
        layer = "crane"
    elif vertical:
        layer = "post"
    elif inc or e["mark"].startswith(BRACE):
        layer = "brace"
    else:
        layer = "beam"
    rank = {"chord": 1, "crane": 1, "beam": 2, "post": 2, "web": 3, "brace": 3}[layer]
    return {"id": f"M{n + 1}", "ref": e["mark"], "prof": SCHED[e["mark"]], "a": list(e["a"]), "b": list(e["b"]), "layer": layer, "rank": rank,
            "src": f"{DOC} {e['src']} (register {', '.join(e['ids'][:3])})".strip(), "reg": e["ids"][:4], "ids": e["ids"],
            "key": f"{e['mark']}:" + ":".join(str(round(v / 50)) for v in e["a"] + e["b"]),
            "truss": e.get("truss"), "grid": e["src"][:60], "tview": e.get("tview"), "tat": e.get("tat")}


BASE = [spec_of(e, n) for n, e in enumerate(rest + extra_cols)]
P0 = {m["prof"]: profile(m["prof"]) for m in BASE}
# ---------- components: each truss type built and checked alone, then copied to every placement ----------
COMP, comp_rep = components.trusses([m for m in BASE if m.get("tview")], P0)
# details placed by their label (a frame drawn alone at a large scale, labelled on the plans inside a drawn X)
DET_RE = re.compile(CFG.get("detail_title", r"PLAN FOR (\w+)\s*@\s*[A-Z]*\s*[+]?\s*(\d+\.\d+)"))
COV = json.loads((W / "coverage.json").read_text(encoding="utf-8"))["views"]
for det_sheet in CFG.get("detail_sheets") or []:
    for v in [v for v in REG_P if v["sheet"] == det_sheet]:
        mt = DET_RE.search(v["title"])
        if not mt:
            continue
        lab, ztos = mt.group(1), round((float(mt.group(2)) - DATUM) * 1000)
        sh = SHEETS[det_sheet - 1]
        k_ = 25.4 / 72 * (v["scale"] or 20)
        tt = sorted((t for t in COV if t["sheet"] == det_sheet and t["kind"] == "plan"), key=lambda t: t["at"][0])
        me = next((t for t in tt if t["title"].strip()[:20] == v["title"].strip()[:20]), None)
        if me:                                              # a detail has no bubble: its box = the band above its own title,
            i_ = tt.index(me)                               # halfway to the titles either side
            x0_ = (tt[i_ - 1]["at"][0] + me["at"][0]) / 2 if i_ else 20
            x1_ = (tt[i_ + 1]["at"][0] + me["at"][0]) / 2 if i_ + 1 < len(tt) else sh.size[0] - 20
            v["box"] = [x0_, 40, x1_, me["at"][1] - 6]
        cx, cy = (v["box"][0] + v["box"][2]) / 2, (v["box"][1] + v["box"][3]) / 2
        rows, fit = view_read._read(sh, v["box"], lambda px, py: [round((px - cx) * k_, 1), round(-(py - cy) * k_, 1), 0.0], k_, MARK,
                                    f"D{lab}", 300, {}, detail=True)
        loc = []
        for r in rows:
            if not r.get("mark") or str(r["status"]).startswith("not a member"):
                continue
            if r["mark"] not in SCHED:
                asm.fate[r["id"]] = f"{r['mark']}: no section in the schedule"
                continue
            h_ = profile(SCHED[r["mark"]]).get("h", 200)
            loc.append({"id": r["id"], "ref": r["mark"], "prof": SCHED[r["mark"]], "a": [*r["a"][:2], -h_ / 2], "b": [*r["b"][:2], -h_ / 2],
                        "layer": "beam", "rank": 2, "ids": [r["id"]], "src": f"{DOC} p{det_sheet} {v['title'][:40]} (detail)".strip(),
                        "grid": lab, "reg": [r["id"]]})
        for r in rows:
            if not r.get("mark") and not str(r["status"]).startswith("not a member"):
                asm.fate[r["id"]] = "R02 unmarked line (no size given)"
        rects = []
        for pv in REG_P:
            if pv["status"] == "read" and pv["sheet"] != det_sheet:
                rects += components.detail_rects(SHEETS[pv["sheet"] - 1], pv["fit"]["T"], lab, pv["box"])
        rects = [r for i_, r in enumerate(rects) if all(math.dist(r[:2], q[:2]) > 300 for q in rects[:i_])]
        if loc and rects:
            cm, cr = components.details(loc, P0 | {m["prof"]: profile(m["prof"]) for m in loc}, rects, ztos, f"VF-{lab}", v["title"])
            for q in cm:
                q["ids"] = [q["ids"][0]] if q["key"].startswith(f"VF-{lab}.1:") else []   # drawn once; its copies count once
            COMP += cm
            comp_rep.append(cr)
        v.update(status="read", rows=rows, fit=fit, where=f"detail, placed by label {lab} x{len(rects)}")
        if not rects:
            v["status"] = f"needs placement: label {lab} not found inside a drawn X on a read plan"
(DATA_DIR / "components.json").write_text(json.dumps(comp_rep, indent=1))
for c in comp_rep:
    print(f"component {c['component']:<7} {c['view'][:44]:<45} members {c['members']:>3}, joints inside {c['joints inside']:>3}, "
          f"placed x{c['placements']}, defects {len(c['defects'])}")
NONT = [m for m in BASE if not m.get("tview")]
z_top = CC.get("top_default_mm") or max([z for m in BASE for z in (m["a"][2], m["b"][2])] or [z_bos + 6000])


def members_spec(r08):
    M = []
    for n, c in enumerate(cols):
        top = c["z1"] if c["z1"] is not None else z_top
        M.append({"id": f"COL{n + 1}", "ref": c["mark"], "prof": SCHED[c["mark"]], "a": [c["x"], c["y"], c["z0"]], "b": [c["x"], c["y"], top],
                  "layer": "col", "rank": 0, "up": [1, 0, 0] if c["depth_dir"] == "x" else [0, 1, 0], "ends": [None, None], "top_read": c["z1"] is not None,
                  "src": c["src"], "grid": f"{c['x']}/{c['y']}", "ids": c.get("ids", [])})
    gone = {m["id"] for m in NONT + COMP if m["key"] in r08}
    for m in NONT + COMP:
        if m["key"] in r08:
            for i in m.get("ids") or []:
                asm.fate[i] = "R08 B: an end meets nothing read"
            continue
        q = copy.deepcopy(m)
        for k in (0, 1):                                    # joined inside its component to a member left out: free again
            e = (q.get("ends") or [None, None])[k]
            if isinstance(e, list) and any(c.get("host") in gone for c in e):
                q["ends"][k] = None
                q["fixed"] = [f_ and kk != k for kk, f_ in enumerate(q.get("fixed") or [False, False])]
        M.append(q)
    return M


r08 = set(json.loads(R08.read_text(encoding="utf-8")))
reg = {}
for V in (REG_P, REG_E):
    for v in V:
        for r in v["rows"]:
            reg[r["id"]] = (v, r)
for rnd in range(5):
    M = members_spec(r08)
    if not M:
        sys.exit("no member read: check job.json (marks pattern, grids sheet, first_arch_sheet) and Models/_work/coverage.md")
    profiles = {m["prof"]: profile(m["prof"]) for m in M}
    rep = frame.connect(M, profiles, rulings, W)
    bad = frame.r08_leave_out(rep["no_support"], rulings.choice("R08"))
    if rnd == 0:
        (DATA_DIR / "no_support_round1.json").write_text(json.dumps([{"id": m["id"], "ref": m["ref"], "layer": m["layer"], "end": e,
            "a": m["a"], "b": m["b"], "src": m["src"][:120]} for m, e in rep["no_support"]], indent=0))
    for m, _ in bad:
        r08.add(m["key"])
    for m, e in [(m, e) for m, e in rep["no_support"] if m.get("key")]:
        vr = reg.get((m.get("reg") or [None])[0])
        if vr:
            v, r = vr
            xs, ys = [r["a_pt"][0], r["b_pt"][0]], [r["a_pt"][1], r["b_pt"][1]]
            rulings.case(W, "R08", v["sheet"], [min(xs) - 90, min(ys) - 90, max(xs) + 90, max(ys) + 90], {
                "A": [{"line": [*r["a_pt"], *r["b_pt"]], "c": (40, 150, 40), "text": f"{m['ref']} run on to the nearest support"}],
                "B": [{"cross": [(xs[0] + xs[1]) / 2, (ys[0] + ys[1]) / 2], "c": (120, 120, 120)}],
                "C": [{"line": [*r["a_pt"], *r["b_pt"]], "c": (230, 120, 0), "text": f"{m['ref']} cantilever"}],
                "D": [{"line": [*r["a_pt"], *r["b_pt"]], "c": (230, 120, 0), "text": f"{m['ref']} kept, end {e} floating"}]},
                note=f"{r['id']} {m['ref']} ({v['title'][:40]}); end {e} meets nothing")
    for m, e, dist in rep.get("overhangs", []) if rnd == 0 else []:
        vr = reg.get((m.get("reg") or [None])[0])
        if vr:
            v, r = vr
            xs, ys = [r["a_pt"][0], r["b_pt"][0]], [r["a_pt"][1], r["b_pt"][1]]
            rulings.case(W, "R10", v["sheet"], [min(xs) - 60, min(ys) - 60, max(xs) + 60, max(ys) + 60], {
                "A": [{"line": [*r["a_pt"], *r["b_pt"]], "c": (40, 150, 40), "text": f"{m['ref']} overhang {dist} mm"}],
                "B": [{"cross": [(xs[0] + xs[1]) / 2, (ys[0] + ys[1]) / 2], "c": (120, 120, 120)}]},
                note=f"{r['id']} {m['ref']} ({v['title'][:40]}); end {e} {dist} mm past its support", key=f"R10:{v['sheet']}:{m['ref']}:{len(rep.get('overhangs', [])) % 5}")
    print(f"frame round {rnd + 1}: overhangs {len(rep.get('overhangs', []))}, split {rep['split']}, ends joined {rep['joined']}, column tops {rep['col_tops']}, "
          f"ends meeting nothing {len(rep['no_support'])} ({len(bad)} members left out, R08)")
    if not bad:
        break
R08.write_text(json.dumps(sorted(r08)))
for i in rep.get("stub_rows", []):
    asm.fate.setdefault(i, "stub < 150 mm inside a joint (dropped)")
for m, e in rep["no_support"]:                              # what stays: first-read columns etc. -> free end, stated (R08 C)
    if m.get("key"):
        continue
    k = 1 if e in ("b", "top") else 0
    f_ = list(m.get("free") or [False, False]); f_[k] = True
    m["free"], m["free_why"] = f_, "R08 C: nothing read meets this end (column top with nothing read on it)"
    if e == "top":
        m["b"] = [m["b"][0], m["b"][1], round(z_bos + 3000)]
    rulings.case(W, "R08", readings={"A": f"{m['ref']} at {m['grid']}: run up to the nearest member", "B": "leave the column out",
                                     "C": "keep it, top 3 m above its base (free top)"}, note=f"{m['id']} {m['ref']} {m['grid']}", key=f"R08:{m['id']}")
ALIVE = {m["id"] for m in M}                                # a host left out (R08) or dropped: that end is no longer joined to it
for m in M:
    for k in (0, 1):
        e = (m.get("ends") or [None, None])[k]
        if isinstance(e, list) and any(c.get("host") not in ALIVE for c in e):
            m["ends"][k] = None
spec = {"connect": False, "profiles": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in profiles.items()},
        "members": [{k: v for k, v in m.items() if k not in ("top_read",)} for m in M],
        "_profmeta": {k: {kk: vv for kk, vv in v.items() if kk.startswith("_")} for k, v in profiles.items()}}
SPEC.write_text(json.dumps(spec))
run(PY, ROOT / "generate-model/geometry.py", SPEC, PARTS)

# ---------- data ----------
lo = dict(asm.left)
lo.update({k[6:]: v for k, v in asm.n.items() if k.startswith("left: ")})
lo["R08 members left out"] = len(r08)
lo["views not placed"] = sorted({f"p{v['sheet']} {v['title'][:40]} ({v['status'][:60]})" for v in REG_P + REG_E if v["status"] != "read"})
(DATA_DIR / "left_out.json").write_text(json.dumps(lo, indent=1))
NOT_HERE = [tuple(x) for x in CFG.get("not_this_building") or []]      # (title text, why): a view that is not this building
cc = reconcile.run(REG_P + REG_E, asm.fate, M, DATA_DIR / "count_check.json", DATA_DIR / "count_check.md", NOT_HERE)
print(f"count check      {cc['verdict']}: drawn {cc['totals'].get('drawn', 0)}, modelled {cc['totals'].get('modelled', 0)}, "
      f"left out with a reason {cc['totals'].get('left', 0)}, lost {len(cc['lost'])} (per view: Models Data/count_check.md)")
(DATA_DIR / "rule_effects.json").write_text(json.dumps({k: round(v, 1) for k, v in asm.effect.items()}))
run(PY, ROOT / "generate-model/make_data.py", CFG_P, SPEC, PARTS, DATA_DIR / "left_out.json", W / "grids.json", DATA)
run(PY, ROOT / "generate-model/tests/clash_check.py", DATA, SPEC, out=DATA_DIR / "clash.txt")
run(PY, ROOT / "generate-model/build_model.py", DATA, PAGE)
run(PY, ROOT / "generate-model/rulings.py", "sheet", W)
D = json.loads(DATA.read_text(encoding="utf-8"))
c = (DATA_DIR / "clash.txt").read_text(encoding="utf-8").splitlines()
summary = {"parts": len(D["parts"]), "t": round(sum(p["kg"] for p in D["parts"]) / 1000, 1), "drawn": cc["totals"].get("drawn", 0),
           "modelled": cc["totals"].get("modelled", 0), "left": cc["totals"].get("left", 0), "lost": len(cc["lost"]),
           "page": PAGE.name, "clash": [l for l in c if l.startswith(("free ends", "host contacts", "overlaps", "floating"))]}
(DATA_DIR / "summary.json").write_text(json.dumps(summary, indent=1))
print(f"DONE {summary['parts']} parts, {summary['t']:,.1f} t; drawn {summary['drawn']} / modelled {summary['modelled']} / lost {summary['lost']}; "
      f"{c[0] if c else ''}; page {PAGE.relative_to(ROOT)}; rulings {(W / 'rulings/rulings.html').relative_to(ROOT)}")
