#!/usr/bin/env python3
"""Golden set: every drawing-reading rule the curator taught has a fixture cut from a real drawing store + the correct reading.
Every change to the readers must keep all of them passing.

    python generate-model/tests/golden_check.py              run every fixture (exit 1 on any failure)
    python generate-model/tests/golden_check.py cut STORE KEY PAGE --box x0 y0 x1 y1 --name NAME
                                                             cut a sheet region into tests/golden/NAME (a mini store)

A fixture = tests/golden/NAME/ (KEY_pNN.npz / .json: the region of the stored sheet, same format as store.py, so
store.Sheet loads it) + NAME/case.json:
  {"rule": "BOX-R01", "says": "the rule in words", "source": "job, sheet, view",
   "call": {"fn": "read" | "lines" | "assemble", "box": [...], "T": [sh, oh, sz, oz], "axis": "x"|"y"|"plan", "at": mm,
            "k": mm/pt, "mark_re": "...", "view": "E1", "min_mm": 500, "sched": {mark: kg/m}},
   "when": {"R12": "A"},                       optional: the expectation holds under these rulings only (else N/A)
   "expect": [{"near": [x, y], "kind": "line", "mark": "B3a"},       a row through that sheet point has that mark
              {"near": [x, y], "status": "not a member: label box"},  ... or a status starting so
              {"no_line": [[x0, y0], [x1, y1]]},                     no row lies along that segment
              {"none_status": "not a member: label box"},            no row has a status starting so
              {"fit": {"label_boxes": [2, 99]}}]}                    a reader count within [lo, hi]
"""
import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GOLD = HERE / "golden"
sys.path.insert(0, str(HERE.parent))
import store  # noqa: E402
import view_read_store as V  # noqa: E402


def cut(store_dir, key, page, box, name):
    sh = store.Sheet(store_dir, key, page)
    x0, y0, x1, y1 = box
    S = sh.segs_all
    m = (S[:, 0] >= x0) & (S[:, 2] >= x0) & (S[:, 0] <= x1) & (S[:, 2] <= x1) & (S[:, 1] >= y0) & (S[:, 3] >= y0) & (S[:, 1] <= y1) & (S[:, 3] <= y1)
    inb = lambda a, i0, i1: (a[:, i0] >= x0) & (a[:, i0] <= x1) & (a[:, i1] >= y0) & (a[:, i1] <= y1)
    B = sh.bez
    bm = inb(B, 0, 1) & inb(B, 6, 7) if len(B) else np.zeros(0, bool)
    F = sh.fill
    fm = inb(F, 0, 1) & inb(F, 2, 3) if len(F) else np.zeros(0, bool)
    Q = sh.quad
    qm = inb(Q, 0, 1) & inb(Q, 2, 3) if len(Q) else np.zeros(0, bool)
    out = GOLD / name
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{key}_p{page:02d}.npz", segs=S[m], curve=sh.curve_all[m], bez=B[bm] if len(B) else B, fill=F[fm] if len(F) else F,
                        quad=Q[qm] if len(Q) else Q, vis=sh.vis[m], lay=sh.lay[m], dpat=sh.dpat[m])
    words = [w for w in sh.words if x0 <= (w[0] + w[2]) / 2 <= x1 and y0 <= (w[1] + w[3]) / 2 <= y1]
    (out / f"{key}_p{page:02d}.json").write_text(json.dumps({"size": sh.size, "words": words, "hash": "golden", "layers": sh.layers,
                                                            "dashes": sh.dashes}), encoding="utf-8")
    print(f"cut {name}: p{page} {box}: {int(m.sum())} pieces ({int((sh.vis[m] < store.VIS_MIN).sum())} hidden), {len(words)} words, "
          f"{int(bm.sum())} curves, {int(fm.sum())} fills, {int(qm.sum())} quads -> {out}")


def _sheet(d):
    f = next(d.glob("*_p*.npz"))
    key, n = f.stem.rsplit("_p", 1)
    return store.Sheet(d, key, int(n))


def _rows(sh, c):
    box = c["box"]
    if c["fn"] == "lines":
        return [{"a_pt": list(l[:2]), "b_pt": list(l[2:4]), "kind": "line", "status": "piece", "mark": None} for l in V.lines(sh, box)], {}
    sh_, oh, sz, oz = c["T"]
    if c.get("axis") == "plan":
        P3 = lambda px, py: [round(sh_ * px + oh, 1), round(sz * py + oz, 1), c.get("at", 0)]
    elif c.get("axis") == "y":
        P3 = lambda px, py: [c["at"], round(sh_ * px + oh, 1), round(sz * py + oz, 1)]
    else:
        P3 = lambda px, py: [round(sh_ * px + oh, 1), c["at"], round(sz * py + oz, 1)]
    rows, fit = V._read(sh, box, P3, c["k"], c["mark_re"], c.get("view", "G"), c.get("min_mm", 500), {})
    if c["fn"] == "assemble":
        import assemble
        import rulings
        kg = c.get("sched", {})
        with tempfile.TemporaryDirectory() as td:
            asm = assemble.Assembler(td, {m: m for m in kg}, lambda s: {"kgm": kg.get(s, 0.0), "h": 300}, rulings, [])
            view = {"sheet": sh.page, "title": c.get("view", "G"), "rows": rows}
            asm.unmarked(view, "elev" if c.get("axis") != "plan" else "plan", False)
            fit = {**fit, **{f"rule {k}": v for k, v in asm.n.items()}}
    return rows, fit


def _place(sh, c):
    """VIEW-R05: the sheet's view boxes (register.elev_views from the coverage titles), then each no-level view's depth,
    named host trusses and station on them."""
    import assemble
    import register
    boxes = {}
    for v in register.elev_views(sh, c["titles"], set()):
        boxes.setdefault(v["title"], v["box"])
    re_ = __import__("re")
    top, bot = re_.compile(r"^TC1?a?$"), re_.compile(r"^BC1?a?$")
    bad = []
    for w in c["views"]:
        box = boxes.get(w["title"])
        if box is None:
            bad.append(f"no view box for {w['title']}")
            continue
        inb = lambda p: box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3]
        bad += [f"{w['title'][:30]}: box {box} misses {p}" for p in w.get("want_in_box", []) if not inb(p)]
        bad += [f"{w['title'][:30]}: box {box} takes {p} (the view above)" for p in w.get("not_in_box", []) if inb(p)]
        dep, how = assemble.view_depth(sh.words, box)
        if dep != w["want_depth"]:
            bad.append(f"{w['title'][:30]}: depth {dep} {how} (want {w['want_depth']})")
            continue
        hv, names = assemble.named_hosts(sh.words, box, c["hosts"], w["title"])
        f = 1 if w["axis"] == "x" else 0
        hosts = [{"top": [r for r in h["rows"] if r["mark"] and top.match(r["mark"])], "bot": [r for r in h["rows"] if r["mark"] and bot.match(r["mark"])],
                  "posts": sorted({r["a"][f] for r in h["rows"] if abs(r["a"][2] - r["b"][2]) > 300})} for h in hv]
        s, hd = assemble.station_by_depth(dep, hosts, w["axis"], w["guess"])
        if s is None or abs(s - w["want_at"]) > 150:
            bad.append(f"{w['title'][:30]} (label at {w['guess']}): placed at {s} (want {w['want_at']}; hosts {names})")
    return ("ok" if not bad else "FAIL"), bad


def _d(p, a, b):
    return V._seg_dist(p, a, b)


def check(d):
    case = json.loads((d / "case.json").read_text(encoding="utf-8"))
    import rulings
    for rid, letter in (case.get("when") or {}).items():
        if rulings.choice(rid) != letter:
            return "n/a", [f"{rid} is {rulings.choice(rid)} (fixture written for {letter})"]
    sh = _sheet(d)
    if case["call"]["fn"] == "place":
        return _place(sh, case["call"])
    rows, fit = _rows(sh, case["call"])
    bad = []
    for e in case["expect"]:
        if "near" in e:
            cand = [r for r in rows if (not e.get("kind") or r["kind"] == e["kind"]) and _d(e["near"], r["a_pt"], r["b_pt"]) <= e.get("tol", 1.5)]
            if not cand:
                bad.append(f"no row through {e['near']}")
                continue
            r = min(cand, key=lambda r: _d(e["near"], r["a_pt"], r["b_pt"]))
            if "mark" in e and r.get("mark") != e["mark"]:
                bad.append(f"row at {e['near']}: mark {r.get('mark')} (want {e['mark']}; {r['status']})")
            if "status" in e and not str(r.get("status", "")).startswith(e["status"]):
                bad.append(f"row at {e['near']}: status {r.get('status')} (want {e['status']}...)")
        if "no_line" in e:
            (ax, ay), (bx, by) = e["no_line"]
            on = [r for r in rows if not str(r.get("status", "")).startswith("not a member")
                  and _d(r["a_pt"], (ax, ay), (bx, by)) < 1.5 and _d(r["b_pt"], (ax, ay), (bx, by)) < 1.5
                  and math.dist(r["a_pt"], r["b_pt"]) > 3]
            if on:
                bad.append(f"{len(on)} row(s) read along {e['no_line']} (e.g. {on[0]['a_pt']} - {on[0]['b_pt']})")
        if "none_status" in e:
            hit = [r for r in rows if str(r.get("status", "")).startswith(e["none_status"])]
            if hit:
                bad.append(f"{len(hit)} row(s) '{e['none_status']}' (e.g. {hit[0]['a_pt']} - {hit[0]['b_pt']})")
        if "splices" in e:                                   # the member through that point carries [lo, hi] splices
            r_ = min(rows, key=lambda r: _d(e["splices"]["near"], r["a_pt"], r["b_pt"]))
            n_ = len(r_.get("splices") or [])
            lo, hi = e["splices"]["count"]
            if not lo <= n_ <= hi:
                bad.append(f"member at {e['splices']['near']}: {n_} splice(s) (want {lo}-{hi}; runs {r_['a_pt']} - {r_['b_pt']})")
        if "legend" in e:                                    # {name: [lo, hi]} runs of that legend style read in the view
            got = fit.get("legend", {})
            for nm, (lo, hi) in e["legend"].items():
                v = got.get(nm, {}).get("runs", 0)
                if not lo <= v <= hi:
                    bad.append(f"legend {nm}: {v} runs (want {lo}-{hi})")
        if "fit" in e:
            for k, (lo, hi) in e["fit"].items():
                v = fit.get(k, 0)
                if not lo <= v <= hi:
                    bad.append(f"{k} = {v} (want {lo}-{hi})")
    return ("ok" if not bad else "FAIL"), bad


def r08_unit():
    """R08 D (precedent D26): only an isolated line is left out; one loose end stays."""
    import frame
    a, b, c = {"key": 1, "id": "isolated"}, {"key": 2, "id": "one end loose"}, {"id": "column"}
    out = {m["id"] for m, _ in frame.r08_leave_out([(a, "a"), (a, "b"), (b, "a"), (c, "top")], "D")}
    return ("ok", []) if out == {"isolated"} else ("FAIL", [f"R08 D left out {sorted(out)} (want ['isolated'])"])


def layer_unit():
    """LAYER-R01 (the curator 2026-11-22): the structure pass reads steel / unnamed layers only; the purlin pass purlin layers only."""
    names = ["", "STEEL", "Handrail", "DIM", "Purlin", "Civil_Grid", "0"]
    segs = np.zeros((len(names), 10), np.float32)
    for i in range(len(names)):
        segs[i, :5] = (50, 50 + 20 * i, 250, 50 + 20 * i, 0.8)            # black 0.8 pt lines, one per layer
    with tempfile.TemporaryDirectory() as td:
        np.savez_compressed(Path(td) / "u_p01.npz", segs=segs, curve=np.zeros(len(names), np.uint8), bez=np.zeros((0, 11), np.float32),
                            fill=np.zeros((0, 8), np.float32), quad=np.zeros((0, 5), np.float32), vis=np.full(len(names), 100, np.uint8),
                            lay=np.arange(len(names), dtype=np.uint16), dpat=np.zeros(len(names), np.uint16))
        (Path(td) / "u_p01.json").write_text(json.dumps({"size": [400, 300], "words": [], "hash": "u", "layers": names, "dashes": [""]}))
        sh = store.Sheet(td, "u", 1)
        got = sorted(names[round((l[1] - 50) / 20)] for l in V.lines(sh, (0, 0, 400, 300)))
        pur = sorted(names[round((float(s[1]) - 50) / 20)] for s in sh.pass_segs("purlin"))
    bad = [] if got == ["", "0", "STEEL"] else [f"structure pass read {got} (want ['', '0', 'STEEL'])"]
    bad += [] if pur == ["Purlin"] else [f"purlin pass read {pur} (want ['Purlin'])"]
    return ("ok" if not bad else "FAIL"), bad


def r08_run_on_unit():
    """R08 A (the curator 2026-11-22: a member never floats): a beam end stopping 3.5 m short of a column on its own line runs on to
    it; under D the same end is listed floating."""
    import frame

    class Pick:
        def __init__(self, x):
            self.x = x

        def choice(self, rid):
            return self.x if rid == "R08" else "A"
    prof = {"UC": {"type": "I", "h": 300, "b": 300}, "UB": {"type": "I", "h": 400, "b": 180}}
    out = {}
    for x in ("A", "D"):
        M = [{"id": "C1", "layer": "col", "prof": "UC", "a": [0, 0, 0], "b": [0, 0, 6000]},
             {"id": "C2", "layer": "col", "prof": "UC", "a": [9000, 0, 0], "b": [9000, 0, 6000]},
             {"id": "B1", "key": 1, "prof": "UB", "a": [3500, 0, 5000], "b": [9000, 0, 5000]}]
        rep = frame.connect(M, prof, Pick(x))
        b = next(m for m in M if m["id"] == "B1")
        out[x] = (round(b["a"][0]), [(m["id"], e) for m, e in rep["no_support"] if m.get("key")])
    bad = [] if out["A"] == (0, []) else [f"R08 A: B1 starts at x={out['A'][0]}, loose {out['A'][1]} (want x=0, none)"]
    bad += [] if out["D"][1] == [("B1", "a")] else [f"R08 D: loose {out['D'][1]} (want [('B1', 'a')])"]
    return ("ok" if not bad else "FAIL"), bad


def r02_similar_unit():
    """R02 A (the curator 2026-11-22): an unmarked member takes the mark of the similar member (drawn the same way, same drawn
    depth = scale, similar length), not merely the nearest parallel line."""
    import assemble
    row = lambda a, b, hw, ol, mk: {"a_pt": a, "b_pt": b, "hw": hw, "outline": ol, "mark": mk}
    marked = [row([0, 100], [200, 100], 3.0, True, "B1"),          # outline 6 pt deep, 200 pt long (far)
              row([0, 20], [60, 20], 0.4, False, "BR1"),            # single line, short (nearest)
              row([0, 300], [200, 300], 1.2, True, "B9")]           # outline of another depth
    r = row([0, 10], [190, 10], 3.1, True, None)
    nm = assemble.similar(r, marked)
    return ("ok", []) if nm and nm["mark"] == "B1" else ("FAIL", [f"R02 A picked {nm and nm['mark']} (want B1)"])


def column_foot_top_unit():
    """the curator C4.1 + COL-TOP: a column read from an elevation (not on the layout) with its foot at the base stands on its RC
    pedestal, and a column standing on its foot may end with a free top: neither end is floating."""
    import frame

    class Pick:
        def choice(self, rid):
            return "A"
    prof = {"UC": {"type": "I", "h": 300, "b": 300}}
    M = [{"id": "C1", "layer": "col", "prof": "UC", "a": [0, 0, 0], "b": [0, 0, 6000]},
         {"id": "P1", "key": 1, "layer": "post", "prof": "UC", "a": [9000, 0, 100], "b": [9000, 0, 5000]}]
    rep = frame.connect(M, prof, Pick())
    loose = [(m["id"], e) for m, e in rep["no_support"]]
    p1 = next(m for m in M if m["id"] == "P1")
    bad = [] if not loose and p1.get("free") == [True, True] and p1["b"][2] == 5000 else         [f"loose {loose}, P1 free {p1.get('free')}, top z {p1['b'][2]} (want none loose, both ends stated, top as drawn 5000)"]
    return ("ok" if not bad else "FAIL"), bad


def named_note_unit():
    """NAME-R02: the note a named member is written in gives its mark: a tag in the note, else a written section, else the
    name (sized later)."""
    W = [[100, 100, 160, 108, "MONORAIL & HOIST", 0, 8.8], [100, 110, 140, 118, "(1 TONS)", 0, 8.8],
         [100, 120, 170, 128, "UC152X152X37", 0, 8.8], [100, 130, 150, 138, "STUMP &", 0, 8.8], [100, 140, 170, 148, "MONORAIL BEAM", 0, 8.8],
         [300, 100, 312, 108, "B3", 0, 8.8], [300, 110, 370, 118, "MONORAIL BEAM", 0, 8.8], [500, 100, 550, 108, "MONORAIL &", 0, 8.8],
         [500, 110, 530, 118, "HOIST", 0, 8.8]]
    rx = "(SC[0-9]+|B[0-9]+[a-zA-Z]?|BR[0-9]+)"
    got = [(V.named_mark(W, W[i], rx) or [None])[0] for i in (0, 6, 7)]
    want = ["UC152X152X37", "B3", "MONORAIL"]
    return ("ok", []) if got == want else ("FAIL", [f"named marks {got} (want {want})"])


def leader_from_note_unit():
    """LEADER-R01 (the curator, p19 HANGER / B2d (TYP)): the arrow leaves from the note's first line (HANGER), the tag is two lines
    below: the tag still goes to the member the arrow points at."""
    segs = np.zeros((2, 10), np.float32)
    segs[0, :5] = (300, 50, 300, 250, 0.8)          # the hanger (member line)
    segs[1, :5] = (345, 156, 302, 120, 0.2)         # thin leader from the note's first line to the hanger
    words = [[340, 155, 380, 164, "HANGER", 0, 8.8], [340, 166, 356, 175, "B2d", 0, 8.8], [340, 177, 364, 186, "(TYP)", 0, 8.8]]
    with tempfile.TemporaryDirectory() as td:
        np.savez_compressed(Path(td) / "u_p01.npz", segs=segs, curve=np.zeros(2, np.uint8), bez=np.zeros((0, 11), np.float32),
                            fill=np.zeros((0, 8), np.float32), quad=np.zeros((0, 5), np.float32), vis=np.full(2, 100, np.uint8),
                            lay=np.zeros(2, np.uint16), dpat=np.zeros(2, np.uint16))
        (Path(td) / "u_p01.json").write_text(json.dumps({"size": [500, 300], "words": words, "hash": "u", "layers": [""], "dashes": [""]}))
        sh = store.Sheet(td, "u", 1)
        rows = [{"a_pt": [300.0, 50.0], "b_pt": [300.0, 250.0], "ang": 90.0, "hw": 0.4, "id": "r1"}]
        free = [(words[1], "B2d", False)]
        left = V.leaders(sh, (0, 0, 500, 300), free, rows)
    ok = rows[0].get("mark") == "B2d" and not left
    return ("ok", []) if ok else ("FAIL", [f"hanger mark {rows[0].get('mark')} (want B2d), still free {[m for _, m, _ in left]}"])


def mark_type_unit():
    """MARK-TYPE (the curator C2.6): on an elevation C4 labels an upright, BR1 a diagonal, B21 a level line - never the other way;
    a BR (TYP) labels every diagonal of its truss."""
    up, lev, dia = {"ang": 90.0}, {"ang": 0.0}, {"ang": 52.0}
    want = {("C4", "up"): True, ("C4", "lev"): False, ("BR1", "dia"): True, ("BR1", "lev"): False, ("BR1", "up"): False,
            ("B21", "lev"): True, ("B21", "up"): False, ("B21", "dia"): True}
    got = {k: V.type_ok(k[0], {"up": up, "lev": lev, "dia": dia}[k[1]]) for k in want}
    bad = [f"{k[0]} on a {k[1]} line: {got[k]} (want {want[k]})" for k in want if got[k] != want[k]]
    rows = [{"a_pt": [0, 0], "b_pt": [30, 40], "ang": 53.1, "mark": "BR1", "typ": True, "id": "a"},
            {"a_pt": [100, 0], "b_pt": [118, 40], "ang": 65.8, "mark": None},          # an end diagonal of another length / angle
            {"a_pt": [200, 0], "b_pt": [200, 40], "ang": 90.0, "mark": None}]          # a vertical: not a brace
    V.typ_fill(rows)
    if rows[1].get("mark") != "BR1" or rows[2].get("mark"):
        bad.append(f"TYP fill: diagonal {rows[1].get('mark')} (want BR1), vertical {rows[2].get('mark')} (want none)")
    return ("ok" if not bad else "FAIL"), bad


def visibility_unit():
    """HIDDEN-R01 at extraction: a line under a later white fill is 0 % shown, one half under it about 50 %, a line drawn
    after the fill (on top) and one outside the fill 100 %."""
    import pymupdf
    doc = pymupdf.open()
    pg = doc.new_page(width=400, height=300)
    pg.draw_line((20, 50), (180, 50), color=(0, 0, 0), width=0.8)          # under the mask
    pg.draw_line((100, 100), (300, 100), color=(0, 0, 0), width=0.8)       # half under it
    pg.draw_line((20, 250), (380, 250), color=(0, 0, 0), width=0.8)        # away from it
    pg.draw_rect(pymupdf.Rect(10, 20, 200, 120), color=None, fill=(1, 1, 1))
    pg.draw_line((20, 80), (180, 80), color=(0, 0, 0), width=0.8)          # on top of the mask
    data, meta = store.extract_page(pg)
    v = {round(float(s[1])): int(x) for s, x in zip(data["segs"], data["vis"])}
    want = {50: (0, 0), 100: (40, 60), 250: (100, 100), 80: (100, 100)}
    bad = [f"line y={y}: {v.get(y)} % shown (want {lo}-{hi})" for y, (lo, hi) in want.items() if not (v.get(y) is not None and lo <= v[y] <= hi)]
    doc.close()
    return ("ok" if not bad else "FAIL"), bad


def main():
    res = []
    for d in sorted(p for p in GOLD.iterdir() if (p / "case.json").exists()):
        case = json.loads((d / "case.json").read_text(encoding="utf-8"))
        st, why = check(d)
        res.append((d.name, case["rule"], st, why))
    st, why = r08_unit()
    res.append(("r08_isolated_only", "R08 D", st, why))
    st, why = visibility_unit()
    res.append(("white_mask_at_extraction", "HIDDEN-R01", st, why))
    for name, rule, fn in (("one_kind_per_pass", "LAYER-R01", layer_unit), ("loose_end_runs_on", "R08 A", r08_run_on_unit),
                           ("mark_from_similar_member", "R02 A", r02_similar_unit),
                           ("column_foot_and_free_top", "COL-TOP", column_foot_top_unit), ("note_gives_named_mark", "NAME-R02", named_note_unit),
                           ("leader_from_note_line", "LEADER-R01", leader_from_note_unit),
                           ("mark_letters_give_direction", "MARK-TYPE", mark_type_unit)):
        st, why = fn()
        res.append((name, rule, st, why))
    for name, rule, st, why in res:
        print(f"  {st:<4} {rule:<10} {name}" + ("" if not why else ": " + "; ".join(why)))
    n_fail = sum(1 for r in res if r[2] == "FAIL")
    print(f"golden_check: {'OK' if not n_fail else 'FAIL'} {len(res) - n_fail}/{len(res)} "
          f"({sum(1 for r in res if r[2] == 'n/a')} n/a: a ruling answered otherwise)")
    return n_fail


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cut":
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("store")
        ap.add_argument("key")
        ap.add_argument("page", type=int)
        ap.add_argument("--box", nargs=4, type=float, required=True)
        ap.add_argument("--name", required=True)
        a = ap.parse_args()
        cut(a.store, a.key, a.page, a.box, a.name)
    else:
        sys.exit(1 if main() else 0)
