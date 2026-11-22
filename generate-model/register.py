#!/usr/bin/env python3
"""Register every plan view of a job from the drawing store (no PDF opened): one row per drawn steel line.

    python generate-model/register.py WORK --mark REGEX [--scale 75] [--datum 4.850]

WORK = jobs/<job>/Models/_work (needs store/, grids.json, coverage.json). Writes WORK/register/plans.json (rows per view +
the fit of each view) and prints one line per view. Cached: a view whose sheet hash and reader version did not change
is taken from the last run.
View box = the grid bubbles that belong to the title (each bubble goes to the nearest title below it) + a margin.
Z: the title's TOS level; a plan titled with a range or two levels gets z = null (the elevations decide: stage 4).
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import store  # noqa: E402
import view_read_store as view_read  # noqa: E402
import sheet_index  # noqa: E402
import re

READER = hashlib.sha1(Path(view_read.__file__).read_bytes() + Path(__file__).read_bytes()).hexdigest()[:10]   # reader + this register: either changes, reread


def plan_views(sheet, kind=r"\bPLAN\b"):
    """Titled plan views of a sheet with box and scale, from the shared indexer run on the store."""
    out = []
    for v in sheet_index.read_store_sheet(sheet)["views"]:
        if not v["titled"] or not re.search(kind, v["name"], re.I) or "ARCHITECTURAL" in v["name"]:
            continue
        b = v["box"]
        m = re.search(r"1\s*:\s*(\d+)", v.get("scale") or "")
        out.append({"title": v["name"], "box": b, "scale": float(m.group(1)) if m else None,
                    "tiny": (b[2] - b[0]) < 120 or (b[3] - b[1]) < 120,
                    "levels": [float(x) for x in view_read.LEVEL_RE.findall(v["name"])]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--mark", required=True)
    ap.add_argument("--scale", type=float, default=75)
    ap.add_argument("--datum", type=float, default=0.0)
    ap.add_argument("--first-arch", type=int, default=999)
    a = ap.parse_args()
    W = Path(a.work)
    G = json.loads((W / "grids.json").read_text(encoding="utf-8"))
    cov = json.loads((W / "coverage.json").read_text(encoding="utf-8"))
    out_p = W / "register" / "plans.json"
    out_p.parent.mkdir(exist_ok=True)
    old = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {"views": []}
    key_ = READER + ":" + hashlib.sha1(a.mark.encode()).hexdigest()[:6]   # the cache knows the reader AND the mark pattern
    cache = {}                                  # a view read once per grid it sits on (VIEW-R03) keeps every copy
    for v in old["views"]:
        if v.get("reader") == key_:
            cache.setdefault((v["sheet"], v["title"], tuple(v["box"])), []).append(v)
    sheets = {s.page: s for s in store.sheets(W / "store")}
    res, reused, sheet_views = [], 0, {}
    for pg in sorted({v["sheet"] for v in cov["views"] if v["kind"] == "plan" and v["sheet"] < a.first_arch}):
        sh = sheets[pg]
        sv = old.get("sheet_views", {}).get(str(pg))
        pv = sv["views"] if sv and sv["hash"] == sh.hash and sv.get("reader") == READER else plan_views(sh)
        sheet_views[str(pg)] = {"hash": sh.hash, "reader": READER, "views": pv}
        cv = [v for v in cov["views"] if v["sheet"] == pg]
        mvs = elev_views(sh, cv, set(), kinds=("plan",))
        done = set()
        for i, t in enumerate(pv):
            mv = next((m for m in mvs if m["title"].strip() == t["title"].strip()), None)
            if mv and t["levels"]:                          # the indexer's box must hold most of the title's own bubbles; else
                b_ = t["box"]                               # (title strip only, or a box of the views beside it) the box from them
                inside = sum(b_[0] <= x <= b_[2] and b_[1] <= y <= b_[3] for x, y in mv["marks"])
                if t["tiny"] or inside < len(mv["marks"]) / 2:
                    t = dict(t, box=mv["box"], tiny=False)
            if (t["title"], tuple(t["box"])) in done:
                continue
            done.add((t["title"], tuple(t["box"])))
            if t["tiny"] or not t["levels"]:
                continue                                    # a title strip split off its view / a plan with no level: indexer note
            c = cache.get((pg, t["title"], tuple(t["box"])))
            if c and all(v.get("hash") == sh.hash for v in c):
                res += c; reused += 1
                continue
            e = {"sheet": pg, "title": t["title"], "box": t["box"], "scale": t["scale"], "hash": sh.hash, "reader": key_}
            two = len(t["levels"]) == 2
            z = None if two else round((t["levels"][-1] - a.datum) * 1000)
            try:
                rows, fit = view_read.read_plan(sh, t["box"], G["x"], G["y"], z, a.mark, t["scale"] or a.scale, view=f"P{i + 1}")
                e.update(status="read", z=z, fit=fit, rows=rows)
            except Exception as ex:                         # recorded, never skipped silently
                e.update(status=f"FAILED {ex}"[:200], rows=[])
                anc = anchors(sh, t, cv, G) if "no grid bubble on one axis" in str(ex) else []
                for gname, py in anc:                       # VIEW-R03: no y bubble; its column row on each grid the sheet names
                    q = dict(e, where=f"column row on grid {gname} (VIEW-R03)")
                    try:
                        rows, fit = view_read.read_plan(sh, t["box"], G["x"], G["y"], z, a.mark, t["scale"] or a.scale,
                                                        view=f"P{i + 1}-{gname}", y_anchor=(py, G["y"][gname]))
                        q.update(status="read", z=z, fit=fit, rows=rows, anchor=gname)
                    except Exception as ex2:
                        q.update(status=f"FAILED {ex2}"[:200], rows=[])
                    res.append(q)
                if anc:
                    continue
            res.append(e)
    out_p.write_text(json.dumps({"reader": READER, "sheet_views": sheet_views, "views": res}, ensure_ascii=False), encoding="utf-8")
    n = sum(len(v["rows"]) for v in res)
    lab = sum(1 for v in res for r in v["rows"] if r.get("mark"))
    print(f"plan views {len(res)} ({reused} unchanged from cache): {sum(v['status'] == 'read' for v in res)} read, lines {n}, with a mark {lab}")
    ev = elevations(a, W, G, cov, sheets, res)
    n = sum(len(v["rows"]) for v in ev)
    lab = sum(1 for v in ev for r in v["rows"] if r.get("mark"))
    print(f"elevation / section views {len(ev)}: {sum(v['status'] == 'read' for v in ev)} read, lines {n}, with a mark {lab}")
    for v in res + ev:
        if v["status"] != "read":
            print(f"  NOT READ p{v['sheet']} {v['title'][:44]}: {v['status'][:90]}")
    print(f"-> {out_p.parent}")


def elev_views(sheet, titles, names, kinds=("elevation", "section")):
    """Elevation / section views of a sheet: each grid bubble and level marker goes to the nearest title below it
    (side-by-side views are told apart this way; the indexer merges them), box = those marks + a margin."""
    T = [t for t in titles if t["kind"] in ("elevation", "section", "plan")]   # ("ROOF TRUSS D1" labels inside a view are not titles)
    if not T:
        return []
    marks = [(b[0], b[1], 5) for b in view_read.bubbles(sheet, [0, 0, *sheet.size])          # a bubble: its title is below it
             if re.search(r"[A-Z0-9]", b[2]) and not re.fullmatch(r"S\d+[a-z]?", b[2])]           # ("S3 / -" title tags: not grid marks)
    marks += [((w[0] + w[2]) / 2, (w[1] + w[3]) / 2, -25) for w in sheet.words if view_read.LEVEL_RE.search(w[4]) and w[6] < 12.5]
    own = {i: [] for i in range(len(T))}
    for x, y, below in marks:
        c = [(abs(t["at"][0] - x) + 0.5 * (t["at"][1] - y), i) for i, t in enumerate(T) if t["at"][1] > y + below]
        if c:
            own[min(c)[1]].append((x, y))
    out = []
    for i, t in enumerate(T):
        if t["kind"] not in ("elevation", "section", "plan") or len(own[i]) < 2:
            continue
        xs, ys = [p[0] for p in own[i]], [p[1] for p in own[i]]
        sc = [w for w in sheet.words if re.search(r"SCALE\s*1\s*:\s*\d+", w[4]) and 0 <= w[1] - t["at"][1] < 30 and abs((w[0] + w[2]) / 2 - t["at"][0]) < 200]
        m = re.search(r"1\s*:\s*(\d+)", sc[0][4]) if sc else None
        out.append({"title": t["title"], "core": [min(xs), min(ys), max(xs), max(ys)], "scale": float(m.group(1)) if m else None,
                    "title_at": t["at"], "kind": t["kind"], "marks": own[i]})
    # widen each core (steel runs past the outer grids and above the top level marker), but stop at the middle of the
    # gap to a neighbouring view's core and inside the sheet frame
    W, H = sheet.size
    # a view's extent runs from its marks down to its own title: a truss / section drawn with its grid bubbles on top hangs
    # below them, so stacked views are cut between extents, never between cores (p3 Turbine: the bracing truss B view was
    # cut at its bubbles' level, its bottom chord and depth went to the view below)
    ext = {id(v): (v["core"][1], max(v["core"][3], v["title_at"][1] - 8)) for v in out}
    for v in out:
        c = v["core"]
        ty = v["title_at"][1]                                # the view reaches down to its title (trusses hang below
        b = [max(20, c[0] - 200), max(20, c[1] - 260), min(W - 20, c[2] + 200),  # their bubbles), up past its top mark
             min(H - 20, max(c[3] + 20, ty - 8))]
        for o in out:
            if o is v:
                continue
            d = o["core"]
            if min(c[3], d[3]) > max(c[1], d[1]) - 40:          # side by side
                if d[0] >= c[2]:
                    b[2] = min(b[2], (c[2] + d[0]) / 2)
                elif d[2] <= c[0]:
                    b[0] = max(b[0], (d[2] + c[0]) / 2)
            if min(c[2], d[2]) > max(c[0], d[0]):                # stacked
                (c0, c1), (d0, d1) = ext[id(v)], ext[id(o)]
                if d0 >= c1:
                    b[3] = min(b[3], (c1 + d0) / 2)
                elif d1 <= c0:
                    b[1] = max(b[1], (d1 + c0) / 2)
        v["box"] = [round(x, 1) for x in b]
    return [v for v in out if v["kind"] in kinds]          # every view clips its neighbours, whatever kind is asked for


def _truss_names(txt):
    """'A2' -> [A2]; 'C1 & C5' -> [C1, C5]; 'D1 TO D5' -> [D1 .. D5]."""
    m = re.match(r"([A-Z])(\d)\s*TO\s*[A-Z]?(\d)", txt.strip(), re.I)
    if m:
        return [f"{m.group(1)}{i}" for i in range(int(m.group(2)), int(m.group(3)) + 1)]
    return [t.strip() for t in re.split(r"\s*(?:&|,)\s*", txt) if t.strip()]


_TL = {}


def truss_lines(sheets, name, axis, G, first_arch):
    """Every line of a roof plan labelled 'ROOF TRUSS <name>': the label's position across the truss (mm) on a calibrated
    roof plan. axis 'y' (truss elevation runs along Y) -> fixed x; 'x' -> fixed y. Cached per name."""
    key = (name, axis)
    if key in _TL:
        return _TL[key]
    out = []
    for sh in sheets.values():
        if sh.page >= first_arch:
            continue
        lab = [w for w in sh.words if re.fullmatch(rf"ROOF\s+TRUSS\s+{re.escape(name)}", w[4].strip(), re.I)]
        if not lab:
            continue
        for v in plan_views(sh, r"ROOF\s+PLAN"):
            if "BOTTOM" in v["title"].upper():
                continue                                   # the bottom-chord plan repeats the labels
            try:
                P, fit = view_read.calibrate(sh, v["box"], G["x"], G["y"], v["scale"] or 150)
            except ValueError:
                continue
            for w in lab:
                cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
                if not (v["box"][0] <= cx <= v["box"][2] and v["box"][1] <= cy <= v["box"][3]):
                    continue
                x, y = P(cx, cy)
                vert = w[5] % 180 == 90                     # a vertical label lies along a truss running along Y
                if (axis == "y") == vert:
                    c = x if axis == "y" else y
                    tab = G["x"] if axis == "y" else G["y"]
                    g = min(tab.items(), key=lambda t: abs(t[1] - c))
                    at = g[1] if abs(g[1] - c) < 400 else round(c / 50) * 50     # on its grid when the label sits by one
                    if all(abs(at - o) > 600 for o, _ in out):
                        out.append((at, f"p{sh.page} label at {'x' if axis == 'y' else 'y'} {round(c)}"))
    _TL[key] = out
    return out


def anchors(sh, t, cv, G):
    """A plan with x bubbles only (p22 crane access platform): its drawn column row (filled column symbols, median py)
    lies on a y grid named by another view title on the sheet ("... ALONG GRID LINE 2 & 4") -> [(grid, py)], one per grid."""
    cols = view_read.columns(sh, t["box"])
    if len(cols) < 2:
        return []
    py = sorted(c[1] for c in cols)[len(cols) // 2]
    names = []
    for v in cv:
        names += [g for g in _named(v["title"], {view_read.grid_name(k) for k in G["y"]}) if g not in names]
    return [(next(k for k in G["y"] if view_read.grid_name(k) == g), py) for g in names]


def _row_axis(bx, GX, GY):
    """An elevation / section shows its grid bubbles in one row across its top: the table of the row with most bubbles."""
    rows = {}
    for x, y, n in bx:
        rows.setdefault(round(y / 8), []).append(n)
    best = max(rows.values(), key=len, default=[])
    best = max(rows.values(), key=lambda r: max(sum(n in GX for n in r), sum(n in GY for n in r)), default=[])
    nx, ny = sum(n in GX for n in best), sum(n in GY for n in best)
    if not best:
        nx, ny = sum(b[2] in GX for b in bx), sum(b[2] in GY for b in bx)
    return ("x", GX) if nx >= ny else ("y", GY)


def derived_grids(plans, sheets, G):
    """Grid lines drawn as bubbles on a read plan but missing from the grid table (2S on p11): their position from that
    plan's calibration. Two bubbles of one name on one row -> a y grid; in one column -> an x grid."""
    known = {view_read.grid_name(k) for k in list(G["x"]) + list(G["y"])}
    found = {}
    for v in plans:
        if v["status"] != "read" or "VIEW-R03" in str(v["fit"].get("method")):
            continue                                        # not from a plan put on a grid by its column row
        sx, ox, sy, oy = v["fit"]["T"]
        b = [q for q in view_read.bubbles(sheets[v["sheet"]], v["box"]) if q[2] not in known
             and re.fullmatch(r"\d[A-Z]{1,2}|[A-Z]\d[A-Z]?", q[2]) and not re.fullmatch(r"S\d+[a-z]?", q[2])]
        allb = view_read.bubbles(sheets[v["sheet"]], v["box"])
        ky = [p for p in allb if p[2] in {view_read.grid_name(k) for k in G["y"]}]
        kx = [p for p in allb if p[2] in {view_read.grid_name(k) for k in G["x"]}]
        for n in {q[2] for q in b}:
            q = [p for p in b if p[2] == n]
            if (len(q) >= 2 and max(p[1] for p in q) - min(p[1] for p in q) < 4) or any(abs(p[0] - q[0][0]) < 3 for p in ky):
                found.setdefault(("y", n), []).append(round(sy * q[0][1] + oy))     # in the column of the y-grid bubbles
            elif (len(q) >= 2 and max(p[0] for p in q) - min(p[0] for p in q) < 4) or any(abs(p[1] - q[0][1]) < 3 for p in kx):
                found.setdefault(("x", n), []).append(round(sx * q[0][0] + ox))
    for k, v in found.items():
        print(f"  grid {k[1]} ({k[0]}) from plan bubbles: {sorted(v)}")
    return {k: sorted(v)[len(v) // 2] for k, v in found.items()}


def _named(title, names):
    m = re.search(r"(?:GRID(?:\s+LINE)?|\bG\.\s*L\.?)\s*-?\s*(.+?)(?:\(|$)", title, re.I)   # G.L = grid line
    if not m:
        return []
    toks = [view_read.grid_name(t) for t in re.split(r"\s*(?:,|&|AND|TO)\s*", m.group(1).strip())]
    return [t for t in toks if t in names]


def elevations(a, W, G, cov, sheets, plans):
    """Elevations along grid lines (one view may stand for several grids: 'ALONG GRID LINE C, D, E, F') and sections
    (fixed coordinate from their cut marks on the plan: one instance per cut location)."""
    G = {"x": dict(G["x"]), "y": dict(G["y"])}
    der = derived_grids(plans, sheets, G)
    for (ax, n), v in der.items():
        G[ax][n] = v
    if der:
        print("  grids from plan bubbles (not on p1): " + ", ".join(f"{n} {ax} {v}" for (ax, n), v in der.items()))
    GX = {view_read.grid_name(k): v for k, v in G["x"].items()}
    GY = {view_read.grid_name(k): v for k, v in G["y"].items()}
    secs = {(r["sheet"], r["section"]): r for r in cov["sections"]}
    out_p = W / "register" / "elevations.json"
    out = []
    pages = sorted({v["sheet"] for v in cov["views"] if v["kind"] in ("elevation", "section") and v["sheet"] < a.first_arch})
    for pg in pages:
        sh = sheets[pg]
        for i, t in enumerate(elev_views(sh, [v for v in cov["views"] if v["sheet"] == pg], set(GX) | set(GY))):
            title = t["title"]
            jobs = []                                       # (axis, H, at, where)
            truss = re.search(r"ROOF\s+TRUSS\s+([A-Z]\d?(?:\s*(?:&|,|TO)\s*[A-Z]?\d)*)", title, re.I)
            named = _named(title, set(GX) | set(GY))
            if truss and not named:                        # a truss drawn once for its type: placed on every roof-plan line
                bx = view_read.bubbles(sh, t["box"])        # labelled with that type (ROOF TRUSS A2 ...)
                nx, ny = sum(b[2] in GX for b in bx), sum(b[2] in GY for b in bx)
                axis, H = ("x", GX) if nx > ny else ("y", GY)
                for tn in _truss_names(truss.group(1)):
                    for at, src in truss_lines(sheets, tn, axis, G, a.first_arch):
                        if any(abs(at - j[2]) < 600 for j in jobs):
                            continue                        # one line labelled with several bays (B1 ... B5): one placement
                        jobs.append((axis, H, at, f"ROOF TRUSS {tn} {src}"))
                if not jobs:
                    out.append({"sheet": pg, "title": title, "box": t["box"], "status": "needs placement: no roof-plan line labelled with this truss", "rows": []})
                    continue
            elif not re.match(r"SECTION", title, re.I) or named:      # "SECTION FOR GRID 3": on that grid
                for g in _named(title, set(GX) | set(GY)):
                    jobs.append(("y", GY, GX[g], f"grid {g}") if g in GX else ("x", GX, GY[g], f"grid {g}"))
            else:
                nm_ = re.search(r"SECTION\s+(\S+)", title, re.I)
                if not nm_:                                 # a bare "SECTION" title (no name, so no cut mark to find)
                    out.append({"sheet": pg, "title": title, "box": t["box"], "status": "needs placement: section with no name", "rows": []})
                    continue
                nm = nm_.group(1)
                sec = secs.get((pg, nm))
                if not sec or not sec["placed"]:
                    out.append({"sheet": pg, "title": title, "box": t["box"], "status": "needs placement: no cut mark found", "rows": []})
                    continue
                # which axis does the section run along: the grid table with more bubbles in the view
                axis, H = _row_axis(view_read.bubbles(sh, t["box"]), GX, GY)
                spg = int(sec["cut_marks_on"][0][1:])
                locs = []
                pts = sec["cut_xy_pt"]
                # a pair of marks lined up the other way is another cut of this name (its other direction): not this one
                # (exactly two marks lined up = one cut line; a longer row of marks = the section repeated (TYP), kept)
                def _line(i):
                    x0, y0 = pts[i]
                    return [j for j, (x1, y1) in enumerate(pts) if (abs(x0 - x1) < 6 if axis == "x" else abs(y0 - y1) < 6)]
                wrong = {i for i in range(len(pts)) if len(_line(i)) == 2 and
                         max(abs(pts[j][1 if axis == "x" else 0] - pts[i][1 if axis == "x" else 0]) for j in _line(i)) > 20}
                for px, py in [q for i, q in enumerate(pts) if i not in wrong]:
                    pv = min((v for v in plans if v["sheet"] == spg and v["status"] == "read"),   # in a read plan, or beside
                             key=lambda v: max(v["box"][0] - px, px - v["box"][2], 0) + max(v["box"][1] - py, py - v["box"][3], 0), default=None)
                    if pv and max(pv["box"][0] - px, px - pv["box"][2], pv["box"][1] - py, py - pv["box"][3]) > 200:
                        pv = None                                  # (cut marks drawn just outside the plan's box)
                    if pv:
                        sx, ox, sy, oy = pv["fit"]["T"]
                        locs.append(round(sy * py + oy) if axis == "x" else round(sx * px + ox))
                groups = []
                for c in sorted(locs):
                    if groups and c - groups[-1][-1] < 600:
                        groups[-1].append(c)
                    else:
                        groups.append([c])
                if not groups:
                    out.append({"sheet": pg, "title": title, "box": t["box"], "status": f"needs placement: cut marks on p{spg} lie in no read plan", "rows": []})
                    continue
                ext = list((GY if axis == "x" else GX).values())    # a cut placed off the grids: refused, listed
                off = [round(sum(gq) / len(gq)) for gq in groups if not min(ext) - 3000 <= sum(gq) / len(gq) <= max(ext) + 3000]
                groups = [gq for gq in groups if min(ext) - 3000 <= sum(gq) / len(gq) <= max(ext) + 3000]
                if not groups:
                    out.append({"sheet": pg, "title": title, "box": t["box"], "status": f"needs placement: cut mark placed off the grids {off}", "rows": []})
                    continue
                for gq in groups:
                    jobs.append((axis, H, round(sum(gq) / len(gq)), f"cut on p{spg} at {'y' if axis == 'x' else 'x'} = {round(sum(gq) / len(gq))}"))
            if not jobs:
                out.append({"sheet": pg, "title": title, "box": t["box"], "status": "needs placement: no grid named", "rows": []})
                continue
            for axis, H, at, where in jobs:
                e = {"sheet": pg, "title": title, "box": t["box"], "scale": t["scale"], "where": where, "axis": axis, "at": at}
                try:
                    rows, fit = view_read.read_elev(sh, t["box"], H, axis, at, a.mark, t["scale"] or 150, a.datum,
                                                    view=f"E{i + 1}{'' if len(jobs) == 1 else '-' + where.split()[-1]}",
                                                    scale_read=bool(t["scale"]), no_level="relative" if truss else "raise")
                    e.update(status="read", fit=fit, rows=rows)
                except Exception as ex:
                    e.update(status=f"FAILED {ex}"[:200], rows=[])
                out.append(e)
    out_p.write_text(json.dumps({"reader": READER, "views": out}, ensure_ascii=False), encoding="utf-8")
    for v in out:
        f = v.get("fit", {})
        if v["status"] == "read":
            print(f"  p{v['sheet']:<3}{v['title'][:34]:<35}{v['where'][:26]:<27} lines {len(v['rows']):>4} unl {f['unlabelled']:>3} rms {f['rms_mm']} scale {f['scale_err_pct']}% facing {f['facing']}")
    return out


if __name__ == "__main__":
    main()
