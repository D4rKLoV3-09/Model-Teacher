#!/usr/bin/env python3
"""Components first (the curator): a truss or a detail is built once as its own model, its joints made and checked alone, then
copied to every place the drawings put it (1 or 100 times). A fix in the component reaches every copy.

    import components
    comp, rep = components.trusses(members, profiles)          # members with "tview" (truss view) + "tat" (placement)
    comp, rep = components.details(...)                         # a detail plan (no grid) placed by its label on the plans

Truss: the placement with most members is the master. frame.connect runs on the master alone (no column, no beam):
every web end must land on a chord inside the truss; an end that meets nothing inside is a component defect (listed).
Ends joined inside are "fixed" (the building frame leaves them); the outer ends (chord ends, bearing) are joined to the
building by frame.py. Every other placement = the master moved by the placement's own offset (median over its members),
taking the register ids of its own drawn members (count check). Members drawn at one placement only stay as drawn.
"""
import copy
import math
import re

import numpy as np

import frame

MATCH = 400.0     # mm: a placement member this close to the moved master member (same mark) is its copy


def _key(i):
    """register row id without its placement token: p4-E1-C-L001 -> p4-E1-L001."""
    return re.sub(r"^(p\d+-E\d+)-[^-]+-(L\d+)$", r"\1-\2", i)


def solve(ms, profiles, cid):
    """Joints inside one component. -> (members, inner defects, joints)."""
    comp = copy.deepcopy(ms)
    for k, m in enumerate(comp):
        m["_src_id"], m["id"] = m["id"], f"{cid}-{k + 1}"
        m["ends"] = None
        m.pop("free", None), m.pop("free_why", None)
        m["comp"] = cid
    rep = frame.connect(comp, profiles)
    open_ = {(id(m), e) for m, e in rep["no_support"]}
    lo, hi = None, None
    P = np.array([p for m in comp for p in (m["a"], m["b"])], float)
    ax = 0 if np.ptp(P[:, 0]) >= np.ptp(P[:, 1]) else 1
    lo, hi = P[:, ax].min(), P[:, ax].max()
    defects = []
    for m in comp:
        fx = []
        for k, e in enumerate("ab"):
            p = m["a"] if k == 0 else m["b"]
            outer = min(abs(p[ax] - lo), abs(p[ax] - hi)) < 400        # at the truss ends: bears on the building
            joined = (id(m), e) not in open_ and not (m.get("free") or [False, False])[k]
            fx.append(joined)
            if not joined and not outer and m.get("layer") != "chord":   # a chord end bears on the building (columns)
                defects.append(f"{m['ref']} {m['id']} end {e} meets nothing inside the component")
        m["fixed"] = fx
    return comp, defects, rep["joined"]


def trusses(members, profiles):
    groups = {}
    for m in members:
        if m.get("tview"):
            groups.setdefault((m["tview"], tuple(m["tat"] or ("", 0))), []).append(m)
    types = {}
    for (tv, at), ms in groups.items():
        types.setdefault(tv, []).append((at, ms))
    out = [m for m in members if not m.get("tview")]
    report = []
    for n, (tv, pls) in enumerate(sorted(types.items())):
        pls.sort(key=lambda p: -len(p[1]))
        at0, master = pls[0]
        cid = f"K{n + 1}"
        comp, defects, joints = solve(master, profiles, cid)
        mk = {}
        for m in comp:
            for i in m.get("ids") or []:
                mk.setdefault(_key(i), m)
        placed, extra, missing = 0, 0, 0
        for j, (at, ms) in enumerate(pls):
            pairs = []                                          # placement member -> master member (by its drawn row)
            for m in ms:
                c = next((mk[_key(i)] for i in m.get("ids") or [] if _key(i) in mk), None)
                if c is not None and c["ref"] == m["ref"]:
                    src = next(s for s in master if s["id"] == c["_src_id"]) if any(s["id"] == c["_src_id"] for s in master) else None
                    if src is not None:
                        pairs.append((m, c, np.subtract(m["a"], src["a"]) / 2 + np.subtract(m["b"], src["b"]) / 2))
            d = np.median([p[2] for p in pairs], axis=0) if pairs else np.zeros(3)
            used = {id(p[0]) for p in pairs}
            by_c = {}
            for m, c, _ in pairs:
                by_c.setdefault(id(c), []).append(m)
            pid = f"{cid}.{j + 1}"
            ren = {c["id"]: c["id"].replace(cid, pid, 1) for c in comp}
            for c in comp:
                q = copy.deepcopy(c)
                q["id"] = ren[c["id"]]
                q["a"] = [round(float(v), 1) for v in np.add(c["a"], d)]
                q["b"] = [round(float(v), 1) for v in np.add(c["b"], d)]
                own = by_c.get(id(c), [])
                q["ids"] = [i for m in own for i in m.get("ids") or []] if j else list(c.get("ids") or [])
                if j and not own:
                    missing += 1
                for k in (0, 1):
                    e = (q.get("ends") or [None, None])[k]
                    if isinstance(e, list):
                        q["ends"][k] = [dict(x, host=ren.get(x.get("host"), x.get("host"))) for x in e]
                q["truss"] = next((m.get("truss") for m in own), None) or c.get("truss")
                q["key"] = f"{pid}:{c['id']}"
                q.pop("_src_id", None)
                out.append(q)
            if j:
                for m in ms:                                     # drawn at this placement only: kept as drawn
                    if id(m) not in used:
                        extra += 1
                        out.append(m)
            placed += 1
        report.append({"component": cid, "view": tv, "members": len(comp), "joints inside": joints, "placements": placed,
                       "defects": defects, "drawn at one placement only": extra, "copied where not drawn": missing})
    return out, report


def detail_rects(sheet, plan_T, label, box):
    """Where a detail is placed on a plan: the label (VF1) sits at the crossing of a drawn X (two diagonals) whose
    extent is the detail's outline. -> [(cx, cy, w, h) mm]."""
    sx, ox, sy, oy = plan_T
    out = []
    s = sheet.segs
    for w in sheet.words:
        if w[4].strip() != label or not (box[0] <= w[0] <= box[2] and box[1] <= w[1] <= box[3]):
            continue
        cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
        boxes = {}                                      # drawn X: two diagonals of one rectangle around the label
        for q in s:
            x0, y0, x1, y1 = map(float, q[:4])
            if abs(x1 - x0) < 3 or abs(y1 - y0) < 3 or math.hypot(x1 - x0, y1 - y0) < 20:
                continue
            bb = (round(min(x0, x1)), round(min(y0, y1)), round(max(x0, x1)), round(max(y0, y1)))
            if bb[0] <= cx <= bb[2] and bb[1] <= cy <= bb[3]:
                boxes.setdefault(bb, set()).add((x1 - x0) * (y1 - y0) > 0)
        xs_ = [bb for bb, sl in boxes.items() if len(sl) == 2]
        if not xs_:
            continue
        bb = min(xs_, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        mx, my = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
        xs, ys = [bb[0], bb[2]], [bb[1], bb[3]]
        out.append((round(sx * mx + ox), round(sy * my + oy), round(abs(sx) * (max(xs) - min(xs))), round(abs(sy) * (max(ys) - min(ys)))))
    return out


def details(local, profiles, rects, z_top, cid, title):
    """local: members of a detail read in its own frame (mm, origin at its centre). Solved once, then placed at each
    rect (centre; turned 90 deg when its long side runs the other way). -> (members, report)."""
    comp, defects, joints = solve(local, profiles, cid)
    P = np.array([p for m in comp for p in (m["a"], m["b"])], float)
    w0, h0 = np.ptp(P[:, 0]), np.ptp(P[:, 1])
    c0 = (P[:, 0].min() + P[:, 0].max()) / 2, (P[:, 1].min() + P[:, 1].max()) / 2
    out, rep = [], {"component": cid, "view": title, "members": len(comp), "joints inside": joints, "defects": defects,
                    "placements": len(rects), "sizes": []}
    for j, (cx, cy, w, h) in enumerate(rects):
        turn = (w0 >= h0) != (w >= h)
        pid = f"{cid}.{j + 1}"
        ren = {c["id"]: c["id"].replace(cid, pid, 1) for c in comp}
        rep["sizes"].append(f"placement {j + 1}: drawn X {w} x {h} mm, detail {round(h0 if turn else w0)} x {round(w0 if turn else h0)} mm")
        for c in comp:
            q = copy.deepcopy(c)
            for e in ("a", "b"):
                x, y = c[e][0] - c0[0], c[e][1] - c0[1]
                if turn:
                    x, y = -y, x
                q[e] = [round(cx + x, 1), round(cy + y, 1), c[e][2] + z_top]
            q["id"], q["key"] = ren[c["id"]], f"{pid}:{c['id']}"
            for k in (0, 1):
                e = (q.get("ends") or [None, None])[k]
                if isinstance(e, list):
                    q["ends"][k] = [dict(x, host=ren.get(x.get("host"), x.get("host"))) for x in e]
            q.pop("_src_id", None)
            out.append(q)
    return out, rep
