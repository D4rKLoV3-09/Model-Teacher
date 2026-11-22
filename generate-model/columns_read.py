#!/usr/bin/env python3
"""Columns from a column / base plate layout plan, read from the drawing store (no PDF opened).

    from columns_read import read_columns
    cols, rep = read_columns(sheet, box, GX, GY, scale, mark_re=r"SC\\d+")

1. Symbols: the column section drawn as a small I / H in its own colour (Job A: magenta 0.72 pt), one cluster of pieces each
   (pieces within JOIN pt join; a cluster 4-18 pt across with >= 6 pieces). The flanges are the long pieces: flanges drawn
   vertical (an "H") = section depth along X; horizontal (an "I") = depth along Y.
2. Labels: a mark (SC4) in its box, the base plate mark (BP4) under it; the leader (thin line from the box, possibly bent)
   ends at the symbol it names (tip within TIP pt of the symbol). A symbol named by no label and a label reaching no symbol
   are reported (rulings decide).
3. Position: the symbol centre at the sheet's grid calibration (view_read.calibrate), mm.
Returns cols [{mark, bp, x, y, depth_dir: "x"|"y", sym_pt, label_pt, how}] and a report.
"""
import math
import re

import numpy as np

import view_read_store as view_read

JOIN = 1.2
TIP = 22.0          # pt: the arrow stops at the base plate square, up to ~20 pt from the section centre


def symbols(sheet, box, colour=None):
    s = sheet.segs
    x0, y0, x1, y1 = box
    inb = (s[:, 0] >= x0) & (s[:, 0] <= x1) & (s[:, 1] >= y0) & (s[:, 1] <= y1)
    if colour is None:                                       # the most common saturated colour = the symbols' colour
        sat = s[inb & ((s[:, 5:8].max(1) - s[:, 5:8].min(1)) > 0.5)]
        if not len(sat):
            return []
        keys, n = np.unique(np.round(sat[:, 5:8], 1), axis=0, return_counts=True)
        colour = keys[np.argmax(n)]
    m = s[inb & (np.abs(np.round(s[:, 5:8], 1) - colour).max(1) < 0.15)]
    # clusters: single link on piece end points, grid hashing
    pts = [(float(r[0]), float(r[1]), float(r[2]), float(r[3])) for r in m]
    par = list(range(len(pts)))

    def f(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i
    cell = {}
    for i, (a, b, c, d) in enumerate(pts):
        for x, y in ((a, b), (c, d), ((a + c) / 2, (b + d) / 2)):
            cell.setdefault((int(x // 3), int(y // 3)), []).append((i, x, y))
    for (cx, cy), lst in cell.items():
        near = [e for dx in (-1, 0, 1) for dy in (-1, 0, 1) for e in cell.get((cx + dx, cy + dy), [])]
        for i, x, y in lst:
            for j, u, v in near:
                if j != i and abs(x - u) <= JOIN and abs(y - v) <= JOIN:
                    a_, b_ = f(i), f(j)
                    if a_ != b_:
                        par[a_] = b_
    groups = {}
    for i in range(len(pts)):
        groups.setdefault(f(i), []).append(pts[i])
    out = []
    for g in groups.values():
        xs = [v for p in g for v in (p[0], p[2])]
        ys = [v for p in g for v in (p[1], p[3])]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        if len(g) < 6 or not (4 <= max(w, h) <= 18):
            continue
        lv = sum(abs(p[3] - p[1]) for p in g if abs(p[2] - p[0]) < 0.2)       # vertical piece length
        lh = sum(abs(p[2] - p[0]) for p in g if abs(p[3] - p[1]) < 0.2)
        # an H (flanges vertical) has its depth along X; an I (flanges horizontal) along Y
        out.append({"pt": [(max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2], "w": w, "h": h, "depth_dir": "x" if lv > lh else "y"})
    return out


def read_columns(sheet, box, GX, GY, scale, mark_re=r"SC\d+", bp_re=r"BP\d+"):
    P, fit = view_read.calibrate(sheet, box, GX, GY, scale)
    syms = symbols(sheet, box)
    s = sheet.segs
    thin = s[(s[:, 5:8].max(1) < 0.15) & (s[:, 4] < 0.5) & (np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1]) > 1.5)]
    thin = [tuple(map(float, r[:4])) for r in thin if box[0] <= r[0] <= box[2] and box[1] <= r[1] <= box[3]]
    ends = {}
    for l in thin:
        for p in ((l[0], l[1]), (l[2], l[3])):
            ends.setdefault((round(p[0]), round(p[1])), []).append(l)
    words = [w for w in sheet.words if box[0] <= w[0] <= box[2] and box[1] <= w[1] <= box[3]]
    marks = [w for w in words if re.fullmatch(mark_re, w[4].strip())]
    bps = [w for w in words if re.fullmatch(bp_re, w[4].strip())]
    used, cols, free_labels = set(), [], []
    for w in marks:
        bp = min(bps, key=lambda b: math.hypot((b[0] + b[2]) / 2 - (w[0] + w[2]) / 2, b[1] - w[3]), default=None)
        if bp is not None and math.hypot((bp[0] + bp[2]) / 2 - (w[0] + w[2]) / 2, bp[1] - w[3]) > 16:
            bp = None
        # the label box: mark + base plate mark, a few pt round them
        bx0, by0 = min(w[0], bp[0] if bp else w[0]) - 7, w[1] - 6
        bx1, by1 = max(w[2], bp[2] if bp else w[2]) + 7, (bp[3] if bp else w[3]) + 6
        inside = lambda x, y: bx0 <= x <= bx1 and by0 <= y <= by1
        best = None
        for l in thin:                                     # leader pieces leaving the box
            for (ax, ay), (tx, ty) in (((l[0], l[1]), (l[2], l[3])), ((l[2], l[3]), (l[0], l[1]))):
                if not inside(ax, ay) or inside(tx, ty):
                    continue
                tip, seen = (tx, ty), {l}
                for _ in range(6):                         # follow a bent leader
                    nxt = None
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            for o in ends.get((round(tip[0]) + dx, round(tip[1]) + dy), []):
                                if o in seen:
                                    continue
                                if math.dist(tip, o[:2]) < 1.0:
                                    nxt = (o, (o[2], o[3]))
                                elif math.dist(tip, o[2:]) < 1.0:
                                    nxt = (o, (o[0], o[1]))
                    if not nxt:
                        break
                    seen.add(nxt[0]); tip = nxt[1]
                    if inside(*tip):
                        break
                cand = [(math.dist(tip, sy["pt"]), k) for k, sy in enumerate(syms)]
                if cand:
                    d, k = min(cand)
                    if d <= TIP and (best is None or d < best[0]):
                        best = (d, k, tip)
        if best is None:
            free_labels.append(w[4]); continue
        d, k, tip = best
        sy = syms[k]
        x, y = P(*sy["pt"])
        cols.append({"mark": w[4].strip(), "bp": bp[4].strip() if bp else None, "x": round(x), "y": round(y), "depth_dir": sy["depth_dir"],
                     "sym_pt": [round(v, 1) for v in sy["pt"]], "label_pt": [round((w[0] + w[2]) / 2, 1), round((w[1] + w[3]) / 2, 1)],
                     "how": "leader", "tip_off_pt": round(d, 1), "sym": k})
        used.add(k)
    # two labels on one symbol: keep the nearer tip, report
    by = {}
    for c in cols:
        by.setdefault(c["sym"], []).append(c)
    dup = [c for cs in by.values() if len(cs) > 1 for c in sorted(cs, key=lambda c: c["tip_off_pt"])[1:]]
    cols = [c for c in cols if c not in dup]
    unl = [syms[k] for k in range(len(syms)) if k not in used]
    rep = {"symbols": len(syms), "columns": len(cols), "labels_reaching_no_symbol": free_labels, "two_labels_one_symbol": [c["mark"] for c in dup],
           "symbols_with_no_label": [[round(v) for v in P(*u["pt"])] for u in unl], "fit": {k: fit[k] for k in ("rms_mm", "scale_err_pct", "dropped")}}
    return cols, rep
