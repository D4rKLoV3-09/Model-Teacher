#!/usr/bin/env python3
"""Grid table read from one plan of the drawing store (no PDF opened): grid lines, their bubbles, the dimension chains.

    python generate-model/grids.py STORE --sheet 1 [--box x0 y0 x1 y1] [--origin B 1] --out grids.json [--md grids.md]

1. Grid lines: long straight lines (> 40 % of the view) that are not black member lines, clustered by position
   (pieces of one dash-dot line join). Vertical lines carry the lettered grids, horizontal the numbered ones (or the other way:
   decided by where the bubbles sit).
2. Bubbles: text in a drawn circle (view_read.bubbles). Close grids have staggered bubbles (2' / 2 / 2"): a bubble is matched to
   its line by ORDER along the axis, never by its own position. A count mismatch is reported, not guessed.
3. Positions: the dimension chain between two neighbouring grid lines (a number lying between them, within 5 % of the line
   distance at the fitted scale) is taken as drawn; a gap with no dimension is measured at the fitted scale and rounded to 50 mm
   ("scaled"). The scale is fitted on the dimensions found (the stated scale only starts the search).
Output: {"x": {name: mm}, "y": {name: mm}, "src": {name: "dim" | "scaled"}, "fit": {...}}
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

LANE = 60.0      # pt: a staggered grid's line starts this close to its row / column of bubbles
LANE_ON = 200.0  # pt: a line a bubble sits on starts this close to the bubbles (secondary grids start later; hatching far)
STAG = 30.0      # pt: the small dimension of a staggered grid sits this close to its bubble (Job A D2: 28)

sys.path.insert(0, str(Path(__file__).parent))
import store  # noqa: E402
import view_read_store as view_read  # noqa: E402


def grid_lines(sheet, box, frac=0.4):
    s = sheet.segs
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    inb = (np.minimum(s[:, 0], s[:, 2]) >= x0 - 5) & (np.maximum(s[:, 0], s[:, 2]) <= x1 + 5) & \
          (np.minimum(s[:, 1], s[:, 3]) >= y0 - 5) & (np.maximum(s[:, 1], s[:, 3]) <= y1 + 5)
    notmember = ~((s[:, 5:8].max(1) < 0.15) & (s[:, 4] >= 0.5))
    s = s[inb & notmember]
    out = {}
    for ax, (c0, c1, l0, l1, span) in (("v", (0, 2, 1, 3, H)), ("h", (1, 3, 0, 2, W))):
        par = s[np.abs(s[:, c0] - s[:, c1]) < 0.3]
        pos = {}
        for r in par:                                   # join the pieces of one (dash-dot) line, per drawing style
            sty = (round(float(r[4]), 2), round(float(r[5]), 2), round(float(r[6]), 2), round(float(r[7]), 2))
            k = (round(float(r[c0]) / 0.5), sty)
            lo, hi = sorted((float(r[l0]), float(r[l1])))
            a = pos.setdefault(k, [lo, hi, 0.0, 0])
            a[0], a[1], a[2], a[3] = min(a[0], lo), max(a[1], hi), a[2] + hi - lo, a[3] + 1
        # (mean piece length > 1.5 pt: dash-dot grid lines, not hatching made of tiny pieces)
        out[ax] = sorted((k[0] * 0.5, v[0], v[1], k[1]) for k, v in pos.items()
                         if v[1] - v[0] > frac * span and v[2] > 0.3 * frac * span and v[2] / v[3] > 1.5)
    return out


def dims_between(words, a, b, axis, k_guess, tol=0.05):
    """Dimension numbers lying between grid lines a < b (pt) along `axis` ('v' lines -> numbers read along x)."""
    out = []
    for w in words:
        if not re.fullmatch(r"\d{3,6}", w[4]):
            continue
        c = (w[0] + w[2]) / 2 if axis == "v" else (w[1] + w[3]) / 2
        if a < c < b:
            v = float(w[4])
            if abs(v - (b - a) * k_guess) <= tol * v:
                out.append(v)
    return out


def _rows(vals, tol=40, n=3):
    """Clusters of near-equal values with at least n members -> list of (mean, members)."""
    out = []
    for v in sorted(vals, key=lambda t: t[0]):
        if out and v[0] - out[-1][-1][0] <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    return [(sum(t[0] for t in g) / len(g), g) for g in out if len(g) >= n]


def _match(bp, lp, skip=60.0):
    """Monotone matching of bubbles to lines (both sorted), minimum total offset; a bubble may stay unmatched at a cost
    (`skip` pt), a line may stay unmatched for free (other long lines). DP."""
    nb, nl = len(bp), len(lp)
    D = [[0.0] * (nl + 1) for _ in range(nb + 1)]
    for i in range(1, nb + 1):
        D[i][0] = i * skip
        for j in range(1, nl + 1):
            D[i][j] = min(D[i][j - 1], D[i - 1][j] + skip, D[i - 1][j - 1] + abs(bp[i - 1] - lp[j - 1]))
    out, i, j = [], nb, nl
    while i > 0 and j > 0:
        if D[i][j] == D[i - 1][j - 1] + abs(bp[i - 1] - lp[j - 1]):
            out.append((i - 1, j - 1)); i -= 1; j -= 1
        elif D[i][j] == D[i][j - 1]:
            j -= 1
        else:
            i -= 1
    out = out[::-1]
    return out, (sum(abs(bp[i] - lp[j]) for i, j in out) / max(len(out), 1))


def read(sheet, box, scale, origin):
    lines = grid_lines(sheet, box)
    bub = view_read.bubbles(sheet, box)
    words = [w for w in sheet.words if box[0] <= w[0] <= box[2] and box[1] <= w[1] <= box[3]]
    res, src, fit = {}, {}, {}
    k0 = 25.4 / 72 * scale
    def lanes(c, o):
        """Bubble rows (c=1: same y, spread along x) or columns (c=0); staggered bubbles within 60 pt join their lane."""
        core = [(m, g) for m, g in _rows([(b[c], b) for b in bub], tol=15) if max(b[o] for _, b in g) - min(b[o] for _, b in g) > 200]
        return [b for b in bub if any(abs(b[c] - m) <= 60 for m, _ in core)]
    vb = lanes(1, 0)                                         # rows of bubbles above / below the plan: lettered grids
    hb = [b for b in lanes(0, 1) if b not in vb]              # columns of bubbles left / right: numbered grids
    # the grid-line style: the style of the lines passing exactly through bubbles (Job A: grey 0.5 dash-dot)
    from collections import Counter
    on = Counter(l[3] for ax, B in (("v", vb), ("h", hb)) for l in lines[ax]
                 if any(abs(l[0] - (b[0] if ax == "v" else b[1])) < 1.5 for b in B))
    styles = {k for k, n in on.items() if n >= 2 and max(k[1:]) > 0.2}   # main + secondary grid styles; never black (members, dims)
    fit["grid_line_styles"] = str(sorted(styles))
    lines = {ax: [l[:3] for l in ls if not styles or l[3] in styles] for ax, ls in lines.items()}
    for ax in lines:                                         # neighbours within 1 pt are one line
        m = []
        for l in lines[ax]:
            if m and l[0] - m[-1][0] < 1.0:
                m[-1] = (m[-1][0], min(m[-1][1], l[1]), max(m[-1][2], l[2]))
                continue
            m.append(l)
        lines[ax] = m
    # 1. the sheet scale from its dimensions (median of value / distance over gaps between on-line bubbles)
    lanes_ = {"v": [b[1] for b in vb], "h": [b[0] for b in hb]}

    def onl(ax, b):
        """The grid line a bubble sits on: within 0.6 pt of its centre and starting at the bubble lane (hatching or an
        outline passing through a bubble does not start there)."""
        c = b[0] if ax == "v" else b[1]
        ok = [l for l in lines[ax] if abs(l[0] - c) < 0.6 and min(min(abs(l[1] - q), abs(l[2] - q)) for q in lanes_[ax]) < LANE_ON]
        return min(ok, key=lambda l: abs(l[0] - c)) if ok else None
    ratios = []
    for ax, B in (("v", vb), ("h", hb)):
        c = 0 if ax == "v" else 1
        P = sorted({round(b[c], 1) for b in B if onl(ax, b)})
        for i in range(len(P) - 1):
            ratios += [v / (P[i + 1] - P[i]) for v in dims_between(words, P[i], P[i + 1], ax, k0, tol=0.06)]
    ratios.sort()
    k = ratios[len(ratios) // 2] if ratios else k0
    fit["mm_per_pt"] = round(k, 3)
    fit["scale_from"] = f"{len(ratios)} dimensions" if ratios else "stated scale (no dimension found)"
    for axis, B in (("v", vb), ("h", hb)):
        c = 0 if axis == "v" else 1
        o = 1 - c
        names = {}
        for b in B:
            names.setdefault(b[2], []).append(b)
        pos, how = {}, {}
        claimed = set()
        for n, bs in names.items():                      # 2. a bubble on its line: that line
            l = next((onl(axis, b) for b in bs if onl(axis, b)), None)
            if l:
                pos[n], how[n] = l[0], "on line"
                claimed.add(l[0])
        lane = [b[o] for b in B]
        for n, bs in names.items():                      # 3. staggered: the grid-style line starting at its jog
            if n in pos:
                continue
            cand = [l for l in lines[axis] if l[0] not in claimed and min(min(abs(l[1] - q), abs(l[2] - q)) for q in lane) < LANE
                    and min(abs(l[0] - b[c]) for b in bs) < 50]
            if cand:
                l = min(cand, key=lambda l: min(abs(l[0] - b[c]) for b in bs))
                pos[n], how[n] = l[0], "staggered: line after the jog"
                claimed.add(l[0])
            else:
                pos[n], how[n] = sum(b[c] for b in bs) / len(bs), "bubble position (no line found)"
        if not pos:
            fit[axis] = "NO GRIDS"
            continue
        sign = 1 if axis == "v" else -1                  # y up: numbered grids count from the bottom of the sheet
        o_ = next((g for g in origin if g in pos), next(iter(pos)))
        # 3b. a grid not exact (staggered / from its bubble) = its exact neighbour +- the dimension written between them
        #     (Job A: 2' = 2 - 500, 2" = 2 + 500, D2 = D1 + 500); the small dimension sits next to the staggered bubble
        exact = {n for n, h in how.items() if h == "on line"}
        nums = [w for w in words if re.fullmatch(r"\d{3,6}", w[4])]
        todo = [n for n in pos if n not in exact]
        for _pass in range(4):                               # a derived grid carries the next (D1 from D, then D2 from D1)
          for n in list(todo):
              bpos = sum(b[c] for b in names[n]) / len(names[n])
              best = None
              for w in nums:
                  wc = (w[0] + w[2]) / 2 if axis == "v" else (w[1] + w[3]) / 2
                  v = float(w[4])
                  for e in exact:
                      wo = (w[1] + w[3]) / 2 if axis == "v" else (w[0] + w[2]) / 2
                      near_lane = min(abs(wo - q) for q in lane) < 160
                      if abs(wc - bpos) < STAG and v <= 1500 and near_lane:   # the small dimension beside a staggered bubble
                          cand = (abs(wc - bpos), e, v)
                      elif min(pos[e], pos[n]) < wc < max(pos[e], pos[n]) and abs(v - abs(pos[e] - pos[n]) * k) <= 0.03 * v:
                          cand = (abs(wc - (pos[e] + pos[n]) / 2) + 20, e, v)   # a dimension between it and an exact neighbour
                      else:
                          continue
                      if abs(pos[e] - bpos) * k > v + 2500:
                          continue                            # the neighbour it is measured from lies next to it
                      if best is None or cand < best:
                          best = cand
              if best:
                  _, e, v = best
                  side = 1 if bpos > pos[e] else -1
                  pos[n] = pos[e] + side * v / k
                  how[n] = f"{e} {'+' if side * sign > 0 else '-'} {v:g} (dimension)"
                  exact.add(n)
                  todo.remove(n)
        tab = {n: round(sign * (p_ - pos[o_]) * k / 50) * 50 for n, p_ in pos.items()}
        for n, h in how.items():
            if "(dimension)" in h:
                e, op, v = h.split()[:3]
                tab[n] = tab[e] + (float(v) if op == "+" else -float(v))
        # 4. check against the dimensions written between neighbouring grids
        srt = sorted(pos, key=lambda n: pos[n])
        worst, nd = 0.0, 0
        for i in range(len(srt) - 1):
            a_, b_ = pos[srt[i]], pos[srt[i + 1]]
            for v in dims_between(words, a_, b_, axis, k, tol=0.03):
                nd += 1
                worst = max(worst, abs(abs(tab[srt[i + 1]] - tab[srt[i]]) - v))
        res["x" if axis == "v" else "y"] = dict(sorted(tab.items(), key=lambda t: t[1]))
        src.update(how)
        fit[axis] = f"{len(tab)} grids; {sum(h == 'on line' for h in how.values())} on line, " \
                    f"{sum('(dimension)' in h for h in how.values())} from a neighbour + dimension, {sum(h.startswith('staggered') for h in how.values())} line after the jog, {sum(h.startswith('bubble') for h in how.values())} from the bubble; " \
                    f"{nd} dimensions agree within {worst:.0f} mm"
    return {"x": res.get("x", {}), "y": res.get("y", {}), "src": src, "fit": fit}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("store")
    ap.add_argument("--sheet", type=int, required=True)
    ap.add_argument("--box", type=float, nargs=4)
    ap.add_argument("--scale", type=float)
    ap.add_argument("--origin", nargs="*", default=["B", "1"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--md")
    a = ap.parse_args()
    sh = store.sheets(a.store)[a.sheet - 1]
    box = a.box or [0, 0, *sh.size]
    scale = a.scale
    if not scale:
        m = next((re.search(r"1\s*:\s*(\d+)", w[4]) for w in sh.words if re.search(r"SCALE\s*1\s*:\s*\d+", w[4])), None)
        scale = float(m.group(1)) if m else 100.0
    G = read(sh, box, scale, a.origin)
    Path(a.out).write_text(json.dumps(G, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"grids from p{a.sheet} (1:{scale:g}): x {len(G['x'])}, y {len(G['y'])}; " + "; ".join(f"{k} {v}" for k, v in G["fit"].items()))
    for k in ("x", "y"):
        print(f"  {k}: " + ", ".join(f"{n} {v:g}{'' if G['src'].get(n) in ('on line',) or '(dimension)' in G['src'].get(n, '') else '*'}" for n, v in G[k].items()))
    print("  (* = not exact: line after a jog or the bubble itself, scaled)")
    if a.md:
        Path(a.md).write_text("# Grids\n\n" + "\n".join(f"- {k} {n}: {v:g} mm ({G['src'].get(n)})" for k in ("x", "y") for n, v in G[k].items()) + "\n")
