#!/usr/bin/env python3
"""Read ONE view from the drawing store (no PDF opened) -> register rows: every drawn steel line and column symbol
of the view, each with its mark, sheet position (pt) and building position (mm), or flagged unlabelled.

    from view_read_store import read_plan
    rows, fit = read_plan(sheet, box, GX, GY, z, mark_re, scale=75)

Steps (each counted, nothing dropped silently):
  1. calibrate: grid bubbles inside the view -> least-squares X(px), Y(py) per axis; bubbles off the fit by more than
     TOL_BUB mm (staggered bubbles of close grids) are left out of the fit and named in `fit`. The fitted scale is
     compared with the stated scale.
  2. lines: thin filled quads (members drawn filled) + black stroked pieces of member weight; touching collinear pieces
     joined (gap <= JOIN pt), and pieces with a label lying in the gap between them joined (a label break).
  3. marks: each mark word is given to the nearest parallel line it lies beside (<= OFF pt, inside its length).
  4. columns: filled black I / box symbols of column size.
Row: {id, kind: line | column, mark, a_pt, b_pt, a, b (mm, X Y Z), L, label_pt, label_off, status: labelled |
unlabelled}.
"""
import math
import os
import re

import numpy as np

JOIN = 2.5        # pt: collinear pieces closer than this are one drawn line
BRIDGE = 60.0     # pt: widest label break bridged
LEADER_SHORT_PT = 40.0  # pt: a row shorter than this is a short piece for a leader
LEADER_TIP = 12.0 # pt: a leader's tip (arrow) this close to a member's line / face names it
BRIDGE_MIN = 15.0 # pt: both pieces either side of a label break must be at least this long
OFF = 24.0        # pt: a mark this close beside a parallel line labels it (nearest wins; a close second = ambiguous)
ACROSS = 2.0      # pt: penalty for a short mark not parallel to its line (written level beside a web)
UPRIGHT_MARK = re.compile(r"^(S?C|COL|ST|P|K)\d", re.I)   # marks naming columns / posts (COL-LEVEL); a job may set its own
LAST = 1000.0     # pt: rank of a level column mark beside an upright line: after every mark written along its line
ANG = 12.0        # deg: mark direction vs line direction (rotated text is set a few degrees off its brace)
TOL_BUB = 60.0    # mm: grid bubble off the fit -> not used for the fit (staggered / mislabelled)
SCALE_TOL = 0.03  # a fit's scale must be within 3 % of the stated scale
MINLEN = 6.0      # pt: shorter pieces are symbols, not members


def _ang(x0, y0, x1, y1):
    return math.degrees(math.atan2(-(y1 - y0), x1 - x0)) % 180


def _fit_axis(pairs, stated, sign):
    """pairs [(px, mm, name)] -> (scale, offset, used, dropped, rms). Consensus fit: every pair of bubbles of two
    different grids proposes a line; only scales within SCALE_TOL of the stated scale count; the proposal most bubbles
    agree with (within TOL_BUB) wins, then a least-squares fit on those. Bubbles off it (staggered for close grids,
    mislabelled on the drawing) are named in `dropped`."""
    best = None
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            (p0, m0, n0), (p1, m1, n1) = pairs[i], pairs[j]
            if n0 == n1 or abs(p1 - p0) < 20:
                continue
            sc = (m1 - m0) / (p1 - p0)
            if abs(sc - sign * stated) > SCALE_TOL * stated:
                continue
            of = m0 - sc * p0
            inl = [q for q in pairs if abs(sc * q[0] + of - q[1]) <= TOL_BUB]
            key = (len({q[2] for q in inl}), len(inl))
            if best is None or key > best[0]:
                best = (key, inl)
    if best is None or best[0][0] < 2:
        raise ValueError("no two grids agree with the stated scale")
    inl = best[1]
    A = np.array([[q[0], 1.0] for q in inl]); y = np.array([q[1] for q in inl])
    (sc, of), *_ = np.linalg.lstsq(A, y, rcond=None)
    r = A @ np.array([sc, of]) - y
    dropped = [q[2] for q in pairs if q not in inl]
    return sc, of, sorted({q[2] for q in inl}), dropped, float(np.sqrt((r ** 2).mean()))


def grid_name(t):
    return t.strip().replace("\u2019", "'").replace("''", '"').replace("\u201d", '"')


def circles(sheet):
    """Drawn circles (a path of 4+ bezier pieces, square bbox, r 5-22 pt) -> [(cx, cy, r)]. Cached on the sheet."""
    if getattr(sheet, "_circles", None) is None:
        by = {}
        for b in sheet.bez:
            by.setdefault(int(b[10]), []).append(b)
        out = []
        for bs in by.values():
            if len(bs) < 4:
                continue
            xs = [v for b in bs for v in b[0:8:2]]; ys = [v for b in bs for v in b[1:8:2]]
            w, h = max(xs) - min(xs), max(ys) - min(ys)
            if 10 <= w <= 44 and abs(w - h) < 0.15 * w:
                out.append((float(max(xs) + min(xs)) / 2, float(max(ys) + min(ys)) / 2, float(w) / 2))
        # CIRCLE-SEG: a circle drawn as one path of many short straight pieces (a flattened arc, common in CAD exports):
        # 8+ pieces, square bbox 10-44 pt, every end point at the same distance from the centre (within 8 %)
        S = sheet.segs
        if len(S):
            L = np.hypot(S[:, 2] - S[:, 0], S[:, 3] - S[:, 1])
            sm = S[L < 12]
            pid = sm[:, 9].astype(int)
            order = np.argsort(pid, kind="stable")
            pid, sm = pid[order], sm[order]
            cuts = np.flatnonzero(np.diff(pid)) + 1
            for g in np.split(sm, cuts):
                if len(g) < 8:
                    continue
                xs, ys = np.r_[g[:, 0], g[:, 2]], np.r_[g[:, 1], g[:, 3]]
                w, h = xs.max() - xs.min(), ys.max() - ys.min()
                if not (10 <= w <= 44 and abs(w - h) < 0.15 * w):
                    continue
                cx, cy = (xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2
                d = np.hypot(xs - cx, ys - cy)
                if d.std() < 0.08 * d.mean() and not any(abs(cx - c[0]) < 2 and abs(cy - c[1]) < 2 for c in out):
                    out.append((float(cx), float(cy), float(w) / 2))
        sheet._circles = out
    return sheet._circles


def bubbles(sheet, box, names=None):
    """Grid bubbles: a word centred in a drawn circle. -> [(cx, cy, name)]; names not in `names` are listed apart."""
    x0, y0, x1, y1 = box
    C = circles(sheet)
    out = []
    for w in sheet.words:
        cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
        if not (x0 <= cx <= x1 and y0 <= cy <= y1) or len(w[4]) > 4:
            continue
        c = next((c for c in C if math.hypot(cx - c[0], cy - c[1]) < 0.45 * c[2] and (w[2] - w[0]) < 1.9 * c[2]), None)
        if c:
            out.append((float(c[0]), float(c[1]), grid_name(w[4])))   # the circle centre: the text box sits off it
    return out


WIDEN = 260.0     # pt: a plan with no bubble on one axis looks this far past its box for that axis (bubbles drawn off the view)


def calibrate(sheet, box, GX, GY, scale, y_anchor=None):
    """Plan: x / y from the grid bubbles. An axis with no bubble in the box: bubbles in line with the box up to WIDEN pt
    past it; none there: y_anchor (py, mm) = a drawn row (columns) put on a grid line by the caller (VIEW-R03)."""
    x0, y0, x1, y1 = box
    GX = {grid_name(k): v for k, v in GX.items()}
    GY = {grid_name(k): v for k, v in GY.items()}
    bub = [(b[0] - 1, b[1] - 1, b[0] + 1, b[1] + 1, b[2]) for b in bubbles(sheet, box)]
    bx = [((w[0] + w[2]) / 2, GX[w[4]], w[4]) for w in bub if w[4] in GX]
    by = [((w[1] + w[3]) / 2, GY[w[4]], w[4]) for w in bub if w[4] in GY]
    if not bx:
        bx = [(b[0], GX[b[2]], b[2]) for b in bubbles(sheet, [x0, y0 - WIDEN, x1, y1 + WIDEN]) if b[2] in GX]
    if not by:
        by = [(b[1], GY[b[2]], b[2]) for b in bubbles(sheet, [x0 - WIDEN, y0, x1 + WIDEN, y1]) if b[2] in GY]
    # a grid name can sit at both ends (and in the other plan on the sheet): keep the row/column holding most bubbles
    stated = 25.4 / 72 * scale

    def axis(pairs, sign):
        names = {p[2] for p in pairs}
        if len(names) >= 2:
            try:
                return _fit_axis(pairs, stated, sign) + ("fit",)
            except ValueError:
                pass
        if not pairs and sign == -1 and y_anchor:        # VIEW-R03: a drawn row put on a grid line
            return -stated, y_anchor[1] + stated * y_anchor[0], ["anchor"], [], 0.0, "row on a grid line (VIEW-R03)"
        if pairs:                                       # one grid on this axis: stated scale + that bubble (median)
            offs = sorted(p[1] - sign * stated * p[0] for p in pairs)
            return sign * stated, offs[len(offs) // 2], [pairs[0][2]], [], 0.0, "one bubble + stated scale"
        raise ValueError("needs placement: no grid bubble on one axis (bubbles: " + ", ".join(sorted({b[4] for b in bub})) + ")")
    sx, ox, ux, dx, rx, mx = axis(bx, 1)
    sy, oy, uy, dy, ry, my = axis(by, -1)
    sx, ox, sy, oy = map(float, (sx, ox, sy, oy))
    fit = {"x_mm_per_pt": round(sx, 3), "y_mm_per_pt": round(-sy, 3), "stated_mm_per_pt": round(stated, 3),
           "x_grids": ux, "y_grids": uy, "dropped": dx + dy, "rms_mm": round(max(rx, ry), 1),
           "scale_err_pct": round(100 * max(abs(sx - stated), abs(-sy - stated)) / stated, 2), "method": [mx, my],
           "T": [sx, ox, sy, oy],
           "grids_not_in_table": sorted({b[4] for b in bub if b[4] not in GX and b[4] not in GY and not re.fullmatch(r"S[A-Z]?\d+[a-z]?|\u00d8\d+", b[4])})}
    if fit["scale_err_pct"] > 2.0 or fit["rms_mm"] > 50:
        raise ValueError(f"calibration off: scale {fit['scale_err_pct']} %, rms {fit['rms_mm']} mm (grids {ux} / {uy})")
    return (lambda px, py: (round(float(sx * px + ox), 1), round(float(sy * py + oy), 1))), fit


def _style(s):
    """A drawn piece's style for legends: colour (to 0.05) + pen width (to 0.02 pt)."""
    return (round(float(s[5]) * 20) / 20, round(float(s[6]) * 20) / 20, round(float(s[7]) * 20) / 20, round(float(s[4]) * 50) / 50)


def legend(sheet):
    """LEGEND-R01 (the curator 2026-11-22, p10 'MS HANDRAIL'): a sheet legend = a short line sample (one style, 20-120 pt, often
    dash-dot drawn as pieces) with words written just right of it on the same line. Every line of that style on the sheet is
    that thing (handrail), never a member. -> {style: (name, [x0, y0, x1, y1] of the sample)}. Black samples are skipped
    (members, leaders and underlines are black)."""
    if getattr(sheet, "_legend", None) is not None:
        return sheet._legend
    s = sheet.segs
    lev = s[(np.abs(s[:, 1] - s[:, 3]) < 0.3) & (s[:, 5:8].max(1) >= 0.15)]
    out = {}
    for w in sheet.words:
        name = w[4].strip()
        if (w[6] >= 12 or len(name) < 3 or not re.search(r"[A-Z]{3}", name, re.I) or re.fullmatch(r"[\d.,x@ ]+", name)
                or re.fullmatch(r"[A-Z]{1,4}\d+[A-Za-z]?\s*(\(?TYP\.?\)?)?", name, re.I)):   # a mark is not a legend entry
            continue
        cy = (w[1] + w[3]) / 2
        near = lev[(np.abs(lev[:, 1] - cy) < 4) & (np.maximum(lev[:, 0], lev[:, 2]) <= w[0] + 1) & (np.maximum(lev[:, 0], lev[:, 2]) >= w[0] - 25)]
        if not len(near):
            continue
        st = _style(near[0])
        y = float(near[0][1])
        same = lev[(np.abs(lev[:, 1] - y) < 0.3) & np.array([_style(q) == st for q in lev])]
        xs = sorted((float(min(q[0], q[2])), float(max(q[0], q[2]))) for q in same if max(q[0], q[2]) <= w[0] + 1)
        run = [xs[-1]] if xs else []
        for a, b in reversed(xs[:-1]):                      # the sample: same-style pieces chained leftwards from the word
            if run[0][0] - b <= 4:
                run.insert(0, (a, b))
        x0, x1 = run[0][0], run[-1][1]
        if 20 <= x1 - x0 <= 120:
            out.setdefault(st, []).append((name, [x0, y - 1, x1, y + 1]))
    # one style naming two different things (a connection legend told apart by end symbols): the style does not decide
    out = {st: v[0] for st, v in out.items() if len({n for n, _ in v}) == 1}
    sheet._legend = out
    return out


def legend_lines(sheet, box, P3=None, gap=4.0):
    """Every drawn run of a legend style in the box (dash / dot pieces chained, gap <= 4 pt), the legend samples left out.
    -> [{"name", "a_pt", "b_pt", "L_pt", "a", "b"}]."""
    L = legend(sheet)
    if not L:
        return []
    x0, y0, x1, y1 = box
    s = sheet.segs
    m = (s[:, 0] >= x0) & (s[:, 2] >= x0) & (s[:, 0] <= x1) & (s[:, 2] <= x1) & (s[:, 1] >= y0) & (s[:, 3] >= y0) & (s[:, 1] <= y1) & (s[:, 3] <= y1)
    groups = {}
    for q in s[m]:
        st = _style(q)
        if st not in L:
            continue
        sx = L[st][1]
        if sx[0] - 1 <= min(q[0], q[2]) and max(q[0], q[2]) <= sx[2] + 1 and sx[1] <= q[1] <= sx[3]:
            continue                                         # the legend sample itself
        a = _ang(*map(float, q[:4]))
        a = 0.0 if a > 179.5 else a
        t = (math.cos(math.radians(a)), -math.sin(math.radians(a)))
        n = (-t[1], t[0])
        off = float(q[0]) * n[0] + float(q[1]) * n[1]
        s0, s1 = sorted((float(q[0]) * t[0] + float(q[1]) * t[1], float(q[2]) * t[0] + float(q[3]) * t[1]))
        groups.setdefault((st, round(a / 0.5), round(off / 0.6)), []).append((s0, s1, t, n, off))
    out = []
    for (st, _, _), g in groups.items():
        g.sort(key=lambda e: e[0])
        cur = list(g[0][:2]); t, n, off = g[0][2], g[0][3], g[0][4]
        for s0, s1, *_ in g[1:] + [(math.inf, math.inf)]:
            if s0 - cur[1] <= gap:
                cur[1] = max(cur[1], s1)
                continue
            if cur[1] - cur[0] >= MINLEN:
                pa = (cur[0] * t[0] + off * n[0], cur[0] * t[1] + off * n[1])
                pb = (cur[1] * t[0] + off * n[0], cur[1] * t[1] + off * n[1])
                e = {"name": L[st][0], "a_pt": [round(pa[0], 2), round(pa[1], 2)], "b_pt": [round(pb[0], 2), round(pb[1], 2)],
                     "L_pt": round(cur[1] - cur[0], 1)}
                if P3:
                    e.update(a=P3(*pa), b=P3(*pb))
                out.append(e)
            cur = [s0, s1]
    return out


def member_segs(sheet, kind="structure"):
    """LAYER-R01: the pieces a member pass reads: structure = steel / unnamed layers only (purlin, dimension, grid, text and other
    trades' layers hidden); purlin = purlin / girt layers only. A sheet stored without layer names reads every shown piece."""
    return sheet.pass_segs(kind) if hasattr(sheet, "pass_segs") else sheet.segs


def lines(sheet, box):
    x0, y0, x1, y1 = box
    q = sheet.quad
    L = [tuple(map(float, r[:4])) + (float(r[4]),) for r in q if 0.4 <= r[4] <= 2.4]
    s = member_segs(sheet)
    blk = s[(s[:, 5:8].max(1) < 0.15) & (s[:, 4] >= 0.5) & (s[:, 4] <= 1.2)]
    L += [tuple(map(float, r[:4])) + (float(r[4]) / 2,) for r in blk]
    inb = lambda l: all(x0 <= x <= x1 and y0 <= y <= y1 for x, y in ((l[0], l[1]), (l[2], l[3])))
    L = [l for l in L if inb(l) and math.dist(l[:2], l[2:4]) >= MINLEN]
    return L


def splice_boxes(sheet, box):
    """SPLICE-R01 (the curator 2026-11-23): a small rectangle / square with an X in it drawn on a member = a splice joint (bolted).
    The X: two black diagonal pieces (5-40 pt) of opposite slope spanning the same box. -> [(x0, y0, x1, y1)] pt."""
    x0, y0, x1, y1 = box
    s = sheet.segs
    cand = [tuple(map(float, r[:4])) for r in s[(s[:, 5:8].max(1) < 0.15)]]
    cand += [tuple(map(float, r[:4])) for r in sheet.quad]
    diag = {}
    for a, b, c, d in cand:
        L_ = math.hypot(c - a, d - b)
        if not (5 <= L_ <= 40 and abs(c - a) > 1 and abs(d - b) > 1 and x0 <= a <= x1 and y0 <= b <= y1):
            continue
        k = (round(min(a, c)), round(min(b, d)), round(max(a, c)), round(max(b, d)))
        diag.setdefault(k, set()).add((c - a) * (d - b) > 0)
    out = []
    for k, sl in diag.items():
        if sl == {True, False} and not any(abs(k[0] - o[0]) <= 1 and abs(k[1] - o[1]) <= 1 for o in out):
            out.append(k)
    return out


def join(L, words, splices=()):
    """Collinear pieces -> drawn lines. A gap is bridged when <= JOIN, or <= BRIDGE with a word lying in it, or when a splice
    symbol (splice_boxes) sits in it: the member runs through its splice (the point is kept in "splices", not a cut)."""
    sc = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, math.hypot(b[2] - b[0], b[3] - b[1])) for b in splices]
    groups = {}
    for l in L:
        a = _ang(*l[:4])
        a = 0.0 if a > 179.5 else a
        t = (math.cos(math.radians(a)), -math.sin(math.radians(a)))
        n = (-t[1], t[0])
        off = l[0] * n[0] + l[1] * n[1]
        s0, s1 = sorted((l[0] * t[0] + l[1] * t[1], l[2] * t[0] + l[3] * t[1]))
        groups.setdefault((round(a / 0.5), round(off / 0.6)), []).append((s0, s1, off, a, l[4], t, n))
    wc = [((w[0] + w[2]) / 2, (w[1] + w[3]) / 2, w) for w in words]
    out = []
    for g in groups.values():
        g.sort()
        cur = list(g[0][:2]); meta = g[0]; cuts = []; spl = []; last = g[0][1] - g[0][0]
        for s0, s1, *_ in g[1:] + [(math.inf, math.inf)]:
            gap = s0 - cur[1]
            bridged = False
            t, n, off = meta[5], meta[6], meta[2]
            for cx, cy, dg in sc:                           # a splice in the gap: the same member on both sides
                al = cx * t[0] + cy * t[1]; of = cx * n[0] + cy * n[1]
                if JOIN < gap <= dg + 6 and cur[1] - 2 < al < s0 + 2 and abs(of - off) < 3:
                    spl.append(al)
                    cur[1] = max(cur[1], s1); last = s1 - s0
                    bridged = None
                    break
            if bridged is None:
                continue
            if JOIN < gap <= BRIDGE and min(last, s1 - s0) >= BRIDGE_MIN:     # end ticks either side of a label are not a member
                t, n, off = meta[5], meta[6], meta[2]
                for cx, cy, w in wc:
                    al = cx * t[0] + cy * t[1]; of = cx * n[0] + cy * n[1]
                    if cur[1] - 1 < al < s0 + 1 and abs(of - off) < 6:
                        bridged = True; break
            if gap <= JOIN or bridged:
                cuts.append((cur[1] + s0) / 2)              # piece boundary: where two butting members may meet
                cur[1] = max(cur[1], s1); last = s1 - s0
                continue
            t, n, off = meta[5], meta[6], meta[2]
            p = (cur[0] * t[0] + off * n[0], cur[0] * t[1] + off * n[1])
            q = (cur[1] * t[0] + off * n[0], cur[1] * t[1] + off * n[1])
            cp = [[round(c * t[0] + off * n[0], 2), round(c * t[1] + off * n[1], 2)] for c in cuts if cur[0] < c < cur[1]]
            sp = [[round(c * t[0] + off * n[0], 2), round(c * t[1] + off * n[1], 2)] for c in spl if cur[0] < c < cur[1]]
            out.append({"a_pt": [round(p[0], 2), round(p[1], 2)], "b_pt": [round(q[0], 2), round(q[1], 2)], "ang": meta[3], "hw": meta[4],
                        "cuts": cp, **({"splices": sp} if sp else {})})
            cur = [s0, s1]; cuts = []; spl = []; last = s1 - s0
    return out


def _seg_dist(c, a, b):
    ax, ay = a; bx, by = b
    L2 = (bx - ax) ** 2 + (by - ay) ** 2 or 1e-9
    t = max(0.0, min(1.0, ((c[0] - ax) * (bx - ax) + (c[1] - ay) * (by - ay)) / L2))
    return math.dist(c, (ax + t * (bx - ax), ay + t * (by - ay)))


NAMED = [(re.compile(r"^MONORAIL\b", re.I), "MONORAIL")]   # NAME-R02: steel named in words (a job may add its own)
SECTION_TXT = re.compile(r"^(UB|UC|LC|UA|EA|PFC|RHS|SHS|CHS)\s*\d+(\.\d+)?\s*[xX]\s*\d+", re.I)


def note_block(words, w, gap=5.0):
    """The note a word belongs to: the words stacked with it (each line starting within 15 pt of its left edge, line gaps
    <= gap pt), top to bottom."""
    blk = [w]
    for _ in range(8):
        grown = [v for v in words if v not in blk and abs(v[0] - blk[0][0]) <= 15 and v[6] < 12
                 and any(-gap <= v[1] - b[3] <= gap or -gap <= b[1] - v[3] <= gap for b in blk)]
        if not grown:
            break
        blk += grown
    return sorted(blk, key=lambda v: v[1])


def named_mark(words, w, mark_re):
    """NAME-R02 (the curator 2026-11-23, MONORAIL & HOIST): a member named in words is steel, modelled first and sized later. Its
    mark from the note it is written in: a member tag in the note (p19 'B3 / MONORAIL BEAM') -> that tag; a section written
    in it (p24 'UC152X152X37 STUMP & MONORAIL BEAM') -> that section; neither -> the name (sized by the build from the
    tag / size found for it elsewhere, else by scale from the most similar member). -> (mark, how) or None."""
    name = next((nm for rx, nm in NAMED if rx.search(w[4].strip())), None)
    if not name:
        return None
    blk = note_block(words, w)
    for v in blk:
        m = re.fullmatch(rf"({mark_re})\s*(\(?TYP\.?\)?)?", v[4].strip())
        if m and v is not w:
            return m.group(1), f"{name}: tag {m.group(1)} in its note"
    for v in blk:
        if SECTION_TXT.match(v[4].strip()):
            return v[4].strip().upper().replace(" ", ""), f"{name}: section written in its note"
    return name, f"{name}: no tag or size in its note"


def named_sizes(sheets, mark_re):
    """NAME-R02: every note naming a member in words across the set -> {name: [tag or section, ...]} (the tags / sizes
    written with that name anywhere): a name-only member takes the size found for it on another page."""
    out = {}
    for sh in sheets:
        for w in sh.words:
            nm = named_mark(sh.words, w, mark_re) if w[6] < 12 else None
            if nm and nm[0] != next(n for rx, n in NAMED if rx.search(w[4].strip())):
                out.setdefault(next(n for rx, n in NAMED if rx.search(w[4].strip())), []).append(nm[0])
    return out


def _marks(words, mark_re):
    """Mark words, incl. "BR1 (TYP)" spans: (word, mark, typ). A member named in words (NAMED) counts as a mark too."""
    out = []
    for w in words:
        m = re.fullmatch(rf"({mark_re})\s*(\(?TYP\.?\)?)?", w[4])
        if m and w[6] < 12:                       # grid bubbles are the big text
            out.append((w, m.group(1), bool(m.group(m.lastindex)) if m.lastindex > 1 else False))
            continue
        elif w[6] < 12:
            nm = named_mark(words, w, mark_re)
            if nm:
                out.append((w, nm[0], False))
    return out


def split_marks(rows, words, mark_re, grids=None):
    """A joined line with two different marks beside it is two members butting in line: split it at the piece
    boundary between the marks. BAY-SPLIT (the curator D39): a level / plumb line drawn through several bays with two different
    marks along it, each written at a label box on this line (label_boxes) and nearer to this line than to any other
    parallel line, and no piece boundary between them, is split at the grid line (`grids` = bubble centres, pt) between
    them nearest their midpoint; each bay keeps its label."""
    mk = _marks(words, mark_re)
    out = []
    for r in rows:
        (ax, ay), (bx, by) = r["a_pt"], r["b_pt"]
        Lr = math.dist((ax, ay), (bx, by)); tx, ty = (bx - ax) / Lr, (by - ay) / Lr
        beside = []
        for w, mark, _ in mk:
            cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
            al = (cx - ax) * tx + (cy - ay) * ty; of = abs(-(cx - ax) * ty + (cy - ay) * tx)
            if 0 <= al <= Lr and of <= OFF and abs(((w[5] % 180) - r["ang"] + 90) % 180 - 90) <= ANG:
                beside.append((al, mark))
        beside.sort()
        cuts = sorted(((c[0] - ax) * tx + (c[1] - ay) * ty) for c in r.get("cuts", []))
        horiz, vert = abs(ay - by) < 0.5, abs(ax - bx) < 0.5
        gl = []
        if grids and (horiz or vert):
            k_ = 0 if horiz else 1
            gl = sorted(((g[k_] - (ax, ay)[k_]) * (tx if horiz else ty)) for g in grids)
            gl = [g for g in gl if 5 < g < Lr - 5]
        at = []
        for (a1, m1), (a2, m2) in zip(beside, beside[1:]):
            if m1 != m2:
                c = [c for c in cuts if a1 < c < a2]
                if c:
                    at.append(min(c, key=lambda c: abs(c - (a1 + a2) / 2)))
                elif gl and all(any(abs(a_ - b_) <= 30 for b_ in r.get("_box_at", [])) for a_ in (a1, a2))                         and _nearest(rows, r, words, m1, a1, ax, ay, tx, ty) and _nearest(rows, r, words, m2, a2, ax, ay, tx, ty):
                    g = [g for g in gl if a1 < g < a2]
                    if g:
                        at.append(min(g, key=lambda g: abs(g - (a1 + a2) / 2)))
                        r["bay"] = True
        if not at:
            out.append(r); continue
        pts = [0.0] + sorted(set(at)) + [Lr]
        for s0, s1 in zip(pts, pts[1:]):
            out.append({**r, "a_pt": [round(ax + s0 * tx, 2), round(ay + s0 * ty, 2)], "b_pt": [round(ax + s1 * tx, 2), round(ay + s1 * ty, 2)],
                        "cuts": [c for c in r.get("cuts", []) if s0 < (c[0] - ax) * tx + (c[1] - ay) * ty < s1], "split": True})
    return out


def _nearest(rows, r, words, mark, al, ax, ay, tx, ty):
    """The mark word at `al` along r lies nearer to r than to any other parallel line passing it."""
    for w in words:
        if mark not in w[4]:
            continue
        cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
        if abs((cx - ax) * tx + (cy - ay) * ty - al) > 0.5:
            continue
        d0 = abs(-(cx - ax) * ty + (cy - ay) * tx)
        for q in rows:
            if q is r:
                continue
            (qx, qy), (px_, py_) = q["a_pt"], q["b_pt"]
            Lq = math.dist((qx, qy), (px_, py_)) or 1e-9
            ux, uy = (px_ - qx) / Lq, (py_ - qy) / Lq
            if abs(ux * ty - uy * tx) > 0.035:
                continue
            a_ = (cx - qx) * ux + (cy - qy) * uy
            if -2 <= a_ <= Lq + 2 and abs(-(cx - qx) * uy + (cy - qy) * ux) < d0:
                return False
        return True
    return False


BRACE_MARK = re.compile(r"^BR\d", re.I)            # MARK-TYPE: bracing -> a diagonal
BEAM_MARK = re.compile(r"^B\d", re.I)               # beam -> a level (or sloped) line, never upright


def type_ok(mark, r):
    """MARK-TYPE (the curator 2026-11-23, C2.6): on an elevation / section a mark's letters say which way its member runs: C / SC
    (column) an upright line, BR (bracing) a diagonal, B (beam) a level or sloped line - never a line running the other way.
    Other marks: any line."""
    a = r["ang"] % 180
    upright, level = abs(a - 90) < 5, min(a, 180 - a) < 5
    if UPRIGHT_MARK.match(mark):
        return upright or bool(r.get("outline")) and not level
    if BRACE_MARK.match(mark):
        return not upright and not level
    if BEAM_MARK.match(mark):
        return not upright
    return True


def pair(rows, words, mark_re, elev=False):
    """Each mark -> the nearest parallel line beside it. Second candidate nearly as close -> 'ambiguous'."""
    mk = _marks(words, mark_re)
    used, cand = set(), []
    for i, (w, _, _) in enumerate(mk):
        cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
        for j, r in enumerate(rows):
            (ax, ay), (bx, by) = r["a_pt"], r["b_pt"]
            Lr = math.dist((ax, ay), (bx, by))
            tx, ty = (bx - ax) / Lr, (by - ay) / Lr
            al = (cx - ax) * tx + (cy - ay) * ty
            of = abs(-(cx - ax) * ty + (cy - ay) * tx) - (r["hw"] if r.get("outline") else 0)
            of = max(of, 0.0)
            dang = abs(((w[5] % 180) - r["ang"] + 90) % 180 - 90)
            short = len(mk[i][1]) <= 4 and r.get("outline") and not mk[i][2]   # written level beside outlined webs / posts (Job A C1)
            # COL-LEVEL (the curator 2026-11-22, p6 SC6): a column / post mark written level beside an upright single line labels
            # it: the member's type first (upright line + column mark), then its mark; only after every mark written along
            # its own line is placed
            col = (not short and dang > ANG and abs(r["ang"] - 90) < 5 and abs(((w[5] % 180) + 90) % 180 - 90) < 5
                   and not mk[i][2] and UPRIGHT_MARK.match(mk[i][1]))
            if elev and not type_ok(mk[i][1], r):
                continue
            if -2 <= al <= Lr + 2 and of <= OFF and (dang <= ANG or short or col):
                cand.append((of + (0 if dang <= ANG else ACROSS if short else LAST), i, j))
    by_mark = {}
    for of, i, j in cand:
        by_mark.setdefault(i, []).append((of, j))
    for of, i, j in sorted(cand):
        if i in used or "mark" in rows[j]:
            continue
        w, mark, typ = mk[i]
        others = sorted(o for o, jj in by_mark[i] if jj != j and "mark" not in rows[jj])
        rows[j].update(mark=mark, label_pt=[round((w[0] + w[2]) / 2, 1), round((w[1] + w[3]) / 2, 1)], label_off=round(of, 1),
                       how="beside", typ=typ)
        if others and others[0] < 1.3 * of + 1.5:
            rows[j]["ambiguous"] = True
        used.add(i)
    # a mark left without a line takes its best line when the mark holding it can move to a free line of its own
    # (Job A truss: C1 written by a web sits 5 pt from the bottom chord, BC1 9 pt below it: C1 -> web, BC1 -> chord)
    holder = {j: i for j, r in enumerate(rows) for i in [next((ii for ii in used if rows[j].get("label_pt") == [round((mk[ii][0][0] + mk[ii][0][2]) / 2, 1),
                                                                                                         round((mk[ii][0][1] + mk[ii][0][3]) / 2, 1)]), None)] if i is not None}
    for i in [i for i in range(len(mk)) if i not in used]:
        for of, j in sorted(by_mark.get(i, [])):
            h = holder.get(j)
            if h is None:
                continue
            alt = sorted((o2, j2) for o2, j2 in by_mark.get(h, []) if j2 != j and "mark" not in rows[j2])
            if not alt:
                continue
            o2, j2 = alt[0]
            wh, mh, th = mk[h]
            rows[j2].update(mark=mh, label_pt=[round((wh[0] + wh[2]) / 2, 1), round((wh[1] + wh[3]) / 2, 1)], label_off=round(o2, 1), how="beside", typ=th)
            w, mark, typ = mk[i]
            rows[j].update(mark=mark, label_pt=[round((w[0] + w[2]) / 2, 1), round((w[1] + w[3]) / 2, 1)], label_off=round(of, 1), how="beside", typ=typ)
            holder[j2], holder[j] = h, i
            used.add(i)
            break
    return [m for i, m in enumerate(mk) if i not in used]


def leaders(sheet, box, free, rows):
    """A mark with no line beside it: follow its leader (thin black line starting at the label) to the member it
    points at. Returns the marks still free."""
    s = sheet.segs
    thin = s[(s[:, 5:8].max(1) < 0.15) & (s[:, 4] < 0.5)]
    thin = [tuple(map(float, r[:4])) for r in thin if box[0] <= r[0] <= box[2] and box[1] <= r[1] <= box[3] and math.dist(r[:2], r[2:4]) > 2]
    left = []
    for w, mark, typ in free:
        # LEADER-R01 (the curator 2026-11-23, 'just follow the arrow'): the leader may start at any line of the note the mark is
        # written in (HANGER / B2d (TYP): the arrow leaves from HANGER) - search from the whole note block
        nb = [v for v in (note_block(sheet.words, w) if w[6] else [w])            # other marks stacked by it are theirs
              if v is w or not re.fullmatch(r"[A-Z]{1,4}\d+[a-zA-Z]?\s*(\(?TYP\.?\)?)?|\(?TYP\.?\)?", v[4].strip())]
        bb = (min(v[0] for v in nb) - 6, min(v[1] for v in nb) - 6, max(v[2] for v in nb) + 6, max(v[3] for v in nb) + 6)
        inside = lambda x, y: bb[0] <= x <= bb[2] and bb[1] <= y <= bb[3]
        start = [(l, (l[2], l[3])) for l in thin if inside(l[0], l[1]) and not inside(l[2], l[3])]
        start += [(l, (l[0], l[1])) for l in thin if inside(l[2], l[3]) and not inside(l[0], l[1])]
        hit = None
        for l0, tip in start:
            seen = {l0}
            for _ in range(6):                      # follow a bent leader, piece by piece
                nxt = next(((l, (l[2], l[3]) if math.dist(tip, l[:2]) < 1 else (l[0], l[1])) for l in thin if l not in seen
                            and (math.dist(tip, l[:2]) < 1 or math.dist(tip, l[2:4]) < 1)), None)
                if not nxt:
                    break
                seen.add(nxt[0]); tip = nxt[1]
            # through an arrowhead: a short piece whose middle the tip reaches carries on to its far point
            ah = [l for l in thin if l not in seen and math.dist(l[:2], l[2:4]) < 8 and _seg_dist(tip, l[:2], l[2:4]) < 1.0]
            for l in ah:
                far = max((l[:2], l[2:4]), key=lambda q: math.dist(q, tip))
                if math.dist(far, tip) < 8:
                    tip = far
            # (a short piece at the tip loses to a long member: a leader names a member, Job A crane runway beam)
            best = min(((_seg_dist(tip, r["a_pt"], r["b_pt"]) - (r.get("hw", 0) if r.get("outline") else 0)
                         + (6.0 if math.dist(r["a_pt"], r["b_pt"]) < LEADER_SHORT_PT else 0.0), j)
                        for j, r in enumerate(rows)), default=(99, None))
            if best[0] <= LEADER_TIP and (hit is None or best[0] < hit[0]):
                hit = (best[0], best[1], tip)
        if hit is not None and " " in mark and math.dist(rows[hit[1]]["a_pt"], rows[hit[1]]["b_pt"]) < LEADER_SHORT_PT:
            hit = None                                       # a named member is not a short piece at the leader's tip
        if hit is None and " " in mark:                      # a member named in words (CRANE RUNWAY BEAM): its leader not
            cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2   # followed -> the nearest long level line running past the label
            cand = [(abs((r["a_pt"][1] + r["b_pt"][1]) / 2 - cy), j) for j, r in enumerate(rows) if "mark" not in r
                    and abs(r["a_pt"][1] - r["b_pt"][1]) < 1 and min(r["a_pt"][0], r["b_pt"][0]) - 5 <= cx <= max(r["a_pt"][0], r["b_pt"][0]) + 5
                    and abs(r["a_pt"][0] - r["b_pt"][0]) >= 30 and abs((r["a_pt"][1] + r["b_pt"][1]) / 2 - cy) <= 80]
            if cand:
                d_, j = min(cand)
                rows[j].update(mark=mark, label_pt=[round(cx, 1), round(cy, 1)], label_off=round(d_, 1), how="name (NAME-R01)", typ=True)
                continue
        if hit and "mark" not in rows[hit[1]]:
            rows[hit[1]].update(mark=mark, label_pt=[round((w[0] + w[2]) / 2, 1), round((w[1] + w[3]) / 2, 1)],
                                label_off=round(hit[0], 1), how="leader", typ=typ, leader_tip=[round(v, 1) for v in hit[2]])
        else:
            left.append((w, mark, typ))
    return left


def typ_fill(rows):
    """A (TYP) mark labels every unlabelled line of the same length and (mirrored) angle in the view."""
    n = 0
    for r in [r for r in rows if r.get("typ")]:
        Lr = math.dist(r["a_pt"], r["b_pt"])
        for o in rows:
            if o.get("mark"):
                continue
            Lo = math.dist(o["a_pt"], o["b_pt"])
            da = min(abs(o["ang"] - r["ang"]), abs(o["ang"] - (180 - r["ang"])))
            diag = 10 < r["ang"] < 170                        # a TYP brace labels its family: every diagonal of the truss
            slant = lambda q: 10 < q["ang"] % 180 < 80 or 100 < q["ang"] % 180 < 170            # a true diagonal, not a vertical
            if slant(r) and slant(o) and 0.5 * Lr <= Lo <= 2.0 * Lr:    # (the curator C2.6: BR1 (TYP) = all its diagonals)
                o.update(mark=r["mark"], how="typ", typ_of=r.get("id")); n += 1
                continue
            if (abs(Lo - Lr) <= (0.20 if diag else 0.05) * Lr) and da <= (10 if diag else 3) and (diag == (10 < o["ang"] < 170)):
                o.update(mark=r["mark"], how="typ", typ_of=r.get("id")); n += 1
    return n


def end_symbols(rows, short):
    """Short pieces at a member end are its connection symbol (tick / triangle / box), kept as evidence on that end."""
    for r in rows:
        r["sym"] = [0, 0]
    for s in short:
        c = ((s["a_pt"][0] + s["b_pt"][0]) / 2, (s["a_pt"][1] + s["b_pt"][1]) / 2)
        best = None
        for j, r in enumerate(rows):
            for k, e in enumerate((r["a_pt"], r["b_pt"])):
                d = math.dist(c, e)
                if d < 10 and (best is None or d < best[0]):
                    best = (d, j, k)
        if best:
            rows[best[1]]["sym"][best[2]] += 1


def columns(sheet, box):
    x0, y0, x1, y1 = box
    out = []
    for f in sheet.fill:
        fx0, fy0, fx1, fy1, grey, npts = f[:6]
        w, h = fx1 - fx0, fy1 - fy0
        if grey < 0.15 and f[7] == 0 and 5 <= w <= 22 and 5 <= h <= 22 and npts >= 8 and x0 <= fx0 and fx1 <= x1 and y0 <= fy0 and fy1 <= y1:
            out.append([round(float(fx0 + fx1) / 2, 2), round(float(fy0 + fy1) / 2, 2), round(float(w), 1), round(float(h), 1)])
    return out


def schedule_boxes(sheet):
    """Member schedule tables printed on the sheet (header MARKING + MEMBER SIZE): their lines and marks are not
    members. A table runs down its MARKING column while rows follow at < 30 pt."""
    if getattr(sheet, "_sched", None) is None:
        out = []
        for h in [w for w in sheet.words if w[4].strip().upper() == "MARKING"]:
            sz = next((w for w in sheet.words if w[4].strip().upper() == "MEMBER SIZE" and abs(w[1] - h[1]) < 3 and 0 < w[0] - h[2] < 200), None)
            if not sz:
                continue
            col = sorted(w for w in sheet.words if abs(w[0] - h[0]) < 25 and w[1] > h[1])
            col.sort(key=lambda w: w[1])
            y = h[3]
            for w in col:
                if w[1] - y > 30 or not re.fullmatch(r"[A-Z]{1,3}\d{0,3}[a-zA-Z]?(\s*/\s*[A-Z]{1,3}\d{0,3}[a-zA-Z]?)*", w[4].strip()):
                    break                                   # the table ends at the first row that is not a mark
                y = w[3]
            out.append([h[0] - 20, h[1] - 12, sz[2] + 140, y + 12])
        sheet._sched = out
    return sheet._sched


LABEL_BOX_MM = 2200.0  # a short outline up to this long, lying on a longer line: a section symbol / label box (D39)


def label_boxes(rows, k):
    """LABEL-BOX (the curator D39, Turbine grid 2' SHD+22.230): a short outlined box (<= LABEL_BOX_MM) lying on a longer parallel
    line (the line runs inside the box or along its face; the box at least 2.5 x as long as deep: a raised member on posts
    is not) is the section symbol / label box drawn on that member, not steel. Marks are paired
    after these are set aside, so the label written by the box names the line itself. -> boxes (flagged not_member)."""
    L = lambda r: math.dist(r["a_pt"], r["b_pt"])
    out = []
    for r in rows:
        if not r.get("outline") or L(r) * k > LABEL_BOX_MM or 2 * r["hw"] > 0.4 * L(r):   # long and thin, as a symbol
            continue
        (ax, ay), (bx, by) = r["a_pt"], r["b_pt"]
        cx, cy = (ax + bx) / 2, (ay + by) / 2
        for q in rows:
            if q is r or L(q) < 2 * L(r) or abs(((q["ang"] - r["ang"]) + 90) % 180 - 90) > 2:
                continue
            (qx, qy), (px_, py_) = q["a_pt"], q["b_pt"]
            Lq = L(q)
            tx, ty = (px_ - qx) / Lq, (py_ - qy) / Lq
            s0 = sorted(((ax - qx) * tx + (ay - qy) * ty, (bx - qx) * tx + (by - qy) * ty))
            if abs(-(cx - qx) * ty + (cy - qy) * tx) <= r["hw"] + 1.0 and s0[0] >= -2 and s0[1] <= Lq + 2:
                r["not_member"] = "label box on a member line (D39)"
                q.setdefault("_box_at", []).append((s0[0] + s0[1]) / 2)     # where its labels sit (BAY-SPLIT)
                out.append(r)
                break
    return out


def _read(sheet, box, P3, k, mark_re, view, min_member_mm, fit, outlines=True, detail=False):
    """Shared core: lines -> rows with marks, mapped to 3D by P3(px, py) -> [x, y, z]."""
    sched = schedule_boxes(sheet)
    in_s = lambda x, y: any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in sched)
    words = [w for w in sheet.words if box[0] <= w[0] <= box[2] and box[1] <= w[1] <= box[3] and not in_s(w[0], w[1])]
    bubs = [(b[0], b[1]) for b in bubbles(sheet, box)]
    raw = lines(sheet, box)
    n_sched = sum(1 for l in raw if in_s(l[0], l[1]) and in_s(l[2], l[3]))
    raw = [l for l in raw if not (in_s(l[0], l[1]) and in_s(l[2], l[3]))]
    in_bub = lambda l: any(math.dist(b, l[:2]) < 16 and math.dist(b, l[2:4]) < 16 for b in bubs)
    n_bub = sum(map(in_bub, raw))
    raw = [l for l in raw if not in_bub(l)]
    spl_boxes = splice_boxes(sheet, box)                  # SPLICE-R01: X boxes on members; their X is never a member
    in_x = lambda l: any(abs(min(l[0], l[2]) - b[0]) <= 1.5 and abs(min(l[1], l[3]) - b[1]) <= 1.5 and abs(max(l[0], l[2]) - b[2]) <= 1.5
                         and abs(max(l[1], l[3]) - b[3]) <= 1.5 for b in spl_boxes)
    raw = [l for l in raw if not in_x(l)]
    joined = join(raw, words, spl_boxes)
    if outlines:
        ol = outline_lines(sheet, box, k)
        ol = [o for o in ol if not (in_s(*o["a_pt"]) and in_s(*o["b_pt"]))]
        joined += ol
    titles = [w for w in words if w[6] >= 12.5 and len(w[4]) > 8]
    under = lambda r: any(t[0] - 4 <= min(r["a_pt"][0], r["b_pt"][0]) and max(r["a_pt"][0], r["b_pt"][0]) <= t[2] + 4
                          and -2 <= r["a_pt"][1] - t[3] <= 6 and abs(r["a_pt"][1] - r["b_pt"][1]) < 0.5 for t in titles)
    n_title = sum(map(under, joined))
    joined = [r for r in joined if not under(r)]
    # SYMBOL-R01 (the curator 2026-11-22, p10 S3): the bar of a section / detail marker runs through its bubble's centre and
    # stops a little past it (both ends within 2.2 r of the centre): an annotation line, never steel
    circ = [c for c in circles(sheet) if box[0] <= c[0] <= box[2] and box[1] <= c[1] <= box[3]]
    bar = lambda r: any(_seg_dist((cx, cy), r["a_pt"], r["b_pt"]) <= 1.5 and math.dist((cx, cy), r["a_pt"]) <= 2.2 * cr
                        and math.dist((cx, cy), r["b_pt"]) <= 2.2 * cr for cx, cy, cr in circ)
    n_bar = sum(map(bar, joined))
    joined = [r for r in joined if not bar(r)]
    rows = [r for r in joined if math.dist(r["a_pt"], r["b_pt"]) * k >= min_member_mm]
    short = [r for r in joined if math.dist(r["a_pt"], r["b_pt"]) * k < min_member_mm]
    boxes = label_boxes(rows, k)                          # section symbols / label boxes on a member: not steel (D39)
    rows = [r for r in rows if not r.get("not_member")]
    rows = split_marks(rows, words, mark_re, grids=bubs)
    for i, r in enumerate(rows):
        r["id"] = f"p{sheet.page}-{view}-L{i + 1:03d}"
    free = pair(rows, words, mark_re, elev=view.startswith("E"))
    free = leaders(sheet, box, free, rows)
    if view.startswith("E"):                              # NAME-R02: a named beam (monorail) is level; on an elevation an upright
        for r in rows:                                    # line its note points at is a hanger / its end-on symbol, not it
            if r.get("mark") and not re.fullmatch(mark_re, r["mark"]) and abs(r["a_pt"][0] - r["b_pt"][0]) < 1.0:
                free.append(((0, 0, 0, 0, r["mark"], 0, 0), r["mark"], False))
                for k_ in ("mark", "label_pt", "label_off", "how", "typ"):
                    r.pop(k_, None)
    n_typ = typ_fill(rows)
    end_symbols(rows, short)
    # member pen (learnt per view): the pen carrying most marked outline length; an outline drawn with another pen is a
    # dimension, detail, note or symbol line - not a member, whatever mark sits near it (Job A p4: members 0.3, dims 0.12)
    from collections import Counter as _C
    pens = _C()
    for r in rows:
        if r.get("outline") and r.get("mark"):
            pens[r["pen"]] += math.dist(r["a_pt"], r["b_pt"])
    member_pen = pens.most_common(1)[0][0] if pens else None
    n_pen = 0
    if member_pen is not None:
        marks_ = [w for w in words if re.fullmatch(rf"({mark_re})\s*(\(?TYP\.?\)?)?", w[4].strip())]

        def boxed(r):                                     # writing between its faces (1260x1260 detail box): not a member
            if detail:                                    # (a 1:20 detail writes dimensions inside its members)
                return False
            (ax, ay), (bx, by) = r["a_pt"], r["b_pt"]
            Lr = math.dist((ax, ay), (bx, by)) or 1e-9
            tx, ty = (bx - ax) / Lr, (by - ay) / Lr
            for w in words:
                if w in marks_:
                    continue
                cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
                if abs(-(cx - ax) * ty + (cy - ay) * tx) < r["hw"] * 0.9 and 0 < (cx - ax) * tx + (cy - ay) * ty < Lr:
                    return True
            return False
        ol_len = sum(math.dist(r["a_pt"], r["b_pt"]) for r in rows if r.get("outline") and r.get("mark"))
        ln_len = sum(math.dist(r["a_pt"], r["b_pt"]) for r in rows if not r.get("outline") and r.get("mark"))
        outline_view = ol_len > 9 * ln_len                # members drawn as outlines here: single lines are not members
        off = [r for r in rows if (r.get("outline") and (abs(r["pen"] - member_pen) > 0.05 or boxed(r)))
               or (outline_view and not r.get("outline"))]
        if off:                                           # drop them and pair the marks again on the members only
            n_pen = len(off)
            keep = [r for r in rows if r not in off]
            for r in keep:
                for f in ("mark", "label_pt", "label_off", "how", "typ", "ambiguous"):
                    r.pop(f, None)
            free = pair(keep, words, mark_re)
            free = leaders(sheet, box, free, keep)
            n_typ = typ_fill(keep)
            for r in off:
                for f in ("mark", "label_pt", "label_off", "how", "typ", "ambiguous"):
                    r.pop(f, None)
                r["not_member"] = (f"pen {r['pen']} (members {member_pen})" if r.get("outline") and abs(r["pen"] - member_pen) > 0.05
                                   else "writing inside it (a box)" if r.get("outline") else "single line in a view drawn in outlines")
            rows = keep + off
    for i, r in enumerate(boxes):
        r["id"] = f"p{sheet.page}-{view}-X{i + 1:03d}"
        r["sym"] = [0, 0]
    rows = rows + boxes
    n_box, n_bay = len(boxes), sum(1 for r in rows if r.get("bay"))
    out = []
    for r in rows:
        a, b = P3(*r["a_pt"]), P3(*r["b_pt"])
        status = "not a member: " + r["not_member"] if r.get("not_member") else \
            ("ambiguous" if r.get("ambiguous") else r.get("how", "beside")) if r.get("mark") else "unlabelled"
        L = round(math.dist(a, b)) if None not in a + b else round(math.dist(a[:2], b[:2]))
        out.append({"id": r["id"], "kind": "outline" if r.get("outline") else "line", "mark": r.get("mark"), "a_pt": r["a_pt"], "b_pt": r["b_pt"],
                    "a": a, "b": b, "L": L, "label_pt": r.get("label_pt"), "label_off": r.get("label_off"),
                    "end_symbols": r["sym"], "status": status, "depth_mm": round(2 * r["hw"] * k) if r.get("outline") else None,
                    **({"bay": True} if r.get("bay") else {}),
                    **({"splices": [P3(*sp) for sp in r["splices"]], "splices_pt": r["splices"]} if r.get("splices") else {})})
    fit.update(pieces=len(raw) + n_bub + n_sched, schedule_pieces=n_sched, bubble_pieces=n_bub, title_underlines=n_title, marker_bars=n_bar, splice_symbols=len(spl_boxes), short_symbols=len(short), lines=len(rows),
               labelled_beside=sum(1 for r in rows if r.get("how") == "beside"), by_leader=sum(1 for r in rows if r.get("how") == "leader"),
               by_typ=n_typ, ambiguous=sum(1 for r in rows if r.get("ambiguous")), unlabelled=sum(1 for r in rows if not r.get("mark")),
               free_marks=[m for _, m, _ in free], member_pen=member_pen, not_member_pen=n_pen, label_boxes=n_box, bay_splits=n_bay)
    leg = legend_lines(sheet, box, P3)                     # LEGEND-R01: lines in a legend's style (handrail): listed, not members
    if leg:
        fit["legend_lines"] = leg
        fit["legend"] = {nm: {"runs": sum(1 for e in leg if e["name"] == nm), "mm": round(sum(e["L_pt"] for e in leg if e["name"] == nm) * k)}
                         for nm in sorted({e["name"] for e in leg})}
    return out, fit


def read_plan(sheet, box, GX, GY, z, mark_re, scale=75, view="V", min_member_mm=500, y_anchor=None):
    """One plan view -> (rows, fit). Lines shorter than min_member_mm are symbols (end ticks), pieces inside grid
    bubbles are skipped (counted). fit["T"] = (sx, ox, sy, oy): x = sx px + ox, y = sy py + oy (for cut marks)."""
    P, fit = calibrate(sheet, box, GX, GY, scale, y_anchor)
    return _read(sheet, box, lambda px, py: [*P(px, py), z], fit["stated_mm_per_pt"], mark_re, view, min_member_mm, fit)


# LEVEL-PREFIX: the word a job writes its levels with ("SHD+4.850", "EL+104.300M"); one setting for every reader, set by
# build_job.py from job.json level_prefix (environment STEEL_LEVEL_PREFIX, so the reader subprocesses get it too)
LEVEL_PREFIX = os.environ.get("STEEL_LEVEL_PREFIX") or "SHD"
LEVEL_RE = re.compile(rf"\b(?:{LEVEL_PREFIX})\.?\s*\+\s*(\d+\.\d+)", re.I)


def level_pairs(sheet, box, datum):
    """Level markers in a view -> [(py of the level line, z mm, text)]: the long thin line along the text's bottom
    edge (Job A: 0.12 pt, 0.24 pt above the text bottom); without one, the text bottom."""
    s = sheet.segs
    hz = s[(abs(s[:, 1] - s[:, 3]) < 0.2) & (abs(s[:, 2] - s[:, 0]) > 40)]
    out = []
    for w in sheet.words:
        m = LEVEL_RE.search(w[4])
        if m and re.search(r"=|\bCD\b", w[4]):
            continue                                       # a note ("SHD +4.000M = CD +5.741M"), not a level marker
        if not m or w[5] % 180 or not (box[0] <= w[0] <= box[2] and box[1] <= w[1] <= box[3]):
            continue
        near = hz[(abs(hz[:, 1] - w[3]) < 1.5) & (np.minimum(hz[:, 0], hz[:, 2]) < w[2] + 30) & (np.maximum(hz[:, 0], hz[:, 2]) > w[0] - 30)]
        py = float(near[np.argmin(abs(near[:, 1] - w[3])), 1]) if len(near) else float(w[3])
        out.append((py, round((float(m.group(1)) - datum) * 1000), m.group(1)))
    return out


def calibrate_elev(sheet, box, H, scale, datum, fallback=False, no_level="raise"):
    """Elevation / section: horizontal from the grid bubbles of table H (either viewing direction), vertical from the
    level markers. -> (h(px), z(py), fit).
    fallback: when no two bubbles agree with the scale (close grids drawn staggered, one bubble only): the scale of the
    level markers (>= 2 levels) or the stated one, anchored on the bubbles (median); facing from the bubbles' order,
    one bubble only: facing +1, flagged (VIEW-R01).
    no_level "relative": a view with no level marker (bracing truss drawn alone) gets z = 0 at its top line (fit
    "z_rel"); the caller sets its level."""
    stated = 25.4 / 72 * scale
    Hn = {grid_name(k): v for k, v in H.items()}
    bx = [(b[0], Hn[b[2]], b[2]) for b in bubbles(sheet, box) if b[2] in Hn]
    lv = level_pairs(sheet, box, datum)
    if not lv:                                             # level markers written once for views side by side (p28 A | B)
        lv = level_pairs(sheet, [0, box[1], sheet.size[0], box[3]], datum)
    zfit = None
    if len({t for _, _, t in lv}) >= 2:
        if not fallback:
            zfit = _fit_axis([(p, z, t) for p, z, t in lv], stated, -1)
        else:                                       # the levels' own scale (any stated scale)
            A = np.array([[p, 1.0] for p, _, _ in lv]); y = np.array([z for _, z, _ in lv], float)
            (sz_, oz_), *_ = np.linalg.lstsq(A, y, rcond=None)
            r = A @ np.array([sz_, oz_]) - y
            zfit = (sz_, oz_, sorted({t for *_, t in lv}), [], float(np.sqrt((r ** 2).mean())))
    best = None
    for sign in (1, -1):
        try:
            r = _fit_axis(bx, stated, sign)
            if best is None or len(r[2]) > len(best[0][2]):
                best = (r, sign)
        except ValueError:
            pass
    method = "fit"
    if best is None:
        if not fallback or not bx:
            raise ValueError("needs placement: no two grid bubbles agree with the stated scale (" + ", ".join(sorted({b[2] for b in bx})) + ")")
        sc = abs(zfit[0]) if zfit and abs(abs(zfit[0]) / stated - 1) < 0.6 else stated
        names = sorted({(p, m) for p, m, _ in bx})
        sign, method = 1, "one bubble, facing assumed (VIEW-R01)"
        if len({m for _, m in names}) >= 2:
            ps = np.array([p for p, _ in names]); ms = np.array([m for _, m in names])
            sign, method = (1 if np.corrcoef(ps, ms)[0, 1] > 0 else -1), "scale of the levels, anchored on staggered bubbles"
        offs = sorted(m - sign * sc * p for p, m, _ in bx)
        oh = offs[len(offs) // 2]
        dev = [abs(sign * sc * p + oh - m) for p, m, _ in bx]       # the anchor (middle bubble) sits on its line; the
        best = ((sign * sc, oh, sorted({n for *_, n in bx}), [], float(min(dev))), sign)   # others are staggered off theirs
        stated = sc
    (sh_, oh, uh, dh, rh), sign = best
    if zfit:
        sz, oz, uz, dz, rz = zfit
    elif lv:                                               # one level written (truss views): that level + the stated scale
        sz, uz, dz, rz = -stated, [lv[0][2]], [], 0.0
        oz = lv[0][1] + stated * lv[0][0]
    elif no_level == "relative":
        # the top member line (job review R04: the widened box took a dimension line 4 m above the truss as its top):
        # long level member lines only - member pieces (layer pass, member pen >= 0.5 pt) and filled member bars
        s_ = member_segs(sheet)
        s_ = s_[(s_[:, 5:8].max(1) < 0.15) & (s_[:, 4] >= 0.5)]
        q_ = sheet.quad
        lv_ = [(float(r[1]), float(r[0]), float(r[2])) for r in s_] + [(float(r[1]), float(r[0]), float(r[2])) for r in q_ if 0.4 <= r[4] <= 2.4]
        lv_ = [y for y, xa, xb in lv_ if box[0] <= min(xa, xb) and max(xa, xb) <= box[2] and box[1] <= y <= box[3] and abs(xb - xa) > 100]
        lvq = [float(r[1]) for r in list(s_) + [r for r in q_ if 0.4 <= r[4] <= 2.4] if abs(float(r[1]) - float(r[3])) < 0.3]
        top = min((y for y in lv_ if y in lvq), default=box[1])
        sz, oz, uz, dz, rz = -stated, stated * top, ["top line = 0 (VIEW-R02)"], [], 0.0
    else:
        raise ValueError("needs placement: no level marker in the view")
    sh_, oh, sz, oz = map(float, (sh_, oh, sz, oz))
    fit = {"h_mm_per_pt": round(sh_, 3), "z_mm_per_pt": round(-sz, 3), "stated_mm_per_pt": round(stated, 3), "h_grids": uh,
           "levels": uz, "dropped": dh + dz, "rms_mm": round(max(rh, rz), 1), "facing": sign, "method": method,
           "scale_err_pct": round(100 * max(abs(abs(sh_) - stated), abs(-sz - stated)) / stated, 2), "T": [sh_, oh, sz, oz]}
    if no_level == "relative" and not lv:
        fit["z_rel"] = True
    lim = 300 if method != "fit" else 50                   # staggered bubbles sit up to ~1 grid gap off their lines
    if fit["scale_err_pct"] > 2.0 or fit["rms_mm"] > lim:
        raise ValueError(f"calibration off: scale {fit['scale_err_pct']} %, rms {fit['rms_mm']} mm ({method})")
    return (lambda px: round(sh_ * px + oh, 1)), (lambda py: round(sz * py + oz, 1)), fit


def read_elev(sheet, box, H, axis, at, mark_re, scale, datum, view="E", min_member_mm=500, scale_read=True, no_level="raise"):
    """Elevation along a grid line / section on a cut line. axis = "x" when the horizontal of the view runs along X
    (H = GX, the view fixed at y = at), "y" when it runs along Y (H = GY, fixed x = at). scale_read False (no SCALE text
    found under the title): the common scales are tried and one is kept only when >= 3 grid bubbles agree with it."""
    try:
        h, zf, fit = calibrate_elev(sheet, box, H, scale, datum, no_level=no_level)
    except ValueError:
        nlev = len({t for *_, t in level_pairs(sheet, box, datum)})
        need = 4 if scale_read else (2 if nlev >= 2 else 3)  # a stated scale is overruled only by >= 4 agreeing bubbles (SCALE-R01);
        for sc in (100, 75, 50, 125, 150, 200):           # no scale written: 2 bubbles when 2 levels agree with the same scale
            if scale_read and sc == scale:
                continue
            try:
                h, zf, fit = calibrate_elev(sheet, box, H, sc, datum, no_level=no_level)
                if len(fit["h_grids"]) >= need:
                    fit["scale_found"] = (f"1:{sc}: {len(fit['h_grids'])} grid bubbles agree" +
                                          (f", the title says 1:{scale:g} (SCALE-R01)" if scale_read else " (no scale text)"))
                    break
            except ValueError:
                continue
        else:                                              # last: the levels' scale (or the stated one) on the bubbles there are
            h, zf, fit = calibrate_elev(sheet, box, H, scale, datum, fallback=True, no_level=no_level)
    P3 = (lambda px, py: [h(px), at, zf(py)]) if axis == "x" else (lambda px, py: [at, h(px), zf(py)])
    return _read(sheet, box, P3, fit["stated_mm_per_pt"], mark_re, view, min_member_mm, fit)

def outline_lines(sheet, box, k, dmin=80.0, dmax=1100.0, wmin=0.1, wmax=0.45, minlen_mm=400.0):
    """Members drawn as outlines (two thin parallel lines, Job A: black 0.3 pt; columns in elevations, truss chords and webs):
    the closest partner line whose spacing at the view scale (k mm/pt) is a member depth (dmin-dmax mm) and that overlaps it
    by >= 60 % of the shorter -> one member: centre line, half width (pt), extent = the overlap. Pieces of one drawn line
    are joined first (gap <= JOIN). Returns rows like join() with "outline": True."""
    s = member_segs(sheet)
    x0, y0, x1, y1 = box
    m = s[(s[:, 5:8].max(1) < 0.15) & (s[:, 4] >= wmin) & (s[:, 4] <= wmax)]
    m = [tuple(map(float, r[:4])) + (float(r[4]),) for r in m
         if x0 <= r[0] <= x1 and y0 <= r[1] <= y1 and x0 <= r[2] <= x1 and y0 <= r[3] <= y1 and math.dist(r[:2], r[2:4]) * k >= 0.5 * minlen_mm]
    L = join(m, [])                                          # joined drawn lines (each with its angle)
    smin, smax = dmin / k, dmax / k
    groups = {}
    for i, r in enumerate(L):
        a = r["ang"]
        t = (math.cos(math.radians(a)), -math.sin(math.radians(a)))
        n = (-t[1], t[0])
        (ax, ay), (bx, by) = r["a_pt"], r["b_pt"]
        off = ax * n[0] + ay * n[1]
        s0, s1 = sorted((ax * t[0] + ay * t[1], bx * t[0] + by * t[1]))
        groups.setdefault(round(a / 1.0), []).append((off, s0, s1, t, n, i))
    out, used = [], set()
    for g in groups.values():
        g.sort()
        for ii, (o1, a1, b1, t, n, i) in enumerate(g):
            if i in used:
                continue
            best = None
            for o2, a2, b2, _, _, j in g[ii + 1:]:
                d = o2 - o1
                if d > smax:
                    break
                if d < smin or j in used:
                    continue
                ov = min(b1, b2) - max(a1, a2)
                if ov >= 0.6 * min(b1 - a1, b2 - a2) and ov >= 0.3 * max(b1 - a1, b2 - a2) and ov * k >= minlen_mm:   # (a short detail
                                                                     # line between a chord's faces is not its partner)
                    best = (d, j, o2, max(a1, a2), min(b1, b2))
                    break                                    # the closest valid partner
            if best is None:
                continue
            d, j, o2, lo, hi = best
            used.update((i, j))
            oc = (o1 + o2) / 2
            p = (lo * t[0] + oc * n[0], lo * t[1] + oc * n[1])
            q = (hi * t[0] + oc * n[0], hi * t[1] + oc * n[1])
            out.append({"a_pt": [round(p[0], 2), round(p[1], 2)], "b_pt": [round(q[0], 2), round(q[1], 2)], "ang": L[i]["ang"],
                        "hw": d / 2, "outline": True, "cuts": [], "_k": (round(L[i]["ang"]), oc, lo, hi),
                        "pen": round(max(L[i]["hw"], L[j]["hw"]), 2)})
    # one member drawn with 4 lines (outer + inner flange faces) pairs twice: keep the wider, drop the copy inside it
    out.sort(key=lambda r: -r["hw"])
    keep = []
    for r in out:
        a, oc, lo, hi = r["_k"]
        dup = any(abs(a - b[0]) <= 1 and abs(oc - b[1]) < kr["hw"] and min(hi, b[3]) - max(lo, b[2]) > 0.8 * (hi - lo)
                  for kr in keep for b in (kr["_k"],))
        if not dup:
            keep.append(r)
    for r in keep:
        r.pop("_k")
    return keep

