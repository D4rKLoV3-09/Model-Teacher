#!/usr/bin/env python3
"""Drawing-style detection and the readers for styles ga_plan / read_drawing do not read on their own.
Learned on Job A Appendix 4 (consultant / consultant sheets); see .claude/rules/drawing-styles.md.

    python generate-model/drawing_styles.py PDF [--out styles.json]

Prints one line per style found (counts of sheets), writes the per-sheet flags to styles.json. Use the flags to pick
the reader before reading a package, instead of exploring it in the chat:
  rotated        page.rotation != 0           -> page.remove_rotation() first
  filled_quads   members as filled black quads -> quads(page) gives centre lines, hand them to ga_plan.merge
  outlined_text  little real text, many paths  -> OCR the title block (split_drawings), scale dimensions
  leader_labels  boxed marks with leader lines -> leader_pairs(page, MARK, symbols)
  thin_outlines  members as pairs of thin lines (1:75 truss elevations) -> parametric / ga_plan.bands

quads(page) -> [(x0, y0, x1, y1, w)] centre lines of thin filled quads + stroked black lines 0.5-1.2 wide.
leader_pairs(page, mark_re, symbols) -> [(label_text, label_xy, symbol_rect)] following each label's leader line.
"""
import argparse
import json
import math
import re

import pymupdf as fitz


def quads(page, wmin=0.4, wmax=2.4):
    """Centre lines of thin FILLED black quads (members drawn as filled shapes) + stroked black lines."""
    out = []
    for d in page.get_drawings():
        f, c = d.get("fill"), d.get("color")
        its = d["items"]
        if f is not None and max(f) < 0.15 and all(i[0] == "l" for i in its) and 3 <= len(its) <= 90:
            pts = []
            for i in its:
                for p in (i[1], i[2]):
                    if not any(abs(p.x - q[0]) < 0.05 and abs(p.y - q[1]) < 0.05 for q in pts):
                        pts.append((p.x, p.y))
            if 4 <= len(pts) <= 60:
                cx, cy = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
                sxx = sum((p[0] - cx) ** 2 for p in pts); syy = sum((p[1] - cy) ** 2 for p in pts)
                sxy = sum((p[0] - cx) * (p[1] - cy) for p in pts)
                th = 0.5 * math.atan2(2 * sxy, sxx - syy)      # principal axis (the diagonal tilted every member)
                tx, ty = math.cos(th), math.sin(th)
                p0 = (cx, cy)
                L = max(math.dist(p, q) for p in pts for q in pts)
                off = [-(p[0] - p0[0]) * ty + (p[1] - p0[1]) * tx for p in pts]
                al = [(p[0] - p0[0]) * tx + (p[1] - p0[1]) * ty for p in pts]
                w1, o = (max(off) - min(off)) / 2, (max(off) + min(off)) / 2
                if wmin <= w1 <= wmax and L > 5 * w1:
                    out.append((p0[0] + min(al) * tx - o * ty, p0[1] + min(al) * ty + o * tx,
                                p0[0] + max(al) * tx - o * ty, p0[1] + max(al) * ty + o * tx, 1.0))
        elif c is not None and max(c) < 0.15 and f is None:
            w = d.get("width") or 0
            for i in its:
                if i[0] == "l" and math.dist(i[1], i[2]) >= 14 and 0.5 <= w <= 1.2:
                    out.append((i[1].x, i[1].y, i[2].x, i[2].y, 1.0))
    return out


def leader_pairs(page, mark_re, symbols, near=40):
    """Each label matching mark_re -> the symbol at the far end of its leader (thin black diagonal starting at the box).
    symbols: list of fitz.Rect. Returns [(text, (lx, ly), rect | None)]."""
    diag = []
    for d in page.get_drawings():
        c = d.get("color")
        if c and max(c) < 0.1 and (d.get("width") or 0) < 1.0:
            for it in d["items"]:
                if it[0] == "l" and abs(it[2].x - it[1].x) > 3 and abs(it[2].y - it[1].y) > 3:
                    diag.append((it[1].x, it[1].y, it[2].x, it[2].y))
    out = []
    for w in page.get_text("words"):
        if not re.fullmatch(mark_re, w[4]):
            continue
        lx, ly = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
        best = None
        for s in diag:
            for nx, ny, fx, fy in ((s[0], s[1], s[2], s[3]), (s[2], s[3], s[0], s[1])):
                dn = math.hypot(nx - lx, ny - ly)
                if dn < near and math.hypot(fx - lx, fy - ly) > dn + 5 and (best is None or dn < best[0]):
                    best = (dn, fx, fy)
        if best is None:
            out.append((w[4], (lx, ly), None))
            continue
        _, fx, fy = best
        sym = min(symbols, key=lambda r: math.hypot((r.x0 + r.x1) / 2 - fx, (r.y0 + r.y1) / 2 - fy)) if symbols else None
        out.append((w[4], (lx, ly), sym))
    return out


def detect(page):
    """Style flags of one sheet (cheap: counts only)."""
    text = page.get_text()
    raw = page.read_contents()
    nfill = nstroke = nthin = 0
    for d in page.get_drawings():
        f, c = d.get("fill"), d.get("color")
        if f is not None and max(f) < 0.15 and 3 <= len(d["items"]) <= 12:
            nfill += 1
        elif c is not None and max(c) < 0.15:
            w = d.get("width") or 0
            nstroke += 0.5 <= w <= 1.2
            nthin += w < 0.2
    paths = raw.count(b" l\n") + raw.count(b" c\n")
    return {"rotated": page.rotation != 0, "filled_quads": nfill > 200 and nfill > nstroke,
            "outlined_text": len(text) < 800 and paths > 5000, "thin_outlines": nthin > 3 * max(1, nstroke),
            "leader_labels": bool(re.search(r"\b(SC|BP|C)\d+\b", text))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--out", default="styles.json")
    a = ap.parse_args()
    doc = fitz.open(a.pdf)
    flags = [detect(p) for p in doc]
    json.dump([{"page": i + 1, **f} for i, f in enumerate(flags)], open(a.out, "w"), indent=1)
    for k in flags[0]:
        pages = [i + 1 for i, f in enumerate(flags) if f[k]]
        if pages:
            print(f"{k}: {len(pages)} sheets (e.g. p{pages[0]})")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
