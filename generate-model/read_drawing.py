#!/usr/bin/env python3
"""Fast reader for vector PDF drawings (PyMuPDF). Replaces hand-cropping and eyeballing.

    python read_drawing.py index   PDF_OR_FOLDER [...] --out DIR [--prev OLD/sheets.json]
                                   START HERE: index.md = overview (about 60 tokens a sheet), index/<sheet>.md = one
                                   sheet in full (title block, views, marks, sizes, levels, dims, member pens; about
                                   1 k tokens), reading.json seeded, sheets.json, thumbnails. Unchanged PDFs come from
                                   the cache; a rerun marks the sheets that changed since the last index.
    python read_drawing.py find    DIR REGEX [--sheet S] [--view V]   every word + note, no PDF opened, no image
    python read_drawing.py crop    DIR --view ID | --find TEXT | --box x0 y0 x1 y1 [--part tl|tr|bl|br] [--max-px 1000]
                                   one view / one mark as an image; a crop taken before says SEEN BEFORE + its notes
    python read_drawing.py note    DIR --sheet S [--view V] "TEXT"     what a crop showed, kept in reading.json
    python read_drawing.py status  DIR                                 crops and notes per sheet, sheets not opened
    python read_drawing.py connections DIR                             connection detail views + joint tags (BP3 ...) from
                                   the index -> connections_found.json; read each detail ONCE, place it by its tag
    python read_drawing.py info    PDF OUTDIR                     page sizes, text per page, overview PNGs
    python read_drawing.py words   PDF PAGE REGEX [x0 y0 x1 y1]   words matching REGEX with centre (pt)
    python read_drawing.py render  PDF PAGE x0 y0 x1 y1 DPI OUT   crop to PNG (only to settle an ambiguity)
    python read_drawing.py members PDF PAGE x0 y0 x1 y1 OUT.json [--wmin 0.9] [--wmax 99] [--minlen 15]
                                   member lines (dashes merged) each paired with its parallel mark label

Coordinates are PDF points (y down). Calibrate once per view: scale = known dim / pt distance
between two grid lines, then X = (px - px_grid) * scale, Y = (py_grid - py) * scale. Levels in a
section: two known levels (FFL / TOS / BOB markers) give z(py) the same way.
Member lines: black only (grey = existing / by others), width >= wmin. Labels: text lines starting
with --prefix (default "X-"), matched when parallel (|cos| > 0.97), within the line's length and
<= 9 pt off it.
"""
import argparse
import json
import math
import re
from pathlib import Path

import pymupdf as fitz  # PyMuPDF (pypdf / poppler are not installed on the curator's PC); the old `fitz` name is deprecated


def segments(page):
    """Every straight black segment: (x0, y0, x1, y1, width)."""
    out = []
    for d in page.get_drawings():
        col = d.get("color") or ()
        if col and max(col) > 0.1:
            continue
        w = d.get("width") or 0
        for it in d["items"]:
            if it[0] == "l":
                out.append((it[1].x, it[1].y, it[2].x, it[2].y, w))
            elif it[0] == "re":
                r = it[1]
                out += [(r.x0, r.y0, r.x1, r.y0, w), (r.x1, r.y0, r.x1, r.y1, w),
                        (r.x1, r.y1, r.x0, r.y1, w), (r.x0, r.y1, r.x0, r.y0, w)]
    return out


def merge_lines(segs, gap=12, ang_tol=0.6, off_tol=0.4, bridge=None):
    """Join collinear pieces into one line each, in O(n log n): sort by angle, then by offset, then join the
    intervals along the line that overlap or leave a gap <= `gap` pt. segs = [(x0, y0, x1, y1, ...)].
    Shared by `merged` (dashed lines, gap 12) and ga_plan.merge (member lines, gap 2).
    bridge(p, q) -> True also joins across a wider gap from point p to point q (a label lying on the line)."""
    S = sorted((math.degrees(math.atan2(a[3] - a[1], a[2] - a[0])) % 180, a) for a in segs)
    groups, cur = [], []
    for g in S:                                      # angle clusters
        if cur and g[0] - cur[-1][0] > ang_tol:
            groups.append(cur); cur = []
        cur.append(g)
    if cur:
        groups.append(cur)
    if len(groups) > 1 and groups[-1][-1][0] > 180 - ang_tol and groups[0][0][0] < ang_tol:
        groups[0] = [(d - 180, a) for d, a in groups.pop()] + groups[0]     # 0 / 180 wrap
    out = []
    for grp in groups:
        ang = math.radians(sum(d for d, _ in grp) / len(grp))
        c, s = math.cos(ang), math.sin(ang)
        iv = sorted(((-a[0] * s + a[1] * c), *sorted((a[0] * c + a[1] * s, a[2] * c + a[3] * s))) for _, a in grp)
        rows, row = [], []
        for v in iv:                                 # offset clusters (one drawn line each)
            if row and v[0] - row[-1][0] > off_tol:
                rows.append(row); row = []
            row.append(v)
        rows.append(row)
        for row in rows:
            o = sum(v[0] for v in row) / len(row)
            row.sort(key=lambda v: v[1])
            lo, hi = row[0][1], row[0][2]
            for _, t0, t1 in row[1:] + [(0, 1e18, 1e18)]:
                if t0 <= hi + gap or (bridge and t0 < 1e17 and
                                      bridge((hi * c - o * s, hi * s + o * c), (t0 * c - o * s, t0 * s + o * c))):
                    hi = max(hi, t1)
                    continue
                out.append((lo * c - o * s, lo * s + o * c, hi * c - o * s, hi * s + o * c))
                lo, hi = t0, t1
    return out


def merged(page, region, wmin=0.9, wmax=99, gap=12, ang_tol=0.6, off_tol=0.4):
    """Join collinear pieces (dashed / broken lines) into one line each."""
    x0, y0, x1, y1 = region
    segs = [a for a in segments(page)
            if wmin <= a[4] <= wmax and math.hypot(a[2] - a[0], a[3] - a[1]) >= 1
            and x0 < min(a[0], a[2]) and max(a[0], a[2]) < x1 and y0 < min(a[1], a[3]) and max(a[1], a[3]) < y1]
    return merge_lines(segs, gap, ang_tol, off_tol)


def labels(page, region, prefix="X-"):
    """Text lines starting with prefix: (text, cx, cy, direction). Run-together marks are split."""
    x0, y0, x1, y1 = region
    out = []
    for b in page.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            t = "".join(s["text"] for s in ln["spans"]).strip()
            bb = ln["bbox"]
            cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
            if x0 < cx < x1 and y0 < cy < y1 and t.startswith(prefix):
                for part in re.split(rf"\s*(?={re.escape(prefix)})", t):
                    if part.strip():
                        out.append((part.strip(), cx, cy, ln["dir"]))
    return out


def members(page, region, wmin=0.9, wmax=99, minlen=15, maxd=9, prefix="X-"):
    lab = labels(page, region, prefix)
    res = []
    for m in merged(page, region, wmin, wmax):
        dx, dy = m[2] - m[0], m[3] - m[1]
        L = math.hypot(dx, dy)
        if L <= minlen:
            continue
        tx, ty = dx / L, dy / L
        best = None
        for t, cx, cy, d in lab:
            if abs(tx * d[0] + ty * d[1]) < 0.97:
                continue
            s = (cx - m[0]) * tx + (cy - m[1]) * ty
            dist = abs(-(cx - m[0]) * ty + (cy - m[1]) * tx)
            if -2 <= s <= L + 2 and dist < maxd and (best is None or dist < best[0]):
                best = (dist, t)
        res.append({"line": [round(v, 2) for v in m], "len": round(L, 2), "mark": best[1] if best else None})
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("info"); s.add_argument("pdf"); s.add_argument("outdir")
    s = sub.add_parser("words"); s.add_argument("pdf"); s.add_argument("page", type=int); s.add_argument("regex")
    s.add_argument("box", nargs="*", type=float)
    s = sub.add_parser("render"); s.add_argument("pdf"); s.add_argument("page", type=int)
    s.add_argument("box", nargs=4, type=float); s.add_argument("dpi", type=int); s.add_argument("out")
    s = sub.add_parser("members"); s.add_argument("pdf"); s.add_argument("page", type=int)
    s.add_argument("box", nargs=4, type=float); s.add_argument("out")
    for k, v in (("--wmin", 0.9), ("--wmax", 99.0), ("--minlen", 15.0)):
        s.add_argument(k, type=float, default=v)
    s.add_argument("--prefix", default="X-")
    s = sub.add_parser("index"); s.add_argument("inputs", nargs="+"); s.add_argument("--out", required=True)
    s.add_argument("--prev"); s.add_argument("--no-thumbs", action="store_true")
    s = sub.add_parser("find"); s.add_argument("read_dir"); s.add_argument("regex"); s.add_argument("--sheet"); s.add_argument("--view")
    s = sub.add_parser("note"); s.add_argument("read_dir"); s.add_argument("text"); s.add_argument("--sheet", required=True); s.add_argument("--view")
    s = sub.add_parser("status"); s.add_argument("read_dir")
    s = sub.add_parser("connections"); s.add_argument("read_dir")
    s = sub.add_parser("crop"); s.add_argument("read_dir"); s.add_argument("--view"); s.add_argument("--find")
    s.add_argument("--box", nargs=4, type=float); s.add_argument("--sheet"); s.add_argument("--pad", type=float, default=25)
    s.add_argument("--max-px", type=int, default=1000, help="longest side of the crop; 1000 px is about 1k tokens. Raise to 1400 only when the text cannot be read"); s.add_argument("--dpi", type=int); s.add_argument("--part", choices=["tl", "tr", "bl", "br"])
    a = ap.parse_args()
    if a.cmd in ("index", "crop", "find", "note", "status", "connections"):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import sheet_index
        if a.cmd == "index":
            return sheet_index.build_index(a.inputs, a.out, a.prev, not a.no_thumbs)
        if a.cmd == "find":
            return sheet_index.find(a.read_dir, a.regex, a.sheet, a.view)
        if a.cmd == "note":
            return sheet_index.note(a.read_dir, a.sheet, a.text, a.view)
        if a.cmd == "status":
            return sheet_index.status(a.read_dir)
        if a.cmd == "connections":
            return sheet_index.connections(a.read_dir)
        if not (a.view or a.find or a.box):
            ap.error("crop needs --view, --find or --box")
        return sheet_index.crop(a.read_dir, a.view, a.find, a.box, a.sheet, a.pad, a.max_px, a.dpi, a.part)
    doc = fitz.open(a.pdf)
    if a.cmd == "info":
        out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
        for i, pg in enumerate(doc):
            (out / f"p{i + 1}.txt").write_text(pg.get_text(), encoding="utf-8")
            pg.get_pixmap(dpi=40).save(str(out / f"p{i + 1}_overview.png"))
            print(f"page {i + 1}: {pg.rect.width:.0f} x {pg.rect.height:.0f} pt, {len(pg.get_drawings())} drawings, "
                  f"{len(pg.get_images())} images{' (scanned: read the image)' if pg.get_images() and not pg.get_drawings() else ''}")
    elif a.cmd == "words":
        box = a.box or [-1e9, -1e9, 1e9, 1e9]
        for w in doc[a.page - 1].get_text("words"):
            cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
            if re.search(a.regex, w[4]) and box[0] < cx < box[2] and box[1] < cy < box[3]:
                print(f"{cx:8.1f} {cy:8.1f}  {w[4]}")
    elif a.cmd == "render":
        doc[a.page - 1].get_pixmap(dpi=a.dpi, clip=fitz.Rect(*a.box)).save(a.out)
    else:
        res = members(doc[a.page - 1], a.box, a.wmin, a.wmax, a.minlen, prefix=a.prefix)
        Path(a.out).write_text(json.dumps(res, indent=0), encoding="utf-8")
        marks = {}
        for r in res:
            marks[r["mark"]] = marks.get(r["mark"], 0) + 1
        print(f"{len(res)} lines -> {a.out}; per mark: " + ", ".join(f"{k} {v}" for k, v in sorted(marks.items(), key=lambda kv: str(kv[0]))))


if __name__ == "__main__":
    main()
