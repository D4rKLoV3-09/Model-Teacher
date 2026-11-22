#!/usr/bin/env python3
"""Make non-vector drawing pages true vector PDF pages before they are sorted or read (step 0a of split-drawings).

    python generate-model/vectorize_drawings.py PDF --pages 51-79 --out OUT.pdf      (stand-alone)
    import vectorize_drawings as V; V.vectorize_pages(src_pdf, page_indices, out_pdf)  (from split_drawings.py)

Per page (only the pages asked for; everything else is copied unchanged, so unrelated sheets stay as they are):
  vector   real text + vector lines            -> unchanged
  outlined vector lines, text drawn as glyph outlines (no real text)
           -> full-page OCR (rapidocr-onnxruntime, horizontal + vertical text, 2 x 2 tiles at 150 dpi) written as an
              invisible real-text layer over the glyphs: the lines stay the original vectors, the text becomes
              searchable / extractable (read_drawing.py, split_drawings.py, ga_plan see marks, sizes, levels)
  raster   a scan / image page                 -> the image traced to vector outlines (potrace, potracer package)
              + the same OCR text layer; the image is removed
Speed (a test job, 20 outlined A1 sheets, 2026-11-18): each line is read once (upright pass: lines wider than tall; rotated
pass: only lines that were vertical on the sheet), the text layer is written with one TextWriter per direction (was
insert_text per word: 30 s a dense sheet, serial). Pages run in parallel worker processes; results are cached by image.
A laptop on a 'Silent' / battery power plan runs this about 2-3x slower. Prints one line per page class.
Libraries searched first: ocrmypdf does the text layer but needs Tesseract + Ghostscript installed on Windows;
rapidocr + PyMuPDF and potracer are pip-only.
"""
import argparse
import math
import os

import pymupdf as fitz

_ocr = None


def classify(page):
    text = page.get_text()
    raw = page.read_contents()
    paths = raw.count(b" l\n") + raw.count(b" c\n") + raw.count(b" re\n")
    area = abs(page.rect)
    img_area = 0.0
    for info in page.get_image_info():
        r = fitz.Rect(info["bbox"]) & page.rect
        img_area += abs(r)
    if img_area > 0.5 * area and paths < 3000:
        return "raster"
    if len(text) < 800 and paths > 5000:
        return "outlined"
    return "vector"


def _ocr_engine():
    global _ocr
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr = RapidOCR()
    return _ocr


def ocr_page(page, dpi=150, min_conf=0.6):
    """[(text, rect, vertical)] in page coordinates (rotation removed by the caller)."""
    import numpy as np
    pm = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.h, pm.w, pm.n)[:, :, :3]
    return ocr_image(img, 72.0 / dpi, min_conf)


# ---- parallel OCR: one page per worker process, a few onnxruntime threads each (one engine on all cores scales badly)
_THREADS = None


def _worker_init(threads):
    global _THREADS
    _THREADS = threads
    import os
    import onnxruntime
    import rapidocr_onnxruntime.utils as U
    base = onnxruntime.SessionOptions

    def opts():
        o = base()
        o.intra_op_num_threads, o.inter_op_num_threads = threads, 1
        if os.environ.get("SPLIT_OCR_SPIN", "0") == "0":     # idle threads sleep instead of spinning
            o.add_session_config_entry("session.intra_op.allow_spinning", "0")
        return o
    U.SessionOptions = opts


def _ocr_job(job):
    """(index, samples, w, h, n, k, min_conf) -> (index, [(text, (x0, y0, x1, y1), vertical)])"""
    import numpy as np
    i, samples, w, h, n, k, min_conf = job
    img = np.frombuffer(samples, dtype=np.uint8).reshape(h, w, n)[:, :, :3]
    return i, [(t, tuple(r), v) for t, r, v in ocr_image(img, k, min_conf)]


def ocr_raw_job(job):
    """(key, samples, w, h, n) -> (key, [(text, x_px, y_px)]): one plain OCR pass (title blocks in split_drawings)."""
    import numpy as np
    key, samples, w, h, n = job
    img = np.frombuffer(samples, dtype=np.uint8).reshape(h, w, n)[:, :, :3]
    res, _ = _ocr_engine()(np.ascontiguousarray(img))
    return key, [(r[1], r[0][0][0], r[0][0][1]) for r in (res or [])]


def _cache_dir():
    import os
    from pathlib import Path
    d = Path(os.environ.get("SPLIT_OCR_CACHE") or Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "steel-skills" / "ocr-cache")
    d.mkdir(parents=True, exist_ok=True)
    return d


OCR_VERSION = 2   # bump when the OCR reading changes: the cache is keyed on it, so results from older code are read again
                  # (v1 cache from before the 18 Nov fix gave huge rotated duplicates; v2 = one read per line)


def _cache_file(fn, j, cache=None):
    import hashlib
    key = f"v{OCR_VERSION}".encode() + fn.__name__.encode() + j[1] + repr(j[2:]).encode()
    return (cache or _cache_dir()) / f"{hashlib.sha1(key).hexdigest()}.json"


def cached_count(fn, jobs):
    """How many of jobs an earlier run (also one that crashed later) already OCR'd: they are not read again."""
    cache = _cache_dir()
    return sum(1 for j in jobs if _cache_file(fn, j, cache).exists())


def run_parallel(fn, jobs):
    """Yield fn(job) = (key, result) for every job: from the OCR cache when this exact image was read before
    (job = (key, image bytes, ...); cache keyed on the image + settings, so a re-run of a package skips OCR),
    else in several worker processes at once (as they complete). Each result is saved the moment it is read,
    so a run that stops half way (crash, closed window, power cut) resumes where it stopped."""
    import json
    cache, todo = _cache_dir(), []
    for j in jobs:
        f = _cache_file(fn, j, cache)
        try:
            yield j[0], json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            todo.append((f, j))
    if not todo:
        return
    keyfile = {}
    for f, j in todo:
        keyfile.setdefault(j[0], f)
    for key, res in _run_pool(fn, [j for _, j in todo]):
        try:          # written to .tmp then renamed: a run killed mid-write never leaves a half file behind
            tmp = keyfile[key].with_suffix(".tmp")
            tmp.write_text(json.dumps(res), encoding="utf-8")
            os.replace(tmp, keyfile[key])
        except (OSError, KeyError, TypeError):
            pass
        yield key, res


def _run_pool(fn, jobs):
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    cpu = os.cpu_count() or 4
    # measured on the curator's i7-13650HX (20 threads), 6 outlined pages: 1 worker 398 s, 3 workers x 6 threads 189 s
    workers = max(1, min(len(jobs), int(os.environ.get("SPLIT_OCR_WORKERS", 0)) or max(1, cpu // 6), 4))
    threads = int(os.environ.get("SPLIT_OCR_THREADS", 0)) or max(2, min(6, cpu // workers))
    if workers == 1:
        yield from (fn(j) for j in jobs)
        return
    with ProcessPoolExecutor(workers, initializer=_worker_init, initargs=(threads,)) as pool:
        for f in as_completed([pool.submit(fn, j) for j in jobs]):
            yield f.result()


def _box_wh(b):
    import numpy as np
    return (max(np.linalg.norm(b[0] - b[1]), np.linalg.norm(b[2] - b[3])),
            max(np.linalg.norm(b[0] - b[3]), np.linalg.norm(b[1] - b[2])))


def _read_lines(ocr, im, vertical):
    """rapidocr's own steps (detect -> crop -> angle -> recognise), but only the boxes that read horizontally in
    this image: the upright pass reads the lines that are wider than tall, the rotated pass only the lines that were
    vertical on the sheet. Without this filter every horizontal line was read twice (once more, sideways, in the
    rotated pass) and written back as a huge rotated duplicate; it was also half the recognition time."""
    boxes, _ = ocr.text_detector(im)
    if boxes is None or len(boxes) == 0:
        return []
    keep = []
    for b in ocr.sorted_boxes(boxes):
        w, h = _box_wh(b)
        if (w >= 1.5 * h) if vertical else (h < 1.5 * w):
            keep.append(b)
    if not keep:
        return []
    crops = ocr.get_crop_img_list(im, keep)
    if ocr.use_angle_cls:
        crops, _, _ = ocr.text_cls(crops)
    rec, _ = ocr.text_recognizer(crops)
    return [(b, t, c) for b, (t, c) in zip(keep, rec)]


def ocr_image(img, k, min_conf=0.6):
    """[(text, rect, vertical)] of an RGB image; k = page points per pixel."""
    import numpy as np
    ocr = _ocr_engine()
    H, W = img.shape[:2]
    th, tw = min(H, H // 2 + 100), min(W, W // 2 + 100)
    found = {}
    for y0 in sorted({0, H - th}):
        for x0 in sorted({0, W - tw}):
            tile = np.ascontiguousarray(img[y0:y0 + th, x0:x0 + tw])
            for vertical in (False, True):
                im = np.ascontiguousarray(np.rot90(tile)) if vertical else tile
                for box, txt, conf in _read_lines(ocr, im, vertical):
                    if float(conf) < min_conf or not txt.strip():
                        continue
                    pts = [(float(x), float(y)) for x, y in box]
                    if vertical:       # rotated image (xr, yr) -> tile (tw - 1 - yr, xr)
                        pts = [(tw - 1 - yr, xr) for xr, yr in pts]
                    xs, ys = [p[0] + x0 for p in pts], [p[1] + y0 for p in pts]
                    r = fitz.Rect(min(xs) * k, min(ys) * k, max(xs) * k, max(ys) * k)
                    key = (txt.strip(), round(r.x0 / 6), round(r.y0 / 6))
                    found.setdefault(key, (txt.strip(), r, vertical))
    return list(found.values())


def add_text_layer(page, items):
    """Invisible real text (render mode 3) laid over each OCR box, one content stream per direction
    (TextWriter: 0.3 s a sheet; insert_text per item rewrote the page each call, 30 s for a dense sheet)."""
    page.wrap_contents()               # the page's own transforms must not move the new text
    font = fitz.Font("helv")
    hor = fitz.TextWriter(page.rect)
    ver = fitz.TextWriter(page.rect)
    rot = fitz.Matrix(90)              # vertical text: write_text(morph) turns the whole writer by ~rot about (0, 0),
                                       # so each point is appended at point * rot (same result as insert_text rotate=90)
    for txt, r, vertical in items:
        thick, run = (r.width, r.height) if vertical else (r.height, r.width)
        unit = font.text_length(txt, fontsize=1) or 1.0
        fs = max(1.0, min(thick * 0.85, run / unit))   # the text fills the OCR box and never runs past it,
                                                       # so a highlight covers the word on the sheet, not more
        if vertical:
            ver.append(fitz.Point(r.x1 - r.width * 0.2, r.y1) * rot, txt, font=font, fontsize=fs)
        else:
            hor.append(fitz.Point(r.x0, r.y1 - r.height * 0.2), txt, font=font, fontsize=fs)
    hor.write_text(page, render_mode=3)
    ver.write_text(page, render_mode=3, morph=(fitz.Point(0, 0), rot))




def trace_raster(page, dpi=150):
    """Replace the page image by its potrace outlines (black fill, even-odd)."""
    import numpy as np
    import potrace
    pm = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    img = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.h, pm.w)
    bm = potrace.Bitmap(img < 128)
    curves = bm.trace(turdsize=3, alphamax=0.8, opticurve=True)
    k = 72.0 / dpi
    for info in page.get_image_info():
        page.add_redact_annot(fitz.Rect(info["bbox"]))
    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_REMOVE, graphics=fitz.PDF_REDACT_LINE_ART_NONE,
                          text=fitz.PDF_REDACT_TEXT_NONE)
    sh = page.new_shape()
    P = lambda p: fitz.Point(p[0] * k, p[1] * k)
    n = 0
    for curve in curves:
        start = (curve.start_point.x, curve.start_point.y)
        cur = start
        for seg in curve.segments:
            if seg.is_corner:
                sh.draw_line(P(cur), P((seg.c.x, seg.c.y)))
                sh.draw_line(P((seg.c.x, seg.c.y)), P((seg.end_point.x, seg.end_point.y)))
            else:
                sh.draw_bezier(P(cur), P((seg.c1.x, seg.c1.y)), P((seg.c2.x, seg.c2.y)), P((seg.end_point.x, seg.end_point.y)))
            cur = (seg.end_point.x, seg.end_point.y)
        n += 1
    sh.finish(color=None, fill=(0, 0, 0), even_odd=True, closePath=True)
    sh.commit()
    return n


def vectorize_pages(src_pdf, indices, out_pdf, log=print, progress=None, force=()):
    """Copy src_pdf to out_pdf with the listed pages made true vector. Returns {index: class}.
    progress(index, kind) is called after each page (the Split Drawings window shows it).
    force: pages to treat as outlined although the title block is real text (the drawing itself is outlines)."""
    doc = fitz.open(src_pdf)
    kinds, jobs, dpi = {}, [], 150
    for i in indices:
        page = doc[i]
        kind = classify(page)
        if kind == "vector" and i in force:
            kind = "outlined"
        kinds[i] = kind
        if kind == "vector":
            if progress:
                progress(i, kind)
            continue
        page.remove_rotation()
        if kind == "raster":
            trace_raster(page)
        pm = page.get_pixmap(dpi=dpi)
        jobs.append((i, pm.samples, pm.w, pm.h, pm.n, 72.0 / dpi, 0.6))
    for i, items in run_parallel(_ocr_job, jobs):
        add_text_layer(doc[i], [(t, fitz.Rect(r), v) for t, r, v in items])
        if progress:
            progress(i, kinds[i])
    if jobs:       # every page already vector: no copy is written, the caller keeps the source (a damaged source is not rewritten)
        doc.save(out_pdf, garbage=3, deflate=True)
    c = {k: sum(1 for v in kinds.values() if v == k) for k in ("vector", "outlined", "raster")}
    log(f"vectorize: {c['vector']} vector (unchanged), {c['outlined']} outlined text -> OCR text layer, "
        f"{c['raster']} raster -> traced + text layer")
    return kinds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--pages", help="e.g. 51-79,3 (1-based); default all")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    n = fitz.open(a.pdf).page_count
    idx = []
    for part in (a.pages or f"1-{n}").split(","):
        lo, _, hi = part.partition("-")
        idx += list(range(int(lo) - 1, int(hi or lo)))
    vectorize_pages(a.pdf, idx, a.out)


if __name__ == "__main__":
    main()
