#!/usr/bin/env python3
"""Drawing store: every PDF is opened ONCE; everything later reads this store (the curator's rule "done once, never repeated").

    python generate-model/store.py extract PDF [PDF ...] --out jobs/<job>/Models/_work/store
    python generate-model/store.py check   STORE                 # stored items == items on the page, per sheet

What is kept (lossless, nothing filtered at extraction; filters belong to the readers and are counted there):
  pNN.npz  segs  float32 [n, 10]: x0 y0 x1 y1 width r g b dashed path   (every straight piece; rect -> 4 pieces;
                                   quad -> 4; a bezier -> its chord, flag in `curve`)
           curve uint8 [n]  1 = chord of a bezier piece (curves also kept whole in `bez`)
           bez   float32 [m, 11]: 4 control points (8) + width + dashed + path
           fill  float32 [k, 8]: filled path bbox x0 y0 x1 y1, grey, n_points, path, quad-ness
           quad  float32 [q, 5]: centre lines of thin filled quads (members drawn as filled shapes): x0 y0 x1 y1 half_width
           vis   uint8 [n]: % of the piece the PDF shows (5 samples along it). A piece outside the clip of its path (a
                 viewport) or under a later fill of another colour (white masks: diamonds inside grid bubbles) is hidden.
                 Sheet.segs keeps the pieces >= VIS_MIN % shown; Sheet.segs_all keeps all, Sheet.n_hidden counts the rest
           lay   uint16 [n]: CAD layer of the piece (index into `layers`; "" = none; e.g. STEEL / DIM / Civil_Grid)
           dpat  uint16 [n]: dash pattern of the piece (index into `dashes`; "" = solid; centre vs hidden lines)
  pNN.json words: [[x0, y0, x1, y1, text, dir]] (dir 0 = horizontal, 90 / 270 = vertical text), size, rotation, hash,
           layers, dashes, hidden (pieces under VIS_MIN)
  Fills stay boxes (bbox + grey): no reader needs their outline; visibility uses the outlines at extraction.
  store.json  pdf sha1 -> pages; a PDF whose bytes did not change is not opened again.
Coordinates: PDF points, page rotation removed (as displayed), y down.
"""
import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

VERSION = 3          # 3: vis / lay / dpat per piece (hidden lines were read as drawn); 2: quad centre lines on the principal axis
VIS_MIN = 50         # % of a piece that must show for a reader to see it


def _sha(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _grey(c):
    return (1.0, 1.0, 1.0) if c is None else tuple(float(v) for v in (list(c) + [0, 0, 0])[:3])


def _quad_centre(pts):
    """Thin filled shape -> its centre line (x0, y0, x1, y1, half_width) or None."""
    if not 4 <= len(pts) <= 60:
        return None
    P = np.array(pts, dtype=float)
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)          # principal axis, not the diagonal of the quad
    t = vt[0]
    n = np.array([-t[1], t[0]])
    al = (P - c) @ t
    off = (P - c) @ n
    L = al.max() - al.min()
    w1, o = (off.max() - off.min()) / 2, (off.max() + off.min()) / 2
    if L < 1e-6 or w1 > 3.0 or L < 5 * max(w1, 0.05):
        return None
    a = c + al.min() * t + o * n
    b = c + al.max() * t + o * n
    return (a[0], a[1], b[0], b[1], w1)


def _polys(d):
    """A filled path -> its closed sub-polygons [np.array (k, 2)]."""
    out, cur = [], []
    for it in d["items"]:
        k = it[0]
        if k == "re":
            r = it[1]
            out.append(np.array([(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)]))
        elif k == "qu":
            q = it[1]
            out.append(np.array([(p.x, p.y) for p in (q.ul, q.ur, q.lr, q.ll)]))
        elif k in ("l", "c"):
            p, q = it[1], it[-1]
            if cur and math.dist(cur[-1], (p.x, p.y)) > 0.01:
                if len(cur) >= 3:
                    out.append(np.array(cur))
                cur = []
            if not cur:
                cur.append((p.x, p.y))
            cur.append((q.x, q.y))
    if len(cur) >= 3:
        out.append(np.array(cur))
    return out


def visibility(page, seg_seq, seg_col, segs):
    """% of each stroked piece the page shows (5 samples along it). A sample is hidden when it lies outside the clip in
    force for its path (clip rectangles: viewports) or inside a fill painted later in another colour (> 0.3 apart: a white
    mask over a black line hides it; a black fill over a black line does not)."""
    from matplotlib.path import Path as MPath
    n = len(segs)
    if not n:
        return np.zeros(0, np.uint8)
    S = np.asarray(segs, float)[:, :4]
    t = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    px = S[:, 0, None] + (S[:, 2] - S[:, 0])[:, None] * t
    py = S[:, 1, None] + (S[:, 3] - S[:, 1])[:, None] * t
    hid = np.zeros((n, 5), bool)
    seq = np.asarray(seg_seq)
    col = np.asarray(seg_col, float)
    clip_of, stack, fills = {}, {}, []
    for e in page.get_drawings(extended=True):
        lv = e.get("level", 0) or 0
        if e["type"] == "clip":
            for k in [k for k in stack if k >= lv]:
                del stack[k]
            stack[lv] = e.get("scissor")
            continue
        if e["type"] not in ("s", "f", "fs"):
            continue
        rs = [r for k, r in stack.items() if k < lv and r is not None]
        if rs:
            clip_of[e["seqno"]] = (max(r.x0 for r in rs), max(r.y0 for r in rs), min(r.x1 for r in rs), min(r.y1 for r in rs))
        if e["type"] in ("f", "fs") and e.get("fill") is not None and (e.get("fill_opacity") is None or e["fill_opacity"] > 0.5):
            P = _polys(e)
            if P:
                fills.append((e["seqno"], _grey(e["fill"]), P, e["rect"], e.get("even_odd")))
    rects = sorted(set(clip_of.values()))             # outside the clip of its own path (few distinct clips per sheet)
    rid = {r: i for i, r in enumerate(rects)}
    seg_clip = np.array([rid.get(clip_of.get(int(s)), -1) for s in seq])
    for i, c in enumerate(rects):
        m = seg_clip == i
        hid[m] |= (px[m] < c[0] - 0.5) | (px[m] > c[2] + 0.5) | (py[m] < c[1] - 0.5) | (py[m] > c[3] + 0.5)
    if fills:                                         # under a later fill of another colour (samples bucketed in 50 pt cells)
        C = 50.0
        fx, fy = px.ravel(), py.ravel()
        cell = np.floor(fx / C).astype(np.int64) * 100000 + np.floor(fy / C).astype(np.int64)
        order = np.argsort(cell, kind="stable")
        cs = cell[order]
        for sq, fc, P, r, eo in fills:
            idx = []
            for a in range(int(math.floor(r.x0 / C)), int(math.floor(r.x1 / C)) + 1):
                for b in range(int(math.floor(r.y0 / C)), int(math.floor(r.y1 / C)) + 1):
                    key = a * 100000 + b
                    i0, i1 = np.searchsorted(cs, key), np.searchsorted(cs, key, side="right")
                    if i1 > i0:
                        idx.append(order[i0:i1])
            if not idx:
                continue
            idx = np.concatenate(idx)
            si = idx // 5
            idx = idx[(seq[si] < sq) & (np.abs(col[si] - np.array(fc)).max(1) > 0.3)
                      & (fx[idx] >= r.x0) & (fx[idx] <= r.x1) & (fy[idx] >= r.y0) & (fy[idx] <= r.y1)]
            if not len(idx):
                continue
            pts = np.c_[fx[idx], fy[idx]]
            inside = np.zeros(len(idx), int)
            for poly in P:
                inside += MPath(poly).contains_points(pts, radius=-0.05)
            cov = (inside % 2 == 1) if eo else inside > 0
            hid.ravel()[idx[cov]] = True
    return np.round(100 * (1 - hid.mean(1))).astype(np.uint8)


def extract_page(page):
    page.remove_rotation()
    segs, curve, bez, fill, quad = [], [], [], [], []
    seg_seq, seg_col, seg_lay, seg_dash, layers, dashes = [], [], [], [], [""], [""]
    n_items = 0
    for pid, d in enumerate(page.get_drawings()):
        col = _grey(d.get("color"))
        w = float(d.get("width") or 0)
        dpat = (d.get("dashes") or "[] 0").strip()
        dashed = 1.0 if dpat not in ("[] 0", "[]0", "") else 0.0
        stroke = d.get("type") in ("s", "fs")
        lay = d.get("layer") or ""
        if lay not in layers:
            layers.append(lay)
        dp = dpat if dashed else ""
        if dp not in dashes:
            dashes.append(dp)
        before = len(segs)
        pts = []
        for it in d["items"]:
            n_items += 1
            k = it[0]
            if k == "l":
                p, q = it[1], it[2]
                seg = [(p.x, p.y, q.x, q.y)]
                pts += [(p.x, p.y), (q.x, q.y)]
            elif k == "re":
                r = it[1]
                seg = [(r.x0, r.y0, r.x1, r.y0), (r.x1, r.y0, r.x1, r.y1), (r.x1, r.y1, r.x0, r.y1), (r.x0, r.y1, r.x0, r.y0)]
                pts += [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)]
            elif k == "qu":
                q = it[1]
                c = [q.ul, q.ur, q.lr, q.ll]
                seg = [(c[i].x, c[i].y, c[(i + 1) % 4].x, c[(i + 1) % 4].y) for i in range(4)]
                pts += [(p.x, p.y) for p in c]
            elif k == "c":
                p0, p1, p2, p3 = it[1:5]
                bez.append((p0.x, p0.y, p1.x, p1.y, p2.x, p2.y, p3.x, p3.y, w, dashed, pid))
                seg = [(p0.x, p0.y, p3.x, p3.y)]
                pts += [(p0.x, p0.y), (p3.x, p3.y)]
                if stroke:
                    segs.append((*seg[0], w, *col, dashed, pid)); curve.append(1)
                continue
            else:
                continue
            if stroke:
                for s in seg:
                    segs.append((*s, w, *col, dashed, pid)); curve.append(0)
        k_new = len(segs) - before
        seg_seq += [d["seqno"]] * k_new
        seg_col += [col] * k_new
        seg_lay += [layers.index(lay)] * k_new
        seg_dash += [dashes.index(dp)] * k_new
        f = d.get("fill")
        if f is not None and d.get("type") in ("f", "fs"):
            fc = _grey(f)
            uniq = []
            for p in pts:
                if not any(abs(p[0] - u[0]) < 0.05 and abs(p[1] - u[1]) < 0.05 for u in uniq[-64:]):
                    uniq.append(p)
            xs, ys = [p[0] for p in pts] or [0], [p[1] for p in pts] or [0]
            qc = _quad_centre(uniq) if max(fc) < 0.15 else None
            fill.append((min(xs), min(ys), max(xs), max(ys), sum(fc) / 3, len(uniq), pid, 1.0 if qc else 0.0))
            if qc:
                quad.append(qc)
    words = []
    td = page.get_text("rawdict" if False else "dict")
    for b in td["blocks"]:
        for ln in b.get("lines", []):
            dx, dy = ln["dir"]
            ang = round(math.degrees(math.atan2(-dy, dx))) % 360
            for sp in ln["spans"]:
                t = sp["text"].strip()
                if t:
                    x0, y0, x1, y1 = sp["bbox"]
                    words.append([round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2), t, ang, round(sp["size"], 2)])
    arr = lambda a, n: np.array(a, dtype=np.float32).reshape(-1, n)
    vis = visibility(page, seg_seq, seg_col, segs)
    data = {"segs": arr(segs, 10), "curve": np.array(curve, dtype=np.uint8), "bez": arr(bez, 11), "fill": arr(fill, 8),
            "quad": arr(quad, 5), "vis": vis, "lay": np.array(seg_lay, dtype=np.uint16), "dpat": np.array(seg_dash, dtype=np.uint16)}
    meta = {"size": [page.rect.width, page.rect.height], "words": words, "n_items": n_items, "layers": layers, "dashes": dashes,
            "hidden": int((vis < VIS_MIN).sum()),
            "n_paths": int(max([s[9] for s in segs] + [f[6] for f in fill] + [-1])) + 1}
    return data, meta


def extract(pdfs, out):
    import pymupdf
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    idx_p = out / "store.json"
    idx = json.loads(idx_p.read_text(encoding="utf-8")) if idx_p.exists() else {"version": VERSION, "pdfs": {}}
    done = skipped = 0
    for pdf in pdfs:
        sha = _sha(pdf)
        key = Path(pdf).stem
        old = idx["pdfs"].get(key)
        if old and old["sha1"] == sha and idx.get("version") == VERSION and all((out / f"{key}_p{n:02d}.npz").exists() for n in range(1, old["pages"] + 1)):
            skipped += old["pages"]
            continue
        doc = pymupdf.open(pdf)
        for i, page in enumerate(doc):
            data, meta = extract_page(page)
            n = i + 1
            np.savez_compressed(out / f"{key}_p{n:02d}.npz", **data)
            meta["hash"] = hashlib.sha1(data["segs"].tobytes() + json.dumps(meta["words"]).encode()).hexdigest()[:12]
            (out / f"{key}_p{n:02d}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            done += 1
        idx["pdfs"][key] = {"path": str(pdf), "sha1": sha, "pages": len(doc)}
        doc.close()
    idx["version"] = VERSION
    idx_p.write_text(json.dumps(idx, indent=1))
    print(f"store: {done} sheet(s) extracted, {skipped} unchanged (not opened) -> {out}")


LAYER_PURLIN = re.compile(r"purlin|girt|sag.?rod|cladding.?rail|side.?rail", re.I)
LAYER_ANNOT = re.compile(r"dim|text|anot|annot|grid|gird|axes|title|border|hatch|hacth|(^|[-_ $])cen(ter|tre)?\b|center|centre|"
                         r"leader|symbol|note|viewport|defpoint|^tag|cloud|中心", re.I)
LAYER_TRADE = re.compile(r"hand.?rail|railing|grating|bolt|nut|reinf|rebar|found|fdn|pile|wall|door|window|equip|roof.?sheet|"
                         r"roof.?fl|clad|deck|duct|drain|waterstop|pipe|cable|elect|furnit|floor|stair.?tread|^a-|^m-|^e-|^p-|"
                         r"^c-|concrete|road|fan\b|基礎", re.I)


def layer_class(name):
    """CAD layer name -> 'purlin' | 'annot' (dimensions, text, grids, title) | 'trade' (handrail, grating, bolts, RC, walls,
    equipment, roof sheet, other disciplines) | 'steel' (steel layers and any name that says nothing: kept, Cautious)."""
    n = (name or "").strip()
    if not n:
        return "steel"
    if LAYER_PURLIN.search(n):
        return "purlin"
    if LAYER_ANNOT.search(n):
        return "annot"
    if LAYER_TRADE.search(n):
        return "trade"
    return "steel"


class Sheet:
    """One stored sheet: .segs (the pieces the PDF shows, >= VIS_MIN %) .quad .fill .bez (numpy), .words, .size. No PDF is
    opened. .segs_all / .vis / .lay / .dpat: every stored piece with its visible %, layer and dash pattern (names in
    .layers / .dashes); .n_hidden = pieces under VIS_MIN (counted, never read as drawn)."""

    def __init__(self, store, key, n):
        p = Path(store) / f"{key}_p{n:02d}"
        z = np.load(f"{p}.npz")
        self.segs_all, self.curve_all, self.bez, self.fill, self.quad = (z[k] for k in ("segs", "curve", "bez", "fill", "quad"))
        n_ = len(self.segs_all)
        self.vis = z["vis"] if "vis" in z.files else np.full(n_, 100, np.uint8)
        self.lay = z["lay"] if "lay" in z.files else np.zeros(n_, np.uint16)
        self.dpat = z["dpat"] if "dpat" in z.files else np.zeros(n_, np.uint16)
        shown = self.vis >= VIS_MIN
        self.segs, self.curve = self.segs_all[shown], self.curve_all[shown]
        self.n_hidden = int((~shown).sum())
        m = json.loads(Path(f"{p}.json").read_text(encoding="utf-8"))
        self.words, self.size, self.hash, self.page = m["words"], m["size"], m["hash"], n
        self.layers, self.dashes = m.get("layers", [""]), m.get("dashes", [""])
        cls = np.array([layer_class(n) for n in self.layers] or ["steel"])
        self.lay_cls = cls[np.minimum(self.lay, len(cls) - 1)] if n_ else np.zeros(0, "<U8")

    def pass_mask(self, kind="structure"):
        """LAYER-R01 (the curator 2026-11-22): read one kind of steel with the other lines hidden. 'structure' = shown pieces on
        steel / unnamed layers (purlin, annotation and other trades' layers hidden); 'purlin' = shown pieces on purlin / girt
        layers only. A sheet without layer names hides nothing (every piece is 'steel')."""
        shown = self.vis >= VIS_MIN
        if kind == "purlin":
            return shown & (self.lay_cls == "purlin")
        return shown & (self.lay_cls == "steel")

    def pass_segs(self, kind="structure"):
        return self.segs_all[self.pass_mask(kind)]


def sheets(store, key=None):
    idx = json.loads((Path(store) / "store.json").read_text(encoding="utf-8"))
    key = key or next(iter(idx["pdfs"]))
    return [Sheet(store, key, n) for n in range(1, idx["pdfs"][key]["pages"] + 1)]


def check(store):
    """Stored pieces against the page itself (opens the PDF: a test of the store, never part of a job)."""
    import pymupdf
    idx = json.loads((Path(store) / "store.json").read_text(encoding="utf-8"))
    bad = 0
    for key, e in idx["pdfs"].items():
        doc = pymupdf.open(e["path"])
        for sh in sheets(store, key):
            pg = doc[sh.page - 1]
            pg.remove_rotation()
            n_l = n_w = 0
            for d in pg.get_drawings():
                if d.get("type") in ("s", "fs"):
                    for it in d["items"]:
                        n_l += {"l": 1, "re": 4, "qu": 4, "c": 1}.get(it[0], 0)
            n_w = sum(1 for b in pg.get_text("dict")["blocks"] for l in b.get("lines", []) for s in l["spans"] if s["text"].strip())
            ok = n_l == len(sh.segs_all) and n_w == len(sh.words)
            bad += not ok
            if not ok:
                print(f"  p{sh.page}: lines {len(sh.segs_all)}/{n_l} words {len(sh.words)}/{n_w}")
        doc.close()
    print("store check: " + ("OK, every line and word stored" if not bad else f"{bad} sheet(s) differ"))
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "check"])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.paths, a.out)
    else:
        sys.exit(1 if check(a.paths[0]) else 0)
