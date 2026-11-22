#!/usr/bin/env python3
"""Sheet index for vector PDF drawings (PyMuPDF): read a sheet as text first, look only at crops.
Used through read_drawing.py (same code on the curator's PC and in a claude.ai cloud session):

    python read_drawing.py index PDF_OR_FOLDER [...] --out jobs/<job>/out/read [--prev OLD/sheets.json]
    python read_drawing.py crop  jobs/<job>/out/read --view 2A | --find F-2FAPT2-SC-01 | --box x0 y0 x1 y1
                                 [--sheet NAME] [--pad 25] [--max-px 1000] [--dpi N] [--part tl|tr|bl|br]

index writes into --out:
  index.md       read this first: one row per sheet (dwg no, rev, status, title, views, member pens, notes, changed)
                 and one line of views per sheet (about 60 tokens a sheet).
  index/<sheet>.md  one sheet in full (about 1 k tokens per A1 sheet): title block, then every view with its scale,
                 grids, member marks, section sizes, levels, dimensions and other text; legends and notes in reading order.
                 Open only the sheets the job needs.
  reading.json   seeded / refreshed: `drawings` (title block, scales, grids, levels, member pens per sheet) and
                 `notes`; the agent's own keys are never touched.
  sheets.json    the same with coordinates (points, as displayed) for crop, find and later scripts. Do not read it.
  thumbs/<sheet>.png   one low-resolution overview per sheet (look once, for the arrangement).
  _pages/        cache: a PDF whose bytes have not changed is not parsed again.
  The previous sheets.json (or --prev) marks each sheet new / changed / same (text AND linework), so only changed
  sheets are read again.
find / note / status / crop keep the reading memory: `find` answers "where is X" from sheets.json without a PDF or an
image; `crop` remembers every crop (ledger.json) and prints the notes written about that view; `note` keeps what a
crop showed; `status` lists what has been read and what nobody opened.
crop renders one view (id from index.md), the surroundings of a text, or a box, capped at --max-px, and prints
mm per pixel for a scaled view (a dimension measured that way is "scaled off the drawing").

How views are found: linework and text are clustered by white space; a cluster is named by the view title inside
it (the line above a scale such as "1 : 50"); a cluster with several titles is cut between them. A sheet with
fewer than MIN_WORDS words is flagged "scan_or_outlined_text": thumbnail and crops only.
Needs only PyMuPDF and numpy.
"""
import hashlib
import json
import math
import pathlib
import re
import sys

import numpy as np
import pymupdf

CELL = 4.0            # pt per mask cell
GAP = 13.0            # pt: linework closer than 2 x GAP belongs to the same view
MIN_WORDS = 15
THUMB_PX = 1300       # long side of the overview image

SCALE_RE = re.compile(r"^(?:SCALE\s*)?1\s*:\s*(\d+(?:\.\d+)?)$|^N\.?T\.?S\.?$", re.I)
SECTION_RE = re.compile(
    r"\b(?:UB|UC|UKB|UKC|PFC|TFC|UPN|IPE|HE[AB]?|SHS|RHS|CHS|EA|UA|FB|PL|PLT|L|C|Z)\s?\d+(?:\.\d+)?"
    r"(?:\s?[xX×]\s?\d+(?:\.\d+)?){0,3}(?:\s?(?:kg/m|THK|mm))?", re.I)
LEVEL_RE = re.compile(r"^[+-]?\d{1,3}\.\d{2,3}$")
LEVEL_WORDS = ("TOC", "TOS", "TOG", "FFL", "SFL", "EL", "RL", "BOS", "SSL", "LEVEL", "LVL", "T.O.S", "T.O.C")
DIM_RE = re.compile(r"^\d{2,6}$|^\d+\s*(?:x|X|@|c/c|TYP).*$")
MARK_RE = re.compile(r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9]+(?:[-_/.][A-Za-z0-9]+)+$")
SHORT_MARK_RE = re.compile(r"^[A-Z]{1,4}\d{1,4}[A-Z]{0,3}\d?$")     # B45J, C35W, GB12 (member pens only)
GRID_RE = re.compile(r"^[A-Z]{1,2}\d{0,3}'?$|^\d{1,3}[A-Z]?'?$")
TB_LABELS = ("DRAWING NO", "DWG NO", "DRAWING TITLE", "PROJECT TITLE", "REV", "SCALE", "DATE", "DRAWN", "CHECKED",
             "JOB NO", "SHEET", "PROJECT PHASE", "STATUS")


# ---------------- text ----------------
def text_lines(page):
    """Lines of text with boxes in displayed coordinates. Text along a sloping member comes as one line ("slant")."""
    out = []
    for bi, b in enumerate(page.get_text("dict")["blocks"]):
        for ln in b.get("lines", []):
            # spans far apart on one baseline are separate labels (two marks, a mark and a dimension): split them
            runs, cur = [], None
            horiz = abs(ln["dir"][0]) >= abs(ln["dir"][1])
            for s in ln["spans"]:
                if not s["text"].strip():
                    continue
                gap = 0
                if cur:
                    gap = (s["bbox"][0] - cur["box"][2]) if horiz else min(abs(cur["box"][1] - s["bbox"][3]), abs(s["bbox"][1] - cur["box"][3]))
                if cur and abs(gap) <= 1.2 * s["size"]:
                    cur["t"] += s["text"]
                    cur["box"] = [min(cur["box"][0], s["bbox"][0]), min(cur["box"][1], s["bbox"][1]),
                                  max(cur["box"][2], s["bbox"][2]), max(cur["box"][3], s["bbox"][3])]
                else:
                    cur = {"t": s["text"], "box": list(s["bbox"]), "block": bi}
                    runs.append(cur)
            for r in runs:
                r["t"] = re.sub(r"\s+", " ", r["t"]).strip()
                if abs(ln["dir"][0]) > 0.02 and abs(ln["dir"][1]) > 0.02:
                    r["slant"] = True
                if r["t"]:
                    out.append(r)
    return out


# ---------------- linework ----------------
def segments(page):
    """Every straight piece of linework (curves by their chord): (x0, y0, x1, y1)."""
    segs = []
    for d in page.get_drawings():
        for it in d["items"]:
            if it[0] == "l":
                segs.append((it[1].x, it[1].y, it[2].x, it[2].y))
            elif it[0] == "re":
                r = it[1]
                segs += [(r.x0, r.y0, r.x1, r.y0), (r.x1, r.y0, r.x1, r.y1), (r.x1, r.y1, r.x0, r.y1), (r.x0, r.y1, r.x0, r.y0)]
            elif it[0] == "c":
                segs.append((it[1].x, it[1].y, it[4].x, it[4].y))
            elif it[0] == "qu":
                q = it[1]
                pts = [q.ul, q.ur, q.lr, q.ll]
                segs += [(a.x, a.y, b.x, b.y) for a, b in zip(pts, pts[1:] + pts[:1])]
    return segs


def frame_and_titleblock(W, H, segs):
    """Sheet frame = the long lines; title block = the narrow strip the frame cuts off at the right or bottom."""
    vs = sorted({round(s[0]) for s in segs if abs(s[0] - s[2]) < 1 and abs(s[1] - s[3]) > 0.8 * H})
    hs = sorted({round(s[1]) for s in segs if abs(s[1] - s[3]) < 1 and abs(s[0] - s[2]) > 0.8 * W})
    tb = None
    if len(vs) >= 2:
        inner = [x for x in vs if 0.6 * W < x < vs[-1] - 0.04 * W]
        if inner:
            tb = [inner[-1], 0, W, H]
    if tb is None and len(hs) >= 2:
        inner = [y for y in hs if 0.7 * H < y < hs[-1] - 0.03 * H]
        if inner:
            tb = [0, inner[-1], W, H]
    return vs, hs, tb


def inside(box, region, tol=1.0):
    return region is not None and box[0] >= region[0] - tol and box[1] >= region[1] - tol and \
        box[2] <= region[2] + tol and box[3] <= region[3] + tol


# ---------------- views ----------------
def find_titles(lines):
    """A view title is the text line just above a scale line ("1 : 50") and left-aligned with it."""
    titles = []
    for s in lines:
        m = SCALE_RE.match(s["t"])
        if not m:
            continue
        sx0, sy0 = s["box"][0], s["box"][1]
        best = None
        for l in lines:
            if l is s or SCALE_RE.match(l["t"]) or len(l["t"]) < 3:
                continue
            dy = sy0 - l["box"][3]
            if -3 <= dy <= 30 and abs(l["box"][0] - sx0) <= 25 and (best is None or dy < best[0]):
                best = (dy, l)
        if not best:
            continue
        t = best[1]
        # view number: a short token just left of the title (the bubble)
        no = None
        for l in lines:
            if len(l["t"]) <= 4 and 0 < t["box"][0] - l["box"][2] < 60 and \
                    l["box"][1] < s["box"][3] + 4 and l["box"][3] > t["box"][1] - 14:
                no = l["t"]
        title = t["t"]
        m2 = re.match(r"^(\d{1,2}[A-Z]?|[A-Z]\d?)\s+(.{3,})$", title)
        if no is None and m2:
            no, title = m2.group(1), m2.group(2)
        elif no is not None and m2 and m2.group(1) == no:
            title = m2.group(2)
        titles.append({"no": no, "title": title, "scale": s["t"].replace(" ", ""),
                       "box": [min(t["box"][0], s["box"][0]) - (40 if no else 0), t["box"][1],
                               max(t["box"][2], s["box"][2]), s["box"][3]]})
    return titles


def cluster(W, H, segs, lines, skip_regions):
    """Boxes of the white-space-separated groups of linework and text (numpy only: coarse mask, dilate, flood fill)."""
    gw, gh = int(W / CELL) + 2, int(H / CELL) + 2
    mask = np.zeros((gh, gw), bool)
    for x0, y0, x1, y1 in segs:
        if (abs(x0 - x1) < 1 and abs(y0 - y1) > 0.8 * H) or (abs(y0 - y1) < 1 and abs(x0 - x1) > 0.8 * W):
            continue                                              # sheet frame
        if any(inside([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], r) for r in skip_regions):
            continue
        n = int(max(abs(x1 - x0), abs(y1 - y0)) / CELL) + 2
        xs = np.clip((np.linspace(x0, x1, n) / CELL).astype(int), 0, gw - 1)
        ys = np.clip((np.linspace(y0, y1, n) / CELL).astype(int), 0, gh - 1)
        mask[ys, xs] = True
    for l in lines:
        b = l["box"]
        if any(inside(b, r) for r in skip_regions):
            continue
        mask[max(0, int(b[1] / CELL)):min(gh, int(b[3] / CELL) + 1), max(0, int(b[0] / CELL)):min(gw, int(b[2] / CELL) + 1)] = True
    fat = mask.copy()
    for _ in range(int(GAP / CELL)):                              # square dilation, one cell per pass
        f = fat.copy()
        f[1:, :] |= fat[:-1, :]
        f[:-1, :] |= fat[1:, :]
        g = f.copy()
        g[:, 1:] |= f[:, :-1]
        g[:, :-1] |= f[:, 1:]
        fat = g
    lab = np.zeros(fat.shape, bool)
    boxes = []
    for sy, sx in zip(*np.nonzero(fat)):
        if lab[sy, sx]:
            continue
        stack, lab[sy, sx] = [(int(sy), int(sx))], True
        x_lo = x_hi = int(sx)
        y_lo = y_hi = int(sy)
        while stack:
            y, x = stack.pop()
            x_lo, x_hi, y_lo, y_hi = min(x_lo, x), max(x_hi, x), min(y_lo, y), max(y_hi, y)
            for yy in (y - 1, y, y + 1):
                for xx in (x - 1, x, x + 1):
                    if 0 <= yy < gh and 0 <= xx < gw and fat[yy, xx] and not lab[yy, xx]:
                        lab[yy, xx] = True
                        stack.append((yy, xx))
        pad = GAP - CELL
        boxes.append([max(0, x_lo * CELL + pad), max(0, y_lo * CELL + pad), min(W, (x_hi + 1) * CELL - pad), min(H, (y_hi + 1) * CELL - pad)])
    return boxes


def split_by_titles(box, titles):
    """Titles sit under their view: cut a multi-title cluster under each title (stacked) or left of it (side by side)."""
    if len(titles) <= 1:
        return [(box, titles[0] if titles else None)]
    ts = sorted(titles, key=lambda t: (t["box"][1], t["box"][0]))
    stacked = all(b["box"][1] > a["box"][3] for a, b in zip(ts, ts[1:]))
    out = []
    if stacked:
        top = box[1]
        for i, t in enumerate(ts):
            bot = box[3] if i == len(ts) - 1 else t["box"][3] + 6
            out.append(([box[0], top, box[2], bot], t))
            top = bot
    else:
        ts = sorted(titles, key=lambda t: t["box"][0])
        left = box[0]
        for i, t in enumerate(ts):
            right = box[2] if i == len(ts) - 1 else ts[i + 1]["box"][0] - 6
            out.append(([left, box[1], right, box[3]], t))
            left = right
    return out


def centre_in(b, box):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def classify(texts, titles_text):
    """Group a view's text lines. Counts kept ("x3") so a quantity can be read without looking."""
    cnt = {}
    for t in texts:
        cnt[t] = cnt.get(t, 0) + 1
    g = {"sections": [], "marks": [], "levels": [], "grids": [], "dims": [], "refs": [], "text": []}
    for t, n in cnt.items():
        s = f"{t} x{n}" if n > 1 else t
        up = t.upper()
        if t in titles_text or SCALE_RE.match(t):
            continue
        if SECTION_RE.search(t) and not MARK_RE.match(t):
            g["sections"].append(s)
        elif LEVEL_RE.match(t) or (any(up.startswith(w) for w in LEVEL_WORDS) and re.search(r"\d", t)):
            g["levels"].append(s)
        elif DIM_RE.match(t):
            g["dims"].append(s)
        elif MARK_RE.match(t):
            (g["refs"] if re.search(r"-(?:DET|GA|LAY|SEC|DWG|SSTEL|STR)\b|-\d{3,}$", up) and len(t) > 14 else g["marks"]).append(s)
        elif GRID_RE.match(t):
            g["grids"].append(s)
        else:
            g["text"].append(s)
    return {k: v for k, v in g.items() if v}


# ---------------- title block ----------------
def title_block(lines, tb, W, H):
    tl = [l for l in lines if inside(l["box"], tb)] if tb else []
    if not tl:                                    # no strip found: take text near the usual labels
        keys = [l for l in lines if any(k in l["t"].upper() for k in ("DRAWING NO", "DWG NO", "DRAWING TITLE"))]
        if keys:
            x0 = min(k["box"][0] for k in keys) - 20
            tl = [l for l in lines if l["box"][0] >= x0]
    tl.sort(key=lambda l: (round(l["box"][1] / 6), l["box"][0]))
    rows, cur, cy = [], [], None
    for l in tl:
        if cy is None or abs(l["box"][1] - cy) > 6:
            if cur:
                rows.append(cur)
            cur, cy = [], l["box"][1]
        cur.append(l)
    if cur:
        rows.append(cur)
    text = [" | ".join(l["t"] for l in sorted(r, key=lambda l: l["box"][0])) for r in rows]
    fields = {}

    def is_label(t):
        up = t.upper().strip()
        return up.endswith(":") or any(up.startswith(k) for k in TB_LABELS)

    anchor = next((l for l in tl if l["t"].upper().startswith(("DRAWING NO", "DWG NO"))), None)

    def value_after(label):
        """Value right of / below a label. Several labels with the same name (REV, DATE also head the revision
        table): take the one nearest the drawing number."""
        occ = [l for l in tl if l["t"].upper().startswith(label)]
        if not occ:
            return None
        if anchor:
            occ.sort(key=lambda l: abs(l["box"][0] - anchor["box"][0]) + abs(l["box"][1] - anchor["box"][1]))
        l = occ[0]
        rest = l["t"][len(label):].strip(" :.")
        if rest.upper().startswith("NO"):
            rest = rest[2:].strip(" :.")
        if rest:
            return rest
        best = None
        for o in tl:
            if o is l or is_label(o["t"]) or o["box"][2] <= l["box"][0] or o["box"][3] <= l["box"][1] + 1:
                continue
            dx, dy = max(0.0, o["box"][0] - l["box"][2]), max(0.0, o["box"][1] - l["box"][3])
            d = dx + 2 * dy
            if d < 160 and (best is None or d < best[0]):
                best = (d, o["t"])
        return best and best[1]

    for key, label in (("dwg_no", "DRAWING NO"), ("dwg_no", "DWG NO"), ("rev", "REV"), ("scale", "SCALE"),
                       ("date", "DATE"), ("status", "PROJECT PHASE"), ("status", "STATUS"), ("job_no", "JOB NO"),
                       ("sheet", "SHEET")):
        if key not in fields:
            v = value_after(label)
            if v:
                fields[key] = v
    # drawing title = the lines between "DRAWING TITLE" and the next label row
    for i, r in enumerate(rows):
        if any(l["t"].upper().startswith("DRAWING TITLE") for l in r):
            ttl = []
            for r2 in rows[i + 1:]:
                if any(any(l["t"].upper().startswith(k) for k in TB_LABELS + ("DESIGNED",)) for l in r2):
                    break
                ttl.append(" ".join(l["t"] for l in r2))
            if ttl:
                fields["title"] = " / ".join(ttl)
    return fields, text


# ---------------- one page ----------------
def read_page(page_no, page):
    page.remove_rotation()                                 # coordinates as displayed, whatever the page rotation
    W, H = float(page.rect.width), float(page.rect.height)
    return read_page_data(page_no, W, H, text_lines(page), segments(page), member_widths(page))


def read_store_sheet(sheet):
    """The same reading from the drawing store (generate-model/store.py): no PDF opened."""
    lines = [{"t": w[4], "box": list(w[:4])} | ({"slant": True} if w[5] % 90 else {}) for w in sheet.words]
    segs = [tuple(map(float, r[:4])) for r in sheet.segs]
    return read_page_data(sheet.page, float(sheet.size[0]), float(sheet.size[1]), lines, segs, None)


def read_page_data(page_no, W, H, lines, segs, widths):
    _, _, tb = frame_and_titleblock(W, H, segs)
    fields, tb_text = title_block(lines, tb, W, H)
    body = [l for l in lines if not inside(l["box"], tb)]
    titles = find_titles(body)
    views = []
    boxes = sorted(cluster(W, H, segs, body, [tb] if tb else []), key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    boxes = [b for i, b in enumerate(boxes) if not any(inside(b, o, 2.0) for o in boxes[:i])]      # text inside an outline belongs to it
    # a title standing clear below its view: join the title's cluster to the nearest cluster above it
    for t in titles:
        own = next((b for b in boxes if centre_in(t["box"], b)), None)
        if own is None or (own[3] - own[1]) > (t["box"][3] - t["box"][1]) + 30:
            continue                                        # the title already sits inside its view's cluster
        above = [b for b in boxes if b is not own and b[3] <= own[1] + 2 and min(b[2], own[2]) - max(b[0], own[0]) > 0
                 and own[1] - b[3] < 150 and not any(centre_in(o["box"], b) for o in titles)]
        if above:
            host = min(above, key=lambda b: own[1] - b[3])
            boxes[boxes.index(host)] = [min(host[0], own[0]), min(host[1], own[1]), max(host[2], own[2]), max(host[3], own[3])]
            boxes = [b for b in boxes if b is not own]
    for box in boxes:
        inside_titles = [t for t in titles if centre_in(t["box"], box)]
        for vbox, t in split_by_titles(box, inside_titles):
            texts = [l for l in body if centre_in(l["box"], vbox)]
            if not texts and (vbox[2] - vbox[0]) * (vbox[3] - vbox[1]) < 0.002 * W * H:
                continue                                   # stray mark without text
            views.append({"box": [round(v, 1) for v in vbox], "title": t, "lines": texts})
    views.sort(key=lambda v: (v["title"] is None, round(v["box"][0] / (W / 6)), v["box"][1]))
    out, used = [], set()
    for i, v in enumerate(views):
        t = v["title"]
        texts = [l["t"] for l in v["lines"]]
        if t:
            vid = t["no"] or f"V{i + 1}"
            name, scale = t["title"], t["scale"]
        else:
            first = sorted(v["lines"], key=lambda l: (l["box"][1], l["box"][0]))
            name = first[0]["t"] if first else "(linework only)"
            vid, scale = f"X{i + 1}", None
        while vid in used:
            vid += "'"
        used.add(vid)
        tt = {t["title"], f'{t["no"]} {t["title"]}', t["no"]} if t else set()
        entry = {"id": vid, "name": name, "scale": scale, "box": v["box"], "titled": bool(t)}
        if t:
            entry["items"] = classify(texts, tt)
        else:                                               # legend, notes, schedule: keep the reading order
            rows = sorted(v["lines"], key=lambda l: (round(l["box"][1] / 5), l["box"][0]))
            merged, cy = [], None
            for l in rows:
                if cy is not None and abs(l["box"][1] - cy) <= 5:
                    merged[-1] += " | " + l["t"]
                else:
                    merged.append(l["t"])
                cy = l["box"][1]
            entry["rows"] = merged
        entry["words"] = [{"t": l["t"], "box": [round(b, 1) for b in l["box"]]} for l in v["lines"]]
        out.append(entry)
    words = sum(len(l["t"].split()) for l in lines)
    # text AND linework: a revision that moves a member without touching the text is still "changed"
    geo = sorted(tuple(round(v) for v in g) for g in segs)
    digest = hashlib.sha1(("\n".join(sorted(l["t"] for l in lines)) + repr(geo)).encode()).hexdigest()[:12]
    return {"page": page_no, "size_pt": [W, H], "titleblock": fields, "titleblock_text": tb_text,
            "titleblock_box": tb, "views": out, "words": words, "hash": digest, "widths": widths,
            "flag": None if words >= MIN_WORDS else "scan_or_outlined_text"}


def member_widths(page, off=9.0):
    """Which black line weights carry member marks on this sheet (the drawing's own member pens), so a reader is set
    from the data instead of a guessed 0.9-1.4 pt. -> [{"w": pt, "len": pt drawn, "marked": lines with a mark beside
    them, "share": marked length / drawn length}], member pens first."""
    import read_drawing as rd
    marks = []
    for b in page.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            t = "".join(sp["text"] for sp in ln["spans"]).strip()
            w0 = t.split()[0] if t else ""
            if w0 and (MARK_RE.match(w0) or SHORT_MARK_RE.match(w0) or SECTION_RE.match(t)):
                bb = ln["bbox"]
                marks.append(((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2, ln["dir"]))
    if not marks:
        return []
    grid = {}
    for m in marks:
        grid.setdefault((int(m[0] // 40), int(m[1] // 40)), []).append(m)
    by_w = {}
    for a in rd.segments(page):
        if math.hypot(a[2] - a[0], a[3] - a[1]) >= 0.3:
            by_w.setdefault(round(a[4], 2), []).append(a)
    out = []
    for w, segs in by_w.items():
        lines = [m for m in rd.merge_lines(segs, 2.0) if math.hypot(m[2] - m[0], m[3] - m[1]) > 15]
        tot = sum(math.hypot(m[2] - m[0], m[3] - m[1]) for m in lines)
        if tot < 300:
            continue
        hit = hit_len = 0
        for m in lines:
            L = math.hypot(m[2] - m[0], m[3] - m[1]); tx, ty = (m[2] - m[0]) / L, (m[3] - m[1]) / L
            gx0, gx1 = int(min(m[0], m[2]) // 40) - 1, int(max(m[0], m[2]) // 40) + 1
            gy0, gy1 = int(min(m[1], m[3]) // 40) - 1, int(max(m[1], m[3]) // 40) + 1
            if (gx1 - gx0) * (gy1 - gy0) > 4000:
                continue                                    # frame / border lines
            for gx in range(gx0, gx1 + 1):
                if any(abs(tx * d[0] + ty * d[1]) > 0.97 and -2 <= (cx - m[0]) * tx + (cy - m[1]) * ty <= L + 2
                       and abs(-(cx - m[0]) * ty + (cy - m[1]) * tx) < off
                       for gy in range(gy0, gy1 + 1) for cx, cy, d in grid.get((gx, gy), ())):
                    hit += 1; hit_len += L
                    break
        out.append({"w": w, "len": round(tot), "marked": hit, "share": round(hit_len / tot, 2)})
    out.sort(key=lambda r: -r["marked"] * r["share"])
    return out[:6]


# ---------------- output ----------------
def paper(W, H):
    mm = sorted((W * 25.4 / 72, H * 25.4 / 72))
    for name, (a, b) in {"A0": (841, 1189), "A1": (594, 841), "A2": (420, 594), "A3": (297, 420), "A4": (210, 297)}.items():
        if abs(mm[0] - a) < 8 and abs(mm[1] - b) < 8:
            return name
    return f"{mm[1]:.0f}x{mm[0]:.0f} mm"


INDEX_VERSION = "2"     # bump when read_page changes: cached pages from an older index are read again
AUTO_KEYS = {"drawings", "notes", "_about"}          # reading.json keys this script writes; everything else is the agent's


def _sheet_md(s):
    f = s["titleblock"]
    L = [f'## {s["sheet"]}  ({s["file"]} p{s["page"]}, {paper(*s["size_pt"])})']
    if s["flag"]:
        L.append(f'**{s["flag"]}: only {s["words"]} words of real text. Use the thumbnail and crops.**')
    L.append("Title block: " + "; ".join(f"{k}={v}" for k, v in f.items()))
    vals = {str(v) for v in f.values()} | set(str(f.get("title", "")).split(" / "))
    rest = [t for t in s["titleblock_text"] if t not in vals and not t.rstrip().endswith(":")
            and not all(p.strip() in vals or p.strip().endswith(":") for p in t.split("|"))]
    if rest:
        L.append("Other title-block text: " + " ¦ ".join(rest)[:600])
    if s.get("widths"):
        L.append("Line weights with marks beside them: " + ", ".join(
            f'{w["w"]} pt ({w["marked"]} marked lines, {int(w["share"] * 100)} % of its length)' for w in s["widths"] if w["marked"]))
    for v in s["views"]:
        L.append(f'### {v["id"]}: {v["name"]}' + (f' ({v["scale"]})' if v["scale"] else ""))
        if "rows" in v:
            L += [f"  {r}" for r in v["rows"]]
        else:
            for k, vals in v["items"].items():
                L.append(f"  {k}: " + ", ".join(vals))
    return L


def _pens(s):
    return " / ".join(f'{w["w"]}' for w in s.get("widths", []) if w["marked"] >= 3 and w["share"] >= 0.15) or "-"


def write_index(sheets, out, notes=()):
    """index.md = the overview (about 60 tokens a sheet); index/<sheet>.md = one sheet in full (about 1 k tokens)."""
    out = pathlib.Path(out)
    (out / "index").mkdir(exist_ok=True)
    nn = {}
    for n in notes:
        nn[n.get("sheet")] = nn.get(n.get("sheet"), 0) + 1
    L = ["# Drawing index: overview. Open `index/<sheet>.md` only for the sheet you need.", "",
         "Look things up in this order, never re-read what is already answered: reading.json (incl. `notes`) -> "
         "`read_drawing.py find <read dir> REGEX` -> `index/<sheet>.md` -> `crop` (once per view; write what you saw with "
         "`read_drawing.py note`).", "",
         "| Sheet | Dwg no | Rev | Status | Title | Views | Member pens (pt) | Notes | Changed |",
         "|---|---|---|---|---|---|---|---|---|"]
    for s in sheets:
        f = s["titleblock"]
        L.append(f'| {s["sheet"]} | {f.get("dwg_no", "?")} | {f.get("rev", "?")} | {f.get("status", "?")} | '
                 f'{f.get("title", "?")} | {sum(v["titled"] for v in s["views"])} | {_pens(s)} | {nn.get(s["sheet"], "")} | '
                 f'{s.get("changed", "")}{" " + s["flag"] if s["flag"] else ""} |')
    L.append("")
    for s in sheets:
        vs = [f'{v["id"]} {v["name"][:40]}' + (f' ({v["scale"]})' if v["scale"] else "") +
              (f' [{len(v["items"].get("marks", []))} marks]' if v.get("items", {}).get("marks") else "")
              for v in s["views"] if v["titled"]]
        L.append(f'- {s["sheet"]}: ' + ("; ".join(vs) if vs else "(no titled views)"))
        (out / "index" / f'{s["sheet"]}.md').write_text("\n".join(_sheet_md(s)) + "\n", encoding="utf-8")
    (out / "index.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def _load(p, default):
    try:
        return json.loads(pathlib.Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def seed_reading(sheets, out):
    """reading.json: the title blocks, scales, grids, levels and member pens of every sheet are filled in by the index,
    so nobody transcribes them; the agent's own keys (members, assumptions, left out, ...) and its notes are kept."""
    rp = pathlib.Path(out) / "reading.json"
    r = _load(rp, {})
    r["_about"] = ("drawings + notes are kept by read_drawing.py (index / note); add the reading itself under your own keys "
                   "(members, grids, levels per frame, assumptions, left_out). Read this before opening any sheet.")
    dr = []
    for s in sheets:
        f = s["titleblock"]
        lv, gr = set(), set()
        for v in s["views"]:
            lv.update(x.split(" x")[0] for x in v.get("items", {}).get("levels", []))
            gr.update(x.split(" x")[0] for x in v.get("items", {}).get("grids", []))
        dr.append({"sheet": s["sheet"], "dwg_no": f.get("dwg_no"), "rev": f.get("rev"), "status": f.get("status"),
                   "title": f.get("title"), "file": s["file"], "page": s["page"], "changed": s.get("changed"),
                   "scales": sorted({v["scale"] for v in s["views"] if v["scale"]}), "member_pens_pt": _pens(s),
                   "levels": sorted(lv)[:40], "grids": sorted(gr)[:60], "flag": s["flag"]})
    r["drawings"] = dr
    r.setdefault("notes", [])
    rp.write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")


def build_index(inputs, out, prev=None, thumbs=True):
    """Pages of a PDF that has not changed since the last index are taken from the cache, not read again. Without
    --prev the previous sheets.json in the same folder is the comparison, so a new revision marks its changed sheets."""
    pdfs = []
    for i in inputs:
        p = pathlib.Path(i)
        pdfs += sorted(x for x in p.rglob("*.pdf") if "out" not in x.relative_to(p).parts[:-1]) if p.is_dir() else [p]
    out = pathlib.Path(out)
    (out / "thumbs").mkdir(parents=True, exist_ok=True)
    (out / "_pages").mkdir(exist_ok=True)
    if prev is None and (out / "sheets.json").exists():
        prev = out / "sheets.json"
    old = {}
    if prev:
        for s in _load(prev, {"sheets": []})["sheets"]:
            old[s["titleblock"].get("dwg_no") or s["sheet"]] = s
    sheets, cached = [], 0
    for pdf in pdfs:
        sha = hashlib.sha1(pdf.read_bytes()).hexdigest()[:16]
        doc = None
        n_pages = pymupdf.open(pdf).page_count
        for n in range(1, n_pages + 1):
            cp = out / "_pages" / f"{sha}_{n}_v{INDEX_VERSION}.json"
            s = _load(cp, None)
            if s is None:
                doc = doc or pymupdf.open(pdf)
                page = doc[n - 1]
                s = read_page(n, page)
                base = re.sub(r"[^A-Za-z0-9_.-]+", "_", s["titleblock"].get("dwg_no") or pdf.stem)
                s["sheet"] = base if n_pages == 1 or s["titleblock"].get("dwg_no") else f"{base}_p{n}"
                if thumbs:
                    W, H = s["size_pt"]
                    s["thumb"] = str(out / "thumbs" / (s["sheet"] + ".png"))
                    page.get_pixmap(dpi=max(20, int(THUMB_PX / max(W, H) * 72))).save(s["thumb"])
                cp.write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
            else:
                cached += 1
            s["file"] = str(pdf)
            s.pop("changed", None)
            if prev:
                o = old.get(s["titleblock"].get("dwg_no") or s["sheet"])
                s["changed"] = "new" if o is None else ("same" if o["hash"] == s["hash"] else
                                                         f'changed (was rev {o["titleblock"].get("rev", "?")})')
            sheets.append(s)
    reading = _load(out / "reading.json", {})
    (out / "sheets.json").write_text(json.dumps({"sheets": sheets}, ensure_ascii=False), encoding="utf-8")
    write_index(sheets, out, reading.get("notes", []))
    seed_reading(sheets, out)
    size = (out / "index.md").stat().st_size
    ch = [s["sheet"] for s in sheets if s.get("changed") and s["changed"] != "same"]
    print(f"{len(sheets)} sheet(s), {sum(len(s['views']) for s in sheets)} views -> {out}/index.md "
          f"({size:,} bytes, about {size // 4:,} tokens; one sheet in full: index/<sheet>.md)"
          + (f"; {cached} page(s) unchanged, taken from the cache" if cached else ""))
    if prev:
        print(f"  new or changed since the last index: {len(ch)}" + (": " + ", ".join(ch[:30]) if ch else " (re-read nothing)"))
    for s in sheets:
        if s["flag"]:
            print(f"  {s['sheet']}: {s['flag']}", file=sys.stderr)


# ---------------- reading memory: find / note / status ----------------
def _notes_for(read_dir, sheet, view=None):
    ns = _load(pathlib.Path(read_dir) / "reading.json", {}).get("notes", [])
    return [n for n in ns if n.get("sheet") == sheet and (view is None or n.get("view") in (None, view))]


def find(read_dir, pattern, sheet=None, view=None, limit=12):
    """Search every word of every sheet (and the notes) from sheets.json: no PDF is opened, no image is made.
    Prints hit counts per sheet / view and the first hits with their position (pt), for crop --box or a script."""
    rd = pathlib.Path(read_dir)
    rx = re.compile(pattern, re.I)
    sheets = _load(rd / "sheets.json", {"sheets": []})["sheets"]
    total, shown = 0, 0
    for s in sheets:
        if sheet and s["sheet"] != sheet:
            continue
        tb = [t for t in s.get("titleblock_text", []) if rx.search(t)]
        if tb and not view:
            print(f'{s["sheet"]} title block: ' + " ¦ ".join(tb[:4]))
            total += len(tb)
        for v in s["views"]:
            if view and v["id"].lower() != view.lower():
                continue
            hits = [w for w in v["words"] if rx.search(w["t"])]
            if not hits:
                continue
            total += len(hits)
            cnt = {}
            for w in hits:
                cnt[w["t"]] = cnt.get(w["t"], 0) + 1
            txt = ", ".join(f"{t} x{c}" if c > 1 else t for t, c in sorted(cnt.items())[:limit])
            line = f'{s["sheet"]} {v["id"]} ({v["name"][:30]}{", " + v["scale"] if v["scale"] else ""}): {len(hits)} hit(s): {txt}'
            if shown < limit:
                line += "  at " + " ".join(f'({(w["box"][0] + w["box"][2]) / 2:.0f},{(w["box"][1] + w["box"][3]) / 2:.0f})'
                                           for w in hits[:4])
                shown += 1
            print(line)
    for n in _load(rd / "reading.json", {}).get("notes", []):
        if rx.search(n.get("text", "")) and (not sheet or n.get("sheet") == sheet):
            print(f'NOTE {n.get("sheet")} {n.get("view") or ""}: {n["text"]}')
            total += 1
    print(f"{total} hit(s) for /{pattern}/" + ("" if total else " (not on these sheets: do not crop to look for it)"))


CONN_WORDS = re.compile(r"\b(?:BASE\s*PLATE|SPLICE|END\s*PLATE|FIN\s*PLATE|CLEAT|ANCHOR\s*BOLT|HOLDING\s*DOWN|H\.?D\.?\s*BOLT|"
                        r"CAP\s*PLATE|STIFFENER|GUSSET|CONNECTION)S?\b", re.I)
CONN_CALL = re.compile(r"\b\d+\s*[-xX/]?\s*M(?:12|16|20|24|27|30|36)\b|\bM(?:12|16|20|24|27|30|36)\b|\bPL\s?\d+(?:\s?[xX]\s?\d+)*\b|"
                       r"\b\d+\s*(?:mm)?\s*THK\b", re.I)
CONN_TAG = re.compile(r"^(?:BP|SP|EP|FP|CP)\d{1,3}[A-Za-z]?$")   # not BC / SC / MC: those are member marks (chords, columns)
CONN_NOT = re.compile(r"LAP[\s-]*SPLICE|SPLICE[DS]?\s+(?:BARS?|LENGTH)", re.I)


def connections(read_dir):
    """Every connection callout on the drawings, from sheets.json (no PDF opened, no image, no OCR): the detail views
    (title / callouts: plates, bolts, thicknesses) and the joint tags placed on plans (BP3 at its position, pt).
    Writes connections_found.json beside the reading. Read each detail view ONCE (crop + note), put its data in
    reading.json `connections` / the spec's conn_types, then place it at every joint its tag or legend symbol marks."""
    rd = pathlib.Path(read_dir)
    sheets = _load(rd / "sheets.json", {"sheets": []})["sheets"]
    views, tags = [], {}
    for s in sheets:
        for v in s["views"]:
            words = v.get("words", [])
            text = CONN_NOT.sub(" ", " ".join(w["t"] for w in words))
            kinds = sorted({m.group(0).upper() for m in CONN_WORDS.finditer(text + " " + v.get("name", ""))})
            calls = [m.group(0) for m in CONN_CALL.finditer(text)]
            for w in words:
                if CONN_TAG.match(w["t"]):
                    b = w["box"]
                    tags.setdefault(w["t"].upper(), []).append({"sheet": s["sheet"], "view": v["id"],
                                                                "xy": [round((b[0] + b[2]) / 2, 1), round((b[1] + b[3]) / 2, 1)]})
            if kinds and (calls or re.search(r"DETAIL|CONNECTION|SPLICE|BASE", v.get("name", ""), re.I)):
                views.append({"sheet": s["sheet"], "view": v["id"], "name": v.get("name", ""), "scale": v.get("scale"),
                              "kinds": kinds, "callouts": sorted(set(calls))[:40]})
    out = {"detail_views": views, "tags": tags}
    (rd / "connections_found.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    per = {}
    for v in views:
        per.setdefault(v["sheet"], []).append(v["view"])
    print(f"{len(views)} connection detail view(s) on {len(per)} sheet(s); {len(tags)} joint tag(s) placed on the plans")
    for sh, vs in list(per.items())[:12]:
        print(f"  {sh}: {', '.join(vs[:8])}" + (" ..." if len(vs) > 8 else ""))
    if tags:
        print("  tags: " + ", ".join(f"{k} x{len(v)}" for k, v in sorted(tags.items())[:20]))
    if not views and not tags:
        print("  none: no connection shown on the drawings -> every joint welded, zero gap (record it as a decision)")
    print(f"-> {rd / 'connections_found.json'}")
    return out


def note(read_dir, sheet, text, view=None):
    """Write down what a crop showed, so nobody looks at it again (kept in reading.json `notes`)."""
    rp = pathlib.Path(read_dir) / "reading.json"
    r = _load(rp, {})
    names = {s["sheet"] for s in _load(pathlib.Path(read_dir) / "sheets.json", {"sheets": []})["sheets"]}
    if names and sheet not in names:
        sys.exit(f"no sheet {sheet!r}. Sheets: " + " ".join(sorted(names)))
    import datetime
    r.setdefault("notes", []).append({"sheet": sheet, "view": view, "text": text.strip(),
                                      "at": datetime.datetime.now().isoformat(timespec="minutes")})
    rp.write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f'note {len(r["notes"])} kept for {sheet}{" " + view if view else ""}')


def status(read_dir):
    """What has been read already: per sheet the crops taken and the notes written; sheets nobody opened."""
    rd = pathlib.Path(read_dir)
    sheets = _load(rd / "sheets.json", {"sheets": []})["sheets"]
    crops = _load(rd / "ledger.json", {}).get("crops", {})
    notes = _load(rd / "reading.json", {}).get("notes", [])
    agent = sorted(k for k in _load(rd / "reading.json", {}) if k not in AUTO_KEYS)
    untouched = []
    for s in sheets:
        nc = sum(1 for c in crops.values() if c["sheet"] == s["sheet"])
        nn = sum(1 for n in notes if n.get("sheet") == s["sheet"])
        if not nc and not nn:
            untouched.append(s["sheet"])
            continue
        print(f'{s["sheet"]} rev {s["titleblock"].get("rev", "?")}: {nc} crop(s), {nn} note(s)'
              + (f' [{s["changed"]}]' if s.get("changed") not in (None, "same") else ""))
    print(f"{len(sheets)} sheets, {len(crops)} crops, {len(notes)} notes; reading.json keys of the agent: {', '.join(agent) or 'none yet'}")
    if untouched:
        print(f"not cropped and no note ({len(untouched)}): " + " ".join(untouched[:40]))


# ---------------- crop ----------------
def _render(sheet, box, out, max_px, dpi, label, scale, view=None):
    W, H = sheet["size_pt"]
    x0, y0, x1, y1 = max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])
    if not dpi:
        dpi = int(min(300, max_px / max(x1 - x0, y1 - y0) * 72))
    png = out / (re.sub(r"[^A-Za-z0-9_.-]+", "_", f'{sheet["sheet"]}_{label}_{dpi}dpi') + ".png")
    note_ = f", 1 px = {scale * 25.4 / dpi:.2f} mm real (scale 1:{scale:g})" if scale else ""
    led_p = out.parent / "ledger.json"
    led = _load(led_p, {})
    key = f'{sheet["sheet"]}|{sheet.get("hash", "")}|{round(x0)},{round(y0)},{round(x1)},{round(y1)}|{dpi}'
    seen = led.get("crops", {}).get(key)
    notes = _notes_for(out.parent, sheet["sheet"], view)
    if seen and pathlib.Path(seen["png"]).exists():
        print(f'SEEN BEFORE (crop {seen["n"]}): {seen["png"]}{note_}. Use the notes below instead of opening it again.')
    else:
        page = pymupdf.open(sheet["file"])[sheet["page"] - 1]
        page.remove_rotation()
        pm = page.get_pixmap(dpi=dpi, clip=pymupdf.Rect(x0, y0, x1, y1))
        pm.save(str(png))
        led.setdefault("crops", {})[key] = {"n": len(led.get("crops", {})) + 1, "png": str(png), "sheet": sheet["sheet"],
                                            "view": view, "box": [round(v) for v in (x0, y0, x1, y1)], "dpi": dpi}
        led_p.write_text(json.dumps(led, indent=0), encoding="utf-8")
        print(f"{png}  {pm.width}x{pm.height} px, {dpi} dpi{note_}, box {[round(v) for v in (x0, y0, x1, y1)]}")
        print(f'  after looking: python generate-model/read_drawing.py note {out.parent} --sheet {sheet["sheet"]}'
              + (f" --view {view}" if view else "") + ' "what it shows"')
    for n in notes:
        print(f'  NOTE {n.get("view") or "sheet"}: {n["text"]}')


def crop(read_dir, view=None, find=None, box=None, sheet=None, pad=25, max_px=1400, dpi=None, part=None):
    rd = pathlib.Path(read_dir)
    sheets = json.loads((rd / "sheets.json").read_text(encoding="utf-8"))["sheets"]
    if sheet:
        sheets = [s for s in sheets if s["sheet"] == sheet]
    out = rd / "crops"
    out.mkdir(exist_ok=True)

    def quarter(b):
        if not part:
            return b
        mx, my = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        ov = 0.06 * max(b[2] - b[0], b[3] - b[1])               # overlap so nothing is cut at the seam
        return [b[0] if part[1] == "l" else mx - ov, b[1] if part[0] == "t" else my - ov,
                mx + ov if part[1] == "l" else b[2], my + ov if part[0] == "t" else b[3]]

    def scale_of(v):
        m = re.match(r"1:(\d+(?:\.\d+)?)$", v.get("scale") or "")
        return float(m.group(1)) if m else None

    n = 0
    for s in sheets:
        if box:
            _render(s, quarter(box), out, max_px, dpi, "box" + (part or ""), None)
            n += 1
            continue
        for v in s["views"]:
            if view:
                if v["id"].lower() == view.lower():
                    b = v["box"]
                    _render(s, quarter([b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad]), out, max_px, dpi,
                            "view" + v["id"] + (part or ""), scale_of(v), v["id"])
                    n += 1
            else:
                hits = [w["box"] for w in v["words"] if find.lower() in w["t"].lower()]
                if hits:
                    b = [min(h[0] for h in hits) - 90 - pad, min(h[1] for h in hits) - 90 - pad,
                         max(h[2] for h in hits) + 90 + pad, max(h[3] for h in hits) + 90 + pad]
                    _render(s, b, out, max_px, dpi, f'find_{find}_in_{v["id"]}', scale_of(v), v["id"])
                    n += 1
    if not n:
        sys.exit("nothing matched. Views: " + ", ".join(f'{s["sheet"]}: ' + " ".join(v["id"] for v in s["views"]) for s in sheets))
