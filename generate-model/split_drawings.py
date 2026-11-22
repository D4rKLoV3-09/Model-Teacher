#!/usr/bin/env python3
"""Step 0 of every drawing job: make related non-vector sheets true vector (vectorize_drawings.py: outlined text ->
OCR real-text layer, scans -> potrace outlines + text; unrelated sheets untouched), then split a drawing package into one PDF per building (steel + the drawings that
support it) plus one "unrelated" PDF, check the package for drawing issues, rank the buildings, name the primary,
and write lean reading copies (title block blanked except drawing title / number). Never asks.

    python generate-model/split_drawings.py --job JOB                     (Claude Code on the curator's PC, the normal way)
    python generate-model/split_drawings.py PDF_OR_FOLDER [...] --out DIR [--desktop] [--no-ocr] [--no-lean] [--no-vectorize]

    python generate-model/split_drawings.py --project "jobs/NEW PROJECT"   (SplitDrawings.exe does this)
--project DIR  the curator's new project folder with his drawings loose in it or in DIR/Drawings: makes the job tree
               DIR/{Drawings/{Original Drawings, Organized Drawings}, Models/Models Data, MTO}, moves the drawings
               into Original Drawings, splits into Organized Drawings
--job JOB      the same for jobs/JOB
--desktop      write into <Desktop>/<project name>/ instead (a claude.ai session copies --out there via the device bridge)
Writes: NN_<building>.pdf (01 = primary; the lean copy, no full copy), 99_unrelated.pdf, issues.md, split.md,
manifest.json. Prints about 20 lines; never page text.

Per page (PyMuPDF; OCR only on title blocks whose text is outlined, rapidocr-onnxruntime, about 3 s a page):
  number   the "DRAWING NO" of the title block, else the first drawing-number token in the text (later ones are
           reference drawings); a drawing list (rows "No  DWG-NO  Title  Rev  sheets") corrects pages whose number
           is a reference drawing.
  title    the lines under "DRAWING TITLE" in the title block (text, else OCR), else the list title, else the
           title of an earlier sheet of the same number family.
  rev / sheet x of y / status / project: from the title block when present.
Building = title text before the first discipline word (steel, structure, architectural, foundation, plan, ...);
discipline from the title, else the number code (ST / GA / FD). Every building with structural or architectural
sheets is a group; foundation-only and pages with no building -> unrelated. A group whose readable sheets show no
steel structure word and no member size (site plan, M&E layout, RC-only) -> unrelated (the curator's rule, NO STEEL issue).
Then sheet by sheet (model_units): only sheets that show steel members stay; a structure drawn on its own sheets (canopy,
staircase, link bridge, cladding ...) is its own unit; architectural elevations / sections join a unit with no steel
elevation or section (levels). The rest -> unrelated (NOT FOR THE MODEL issue).
Score = 3 x structural + 1 x architectural + 0.5 x foundation sheets + 0.5 x distinct steel sections named +
content-stream MB. Highest = primary (01).
Issues: exact duplicate pages, same drawing + sheet twice with different content, several revisions of one drawing
(older ones superseded), sheets missing against the drawing list or against "sheet x of y", pages from another
project, status not FOR CONSTRUCTION, pages with no drawing number, near-blank pages, outlined-text pages (OCR used).
"""
import argparse
import hashlib
import json
import re
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pymupdf as fitz

DWG = re.compile(r"\b[A-Z]{2,4}[-_]\d{2}[-_][A-Z]{2,4}[-_][A-Z]{1,3}[-_][A-Z][-_][A-Z]{2}[-_]\d{4,6}(?:_[A-Z]{1,2})?\b")
# the drawing number label in any firm's words: DRAWING NO. / DRAWING NUMBER / DRAWING NUM: / DWG NO / DRG. NO. / DRAWING # /
# SHEET NO. / DOCUMENT NO. / DOC NO / NO. or NUMBER alone on its line
NO_LABEL = (r"(?:DRAWINGS?|DWGS?|DRGS?|SHEET|DOCUMENT|DOC)\s*\.?\s*(?:NOS?\b\.?|NUM\b\.?|NUMBERS?|NBR|#)"
            r"|^\s*(?:NO|NUMBER)\s*[.:]?\s*$")
# the drawing title label in any firm's words: DRAWING TITLE / DRAWING NAME / DWG. TITLE / SHEET TITLE / DRAWING DESCRIPTION /
# TITLE / TITLES / TITLE OF DRAWING alone on its line (a lone DESCRIPTION is a revision-table column, not the title)
TITLE_LABEL = (r"(?:DRAWINGS?|DWGS?|DRGS?|SHEET)\s*\.?\s*(?:TITLES?|NAMES?|DESCRIPTIONS?)\b"
               r"|^\s*TITLES?(?:\s+OF\s+(?:THE\s+)?(?:DRAWING|SHEET))?\s*(?:[.:]|$)")
DWG_LOOSE = re.compile(r"\b[A-Z0-9]{2,}(?:[-_][A-Z0-9]{1,8}){3,}\b")
DISC_WORDS = r"(steel|structur\w*|architect\w*|foundation|piling|pile|lintel|roof|truss\w*|layout|plan|plans|elevation\w*|" \
             r"section\w*|details?|framing|general|arrangement|grid|beam|column|base|drawing|staircase|platform)"
STEELY = re.compile(r"\b(steel|truss|purlin|girt|bracing|rafter|portal|shelter|canopy|crane|framing)\b", re.I)
# steel shown on the sheet itself: a steel-structure word or a member size. "shelter" / "canopy" are building names
# (a site plan or M&E layout names every shelter), not steel.
STEEL_WORK = re.compile(r"\b(steel\w*|truss(es)?|purlins?|girts?|bracings?|rafters?|portal\s+frames?|crane\s+beams?|"
                        r"framing\s+plan)\b", re.I)


def no_steel(ps):
    """the curator's rule: a group whose sheets show no steel structure and no member size is not a steel building
    (site plan, M&E / services layout, RC-only sheets) -> unrelated. Decided only when every sheet's text is
    readable (true vector); an outlined or scanned sheet is not read yet, so it is kept."""
    return all(p.get("kind") == "vector" for p in ps) and not any(p.get("steelwork") or p["secs"] for p in ps)
SECTION = re.compile(r"\b(UB|UC|PFC|SHS|RHS|CHS|UA|HEA|HEB|IPE|W)\s?\d{2,4}\s?[xX]\s?\d{2,4}(?:\s?[xX]\s?[\d.]+)?"
                     # the size before the type: 400x200x8.3mm RHS, 203x203x46kKG/M UC, 250x150x6mm Thk. RHS
                     r"|\b\d{2,4}\s?[xX]\s?\d{2,4}(?:\s?[xX]\s?[\d.]+)?\s*(?:mm|k?kg/m)?\s*(?:thk\.?)?\s*(?:RHS|SHS|CHS|UB|UC|PFC|UA|EA|TFC)\b",
                     re.I)
# ---- the curator's rule (2026-11-18): a sheet is for the model only when it shows steel members; a unit is anything that
# stands alone or is drawn as its own thing (2 buildings + 1 link bridge drawn apart = 3 units) ----
# a steel structure named in the title (structural sheets: the drawing's own subject)
STEEL_TITLE = re.compile(r"\b(steel\w*|M\.\s?S\b\.?|mild\s+steel|truss\w*|purlins?|girts?|rafters?|bracings?|portal|cladd\w*|"
                         r"canopy|metal\s+roof|link\s*bridge|bridge|walkway|linkway|platform|mezzanine|flag\s*pole|pipe\s*rack)\b", re.I)
# architectural sheets: only a title that says steel (the architect's plans name every steel staircase in passing)
STEEL_TITLE_ARCH = re.compile(r"\b(steel\w*|M\.\s?S\b\.?|mild\s+steel|truss\w*|purlins?|metal\s+roof\s+framing)\b", re.I)
# steel members named in the drawing body
STRONG = re.compile(r"\b(truss\w*|purlins?|girts?|rafters?|bracings?|base\s*plates?|anchor\s*bolts?|holding\s+down\s+bolts?|"
                    r"hollow\s+section|fin\s*plate|end\s*plate|cleats?|steel\s+(?:beam|column|stair\w*|platform|frame|member|"
                    r"post|section|channel|angle|structure))\b", re.I)
RC_TITLE = re.compile(r"\bRC\b|reinforc|post[- ]?tension|\bpile|pilecap|\bslab|concrete|beam\s+schedule|column\s+schedule|"
                      r"loading\s+plan|\blift\s+\d|dock\s+leveller|waterproof", re.I)
LEVEL_VIEW = re.compile(r"elevation|\bsection", re.I)       # where the levels are read
# a structure drawn on its own sheets is its own unit (name, title words); the first that matches the title wins
UNIT_NOUNS = [("Link Bridge", r"link\s*bridge|\bbridge"), ("Walkway", r"walkway|linkway|covered\s+link"),
              ("Canopy", r"canopy"), ("Staircase", r"stair\w*"), ("Wall Cladding", r"cladd\w*"),
              ("Platform", r"platform"), ("Flag Pole", r"flag\s*pole"), ("Signage", r"signage"),
              ("Shelter", r"shelter"), ("Pipe Rack", r"pipe\s*rack"), ("Tower", r"\btower")]


def for_model(p):
    """the curator's rule: a sheet is for the model only when it shows steel members (member sizes, a steel subject in the title,
    steel members named on it). Architectural / M&E / RC / site sheets without steel are unrelated. A sheet whose text is
    not read yet (outlined, scanned) is kept."""
    if p.get("kind") != "vector" and not p.get("content_read"):
        return True
    t = p.get("title") or ""
    if p.get("disc") == "architectural":
        return bool(p["secs"]) or bool(STEEL_TITLE_ARCH.search(t))
    return bool(p["secs"]) or (bool(STEEL_TITLE.search(t)) and not re.search(r"\bRC\b", t)) or p.get("strong", 0) >= 3


def body_chars(page):
    """Characters of real text in the drawing area (outside a right-hand title column and a bottom title strip)."""
    W, H = page.rect.width, page.rect.height
    return sum(len(w[4]) for w in page.get_text("words") if w[2] < 0.8 * W and w[3] < 0.88 * H)


OCR_DPI = 150          # the same page image as vectorize_drawings.vectorize_pages: its OCR cache is shared


def read_outlined_content(per_pdf, todo, ev=lambda *a: None, batch=12):
    """OCR the whole page of every outlined sheet (cached, parallel, a batch of pages in memory at a time) and add what
    it reads to the sheet's content: member sizes, steel words, grid labels. The vectorize step reuses the same OCR."""
    import vectorize_drawings as V
    docs = {mine[0]["pdf"]: doc for doc, mine in per_pdf if mine}
    todo = [p for p in todo if p["pdf"] in docs]
    done = 0
    for b in range(0, len(todo), batch):
        chunk, jobs = todo[b:b + batch], []
        for n, p in enumerate(chunk):
            page = docs[p["pdf"]][p["i"]]
            page.remove_rotation()
            pm = page.get_pixmap(dpi=OCR_DPI)
            jobs.append((n, pm.samples, pm.w, pm.h, pm.n, 72.0 / OCR_DPI, 0.6))
        if b == 0:
            c = V.cached_count(V._ocr_job, jobs)
            print(f"  * {len(todo)} sheets have the drawing drawn as outlines -> OCR to read what they show"
                  + (f" ({c} of the first {len(jobs)} already read on an earlier run)" if c else "") + ".", flush=True)
        for n, items in V.run_parallel(V._ocr_job, jobs):
            p = chunk[n]
            t = plain(" ".join(x[0] for x in items))
            p["content"] = (p.get("content") or "") + "\n" + t
            p["ocr_items"] = [(x[0], tuple(x[1])) for x in items]
            p["secs"] |= {x.group(0).replace(" ", "").upper() for x in SECTION.finditer(t)}
            p["strong"] = p.get("strong", 0) + len(STRONG.findall(t))
            p["steely"] = p["steely"] or bool(STEELY.search(t))
            p["steelwork"] = p["steelwork"] or bool(STEEL_WORK.search(t))
            p["content_read"] = True
            done += 1
            ev("PROG", done / len(todo), f"sheet {done}/{len(todo)}")


def grid_labels(page, p):
    """Grid line labels on the sheet: short labels (A, B, 1, 12, A1, C') inside a circle; real text and OCR both."""
    try:
        circ = [d["rect"] for d in page.get_drawings()
                if 4 < d["rect"].width < 45 and abs(d["rect"].width - d["rect"].height) < 0.15 * d["rect"].width
                and sum(1 for it in d["items"] if it[0] == "c") >= 4]
    except Exception:
        return set()
    if not circ:
        return set()
    spans = [(plain(sp["text"]).strip(), fitz.Rect(sp["bbox"])) for b in page.get_text("dict")["blocks"]
             for ln in b.get("lines", []) for sp in ln["spans"]]
    spans += [(t.strip(), fitz.Rect(r)) for t, r in p.get("ocr_items") or []]
    out = set()
    for t, c in spans:
        if re.fullmatch(r"[A-Z]{1,2}'?|\d{1,2}[A-Z]?'?|[A-Z]\d{1,2}'?", t):
            m = fitz.Point((c.x0 + c.x1) / 2, (c.y0 + c.y1) / 2)
            if any(r.contains(m) and r.width < 6 * max(c.height, 1) for r in circ):
                out.add(t)
    return out


def grid_name(g):
    """'A-K / 1-14' for a set of grid labels."""
    let = sorted(x for x in g if x[0].isalpha())
    num = sorted((x for x in g if x[0].isdigit()), key=lambda x: (int(re.match(r"\d+", x).group(0)), x))
    return " / ".join(f"{v[0]}-{v[-1]}" if len(v) > 1 else v[0] for v in (let, num) if v)


def split_by_grid(name, ps):
    """the curator: a different set of grid lines is a different unit. The letter axis and the number axis are compared apart
    (a plan may show one axis only: Job A platform plans show 1-4A or A-G1): two sheets are on different grids only when
    they share an axis and less than half of the smaller set of that axis matches. A cluster of one sheet is not a unit
    (a truss elevation showing four grids of its building); it and the sheets without grids join the biggest cluster."""
    def axes(g):
        return {"let": {x for x in g if x[0].isalpha()}, "num": {x for x in g if x[0].isdigit()}}

    def same(a, b):
        shared = [k for k in ("let", "num") if len(a[k]) >= 2 and len(b[k]) >= 2]
        return all(len(a[k] & b[k]) >= 0.5 * min(len(a[k]), len(b[k])) for k in shared)

    clusters = []
    for p in sorted((p for p in ps if len(p.get("grids") or ()) >= 4), key=lambda p: -len(p["grids"])):
        a = axes(p["grids"])
        c = next((c for c in clusters if same(a, c["a"])), None)
        if c:
            c["a"]["let"] |= a["let"]; c["a"]["num"] |= a["num"]; c["ps"].append(p)
        else:
            clusters.append({"a": a, "ps": [p]})
    clusters.sort(key=lambda c: -len(c["ps"]))
    units = [c for c in clusters if len(c["ps"]) >= 2]
    if len(units) < 2:
        return {name: ps}
    inunit = {id(p) for c in units[1:] for p in c["ps"]}
    units[0]["ps"] = [p for p in ps if id(p) not in inunit]
    return {f"{name} (grid {grid_name(c['a']['let'] | c['a']['num'])})": c["ps"] for c in units}


def subject(p):
    """The structure a sheet draws, from its content (view captions, notes) with the title as one strong vote:
    'Canopy' when canopy is most of what it names, '' for the building itself."""
    votes = Counter()
    body, title = p.get("content") or "", p.get("title") or ""
    for n, rx in UNIT_NOUNS:
        votes[n] = len(re.findall(rx, body, re.I)) + 3 * len(re.findall(rx, title, re.I))
    tot = sum(votes.values())
    if not tot:
        return ""
    n, v = votes.most_common(1)[0]
    return n if v >= 3 and v >= 0.6 * tot else ""


def series(num):
    return re.sub(r"\d+$", "", num or "")


def model_units(base, ps):
    """Split one group into the curator's units -> ({unit name: [sheets]}, [sheets not for the model]).
    - sheets with no steel (for_model) leave; a structural sheet with no RC word in its title is kept when another sheet
      of its number series has steel (its members drawn as outlines: CONNECTION DETAILS beside ROOF TRUSS SECTION ...);
    - a structure drawn on its own sheets (canopy, staircase, link bridge, cladding ...) is its own unit;
    - a unit with no steel elevation or section gets the architectural elevations / sections (the levels)."""
    keep = [p for p in ps if for_model(p)]
    good = {series(p.get("num")) for p in keep if p.get("disc") != "architectural" and p.get("num")}
    keep += [p for p in ps if p not in keep and p.get("disc") in ("structural", "foundation") and p.get("num")
             and series(p["num"]) in good and not RC_TITLE.search(p.get("title") or "")]
    drop = [p for p in ps if p not in keep]
    units = defaultdict(list)
    for p in keep:
        noun = subject(p)
        units[noun if noun and noun.lower() not in base.lower() else ""].append(p)
    if len(units) == 1:
        units = {"": keep}
    # a view with levels: an elevation or a section, not a member / connection "SECTION DETAILS" sheet
    level_view = lambda t: bool(re.search(r"elevation", t, re.I) or re.search(r"\bsection\b(?!\s*details?)", t, re.I))
    named = {}
    for k, v in units.items():
        named.update(split_by_grid(f"{base} {k}" if k else base, v))
    views = [p for p in drop if p.get("disc") == "architectural" and LEVEL_VIEW.search(p.get("title") or "")
             and not re.search(r"detail|schedule", p.get("title") or "", re.I)]
    used = set()
    for name, u in named.items():
        if views and not any(level_view(p.get("title") or "") for p in u):
            u += views
            used |= {id(p) for p in views}
            for p in views:
                p["levels_only"] = True
    return named, [p for p in drop if id(p) not in used]
STATUS = ("FOR CONSTRUCTION", "FOR APPROVAL", "FOR INFORMATION", "AS BUILT", "FOR TENDER", "PRELIMINARY")
_ocr = None


def ocr_lines(page, clip):
    """[(text, x, y)] of the clip, OCR at 150 dpi (rapidocr). [] when OCR is not installed."""
    global _ocr
    try:
        import numpy as np
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return []
    _ocr = _ocr or RapidOCR()
    pm = page.get_pixmap(clip=clip, dpi=int(min(450, max(150, 150 * 2384 / page.rect.width))))   # A4 copies of A1 sheets: more dpi
    img = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.h, pm.w, pm.n)[:, :, :3]
    res, _ = _ocr(img)
    return [(r[1], r[0][0][0], r[0][0][1]) for r in (res or [])]


_PUA = {c: c - 0xF000 for c in range(0xF020, 0xF0FF)}
# copyright / proprietary notices, never a drawing title
NOTICE = re.compile(r"PROPRIETARY|COPYRIGHT|REPRODUC|AUTHORI[SZ]ATION|PERMISSION|CONFIDENTIAL|INTELLECTUAL|DO\s+NOT\s+SCALE", re.I)


def plain(s):
    """Text in a symbol-encoded font back to letters: some CAD exports store 'TITLE' as U+F054 U+F049 U+F054 U+F04C U+F045
    (each character shifted by 0xF000; M.E.I. sheets, the labels and half the title block). Without this the labels,
    the member sizes (UB, SHS) and the steel words are not seen."""
    return s.translate(_PUA) if s and any("" <= ch < "" for ch in s) else s


def page_text(page):
    return plain(page.get_text())


def text_lines(page, clip):
    """[(text, x, y)] of real text in the clip, top to bottom."""
    out = []
    for b in page.get_text("dict", clip=clip)["blocks"]:
        for l in b.get("lines", []):
            t = plain(" ".join(s["text"] for s in l["spans"])).strip()
            if t:
                out.append((t, l["bbox"][0], l["bbox"][1]))
    return sorted(out, key=lambda r: (round(r[2] / 4), r[1]))


def titleblock_rect(page, drawings=None):
    """Right-hand title block column: from the leftmost full-height vertical line in the right 30 % to the page edge
    (page coordinates, rotation removed by the caller). drawings = page.get_drawings() when the caller has it already."""
    W, H = page.rect.width, page.rect.height
    xs = []
    for dr in drawings if drawings is not None else page.get_drawings():
        for it in dr["items"]:
            if it[0] == "l" and abs(it[1].x - it[2].x) < 0.5 and abs(it[1].y - it[2].y) > 0.85 * H and 0.7 * W < it[1].x < W - 40:
                xs.append(it[1].x)          # near-full-height line only: schedule / legend table lines are shorter
            elif it[0] == "re" and it[1].height > 0.85 * H and 0.7 * W < it[1].x0 < W - 40:
                xs.append(it[1].x0)
    return fitz.Rect(min(xs) if xs else 0.84 * W, 0, W, H)


def tb_has_text(lines):
    """The title block has real text: its label, or enough characters that OCR would only read the same again.
    (A firm whose label is not 'DRAWING TITLE' sent all 268 vector title blocks to OCR, 37 min for nothing.)"""
    return any(re.search(TITLE_LABEL, t, re.I) for t, _, _ in lines) or sum(len(t.strip()) for t, _, _ in lines) >= 150


def titleblock_ocr_clip(page, tb=None):
    """(clip, dpi) the title block OCR reads, or None when the title block has real text (no OCR needed)."""
    page.remove_rotation()
    tb = tb or titleblock_rect(page)
    if tb_has_text(text_lines(page, tb)):
        return None
    return fitz.Rect(tb.x0, page.rect.height * 0.5, tb.x1, tb.y1), int(min(450, max(150, 150 * 2384 / page.rect.width)))


def prefetch_titleblock_ocr(items, tbs, check=None, planned=None, progress=None):
    """OCR every title block whose text is outlined, several worker processes in parallel.
    items: [(key, page)]; tbs: {key: title block rect} (filled here, reused by read_titleblock).
    check(d, n) while looking which title blocks need OCR, planned(n) once known, progress(d, n) per OCR'd one.
    -> {key: [(text, x, y)]} (the same lines ocr_lines returns)."""
    try:
        import vectorize_drawings as V
        import rapidocr_onnxruntime  # noqa: F401
    except ImportError:
        V = None
    jobs = []
    for d, (key, page) in enumerate(items, 1):
        page.remove_rotation()
        tbs[key] = titleblock_rect(page)
        c = titleblock_ocr_clip(page, tbs[key]) if V else None
        if c:
            pm = page.get_pixmap(clip=c[0], dpi=c[1])
            jobs.append((key, pm.samples, pm.w, pm.h, pm.n))
        if check:
            check(d, len(items))
    if planned:
        planned(len(jobs), V.cached_count(V.ocr_raw_job, jobs) if jobs else 0)
    out = {}
    for key, lines in (V.run_parallel(V.ocr_raw_job, jobs) if jobs else []):
        out[key] = lines
        if progress:
            progress(len(out), len(jobs))
    return out


def read_titleblock(page, use_ocr, pre_ocr=None, tb=None):
    """{'title','number','rev','sheet','status','project','ocr'} from the title block (text first, OCR if outlined).
    pre_ocr: the OCR lines already read by prefetch_titleblock_ocr (None = OCR here if needed); tb: its rect if known."""
    page.remove_rotation()
    tb = tb or titleblock_rect(page)
    lines = text_lines(page, tb)
    used_ocr = False
    if not tb_has_text(lines) and use_ocr:
        lines = pre_ocr if pre_ocr is not None else ocr_lines(page, fitz.Rect(tb.x0, page.rect.height * 0.5, tb.x1, tb.y1))
        lines = sorted(lines, key=lambda r: (round(r[2] / 6), r[1]))
        used_ocr = bool(lines)
    LABEL = re.compile(TITLE_LABEL + "|" + NO_LABEL + r"|JOB\s*NO|\bSCALE\b|^REV|^REVISION|^SHEET|^SHT|PROJECT|DATE|DRAWN|CHECKED", re.I)

    def below(pat, n, xtol, ytol=80, keep=False):
        """Values written under the label matching pat, in its own column."""
        for lab in [(t, x, y) for t, x, y in lines if re.search(pat, t, re.I)]:     # every instance of the label
            m = re.search(pat, lab[0], re.I)
            same = lab[0][m.end():].strip(" :.-#") if m else ""                      # "DRAWING NO: S-101" on one line
            if same and not LABEL.search(same):
                return [same]
            col = [(y, t) for t, x, y in lines if lab[2] + 2 < y < lab[2] + ytol and lab[1] - 30 < x < lab[1] + xtol]
            stop = min([y for y, t in col if LABEL.search(t)] + [lab[2] + ytol])      # the next label in this column
            c = [(y, t) for y, t in col if y < stop and not NOTICE.search(t) and (keep or (not DWG.fullmatch(t.strip()) and not re.fullmatch(r"\s*\d+\s*/\s*\d+\s*", t)))]
            if c:
                return [t for _, t in sorted(c)][:n]
        return []

    up = [t.upper() for t, _, _ in lines]
    info = {"ocr": used_ocr, "title": "", "number": "", "rev": "", "sheet": "", "status": "", "project": ""}
    tl = [t.strip() for t in below(TITLE_LABEL, 5, 260) if t.strip()]
    info["title"] = re.sub(r"^\d{4,6}\s+", "", " ".join(tl)).strip(" :-")
    info["title_lines"] = tl                 # the title's own line breaks, kept on the lean copy
    for t in [t for t, x, y in lines if DWG.fullmatch(t.strip())] + below(NO_LABEL, 2, 260):
        m = DWG.search(t)
        if m:
            info["number"] = m.group(0)
            break
    if not info["number"]:      # any other firm's numbering (XXXX-SEC-LA1-00-SSTEL-001 ...): read what sits under the DRAWING NO. label
        at_label = below(NO_LABEL, 3, 300)
        for t in at_label + [t for t, x, y in lines]:
            m = DWG_LOOSE.search(t.strip().upper())
            if m and re.search(r"\d", m.group(0)):
                info["number"] = m.group(0)
                break
        else:                   # a short number right at its label (S-101, A/201, 1234): one code-like token with a digit
            for t in at_label:
                m = re.fullmatch(r"\s*([A-Z0-9][A-Z0-9._/-]{1,30}[A-Z0-9])\s*", t.upper())
                if m and re.search(r"\d", m.group(1)) and not re.fullmatch(r"\d{1,3}/\d{1,3}", m.group(1)):
                    info["number"] = m.group(1)
                    break
    rv = [t.strip() for t in below(r"^\s*REV(ISION)?\b", 2, 70, 45, keep=True) if re.fullmatch(r"\s*[A-Z]{1,2}\d?\s*|\s*\d{1,2}\s*", t) and t.strip() != "AS"]
    info["rev"] = rv[0] if rv else ""
    scale = next(((x, y) for t, x, y in lines if re.search(r"\bSCALE\b", t, re.I)), (-1e9, -1e9))
    for t in below(r"^\s*(SHEET|SHT)\b", 2, 70, 45, keep=True) + [t for t, x, y in lines if abs(x - scale[0]) > 60 or abs(y - scale[1]) > 40]:
        sm = re.fullmatch(r"\s*(\d{1,3})\s*(?:/|OF)\s*(\d{1,3})\s*", t, re.I)
        if sm and 0 < int(sm.group(1)) <= int(sm.group(2)) <= 99:
            info["sheet"] = f"{int(sm.group(1))}/{int(sm.group(2))}"
            break
    info["project"] = " ".join(below(r"PROJECT\s*(TITLE|/AREA)", 2, 300))
    present = [st for st in STATUS if any(st in t for t in up)]
    info["status"] = present[0] if len(present) == 1 else ""      # a status box lists every option: only a lone one is read
    if not info["title"] and info["number"]:
        # the labels are outlines, the values real text (M.E.I. sheets): the title is the short lines in the cell just above
        # the number row; disclaimer sentences and stray glyphs are not
        ny = min([y for t, x, y in lines if info["number"] in t.upper().replace(" ", "_") or info["number"] in t.upper()] or [0])
        top = min([y for t, x, y in lines if ny - 60 < y < ny and len(t.split()) > 8] + [ny])   # top of the bottom strip
        cand = sorted(((y, x, t.strip()) for t, x, y in lines if top - 160 < y < top - 4
                       and len(re.sub(r"[^A-Za-z0-9]", "", t)) >= 3 and not LABEL.search(t) and not NOTICE.search(t)), reverse=True)
        tl, last, x0 = [], top, None
        for y, x, t in cand:     # the block right above the strip, its lines starting in line (the address above sits indented)
            if x0 is not None and abs(x - x0) > 15:
                continue
            if last - y > (120 if not tl else 40) or len(t.split()) > 10:
                break
            tl.insert(0, t); last, x0 = y, x
        if tl:
            info["title"], info["title_lines"] = " ".join(tl).strip(" :-"), tl
    # where the title cell starts (the lean copy keeps it from there down): just above its TITLE label when that is text,
    # else just under the last line above the title (the address block), at most 30 pt over the title's first line
    ty = [y for t, x, y in lines if info["title_lines"] and t.strip() == info["title_lines"][0]]
    cut = None
    if ty:
        y0 = min(ty)
        lab = [y for t, x, y in lines if y0 - 40 < y < y0 and re.search(TITLE_LABEL, t, re.I)]
        above = [y for t, x, y in lines if y < (min(lab) if lab else y0) - 2]
        cut = (min(lab) - 3) if lab else max(y0 - 30, (max(above) + 9) if above else 0)
    info["title_y"] = cut
    return info


def base_number(n):
    return re.sub(r"_[A-Z]{1,2}$", "", n.replace("JOBA-10-", "JOBA-10-")) if n else n


def drawing_list(doc):
    """Rows of a drawing-list page: [(dwg_no, title, rev, sheets)] in list order, and the list's first line."""
    rows, head = [], ""
    for page in doc:
        words = page.get_text("words")
        if len(words) > 2500:
            continue
        lines = defaultdict(list)
        for w in words:
            lines[round((w[1] + w[3]) / 2 / 4)].append(w)
        found, first = [], ""
        for k in sorted(lines):
            line = plain(" ".join(w[4] for w in sorted(lines[k], key=lambda w: w[0])))
            first = first or line
            m = re.match(r"^\s*(\d{1,3})\s+(\S+)\s+(.+?)\s+([A-Z]{1,2}\d?|\d{1,2})\s+(\d{1,3})\s*$", line)
            if m and DWG_LOOSE.match(m.group(2)):
                found.append((base_number(m.group(2)), m.group(3).strip(), m.group(4), int(m.group(5))))
        if len(found) >= 3:
            rows += found
            head = head or first
    return rows, head


def building_of(title):
    t = re.sub(r"[_]+", " ", title)
    t = re.sub(r"^\s*(civil\s*(&|and)\s*structural|structural|civil|architectural)\s+", "", t, flags=re.I)   # discipline prefix is not the building
    t = re.sub(r"^\s*overall\s+site\s+", "", t, flags=re.I)

    def cut(s, lead_empty=False):
        s = re.split(r"\s[-–—]\s?|\s?[-–—]\s|(?<=[a-z])-(?=[A-Z])", s)[0]
        m = re.search(r"\b" + DISC_WORDS + r"\b", s, re.I)
        if m:
            s = s[:m.start()] if m.start() > 3 else ("" if lead_empty else s)
        s = re.sub(r"[^A-Za-z0-9 &]+", " ", s).strip()
        return re.sub(r"\s+", " ", s)

    # "<host> BUILDING [LEVEL x] <structure>": a bridge / support / rack ... on a building is its own structure
    # ("FAB/CUP BUILDING UTILITY BRIDGE 3 SECTION" -> Utility Bridge 3, "CUP BUILDING LEVEL R1 BUS BAR DUCT SUPPORT ..." -> Bus Bar Duct Support)
    m = re.match(r"(.*?\bbuilding)\s+(?:(?:level|lvl|floor)\s+\S+\s+)?(.+)$", t, re.I)
    if m:
        rest = cut(m.group(2), lead_empty=True)
        if re.search(r"\b(bridge|support|rack|walkway|access|canopy|shelter|link|tower|pipe)\b", rest, re.I):
            return rest.title()
        return cut(m.group(1)).title()
    t = re.sub(r"\s+(level|lvl|floor)\s+\S+.*$", "", t, flags=re.I)
    return cut(t).title()


def discipline(title, number):
    tl = title.lower()
    if re.search(r"steel|truss|structur|framing|purlin|lintel beam", tl):
        return "structural"
    if re.search(r"architect", tl):
        return "architectural"
    if re.search(r"foundation|piling|pile", tl):
        return "foundation"
    code = re.search(r"-([A-Z]{2})-\d{4,6}", number or "")
    return {"ST": "structural", "GA": "architectural", "FD": "foundation"}.get(code.group(1) if code else "", "other")


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "unnamed"


ISO_TYPES = {"DR", "DW", "DG"}                   # ISO 19650 "type" token: drawing
ISO_ROLE = {"S": "structural", "A": "architectural"}
BUILDING_NOUN = re.compile(r"\b(warehouse|office\s+block|office\s+building|guard\s*house|sub\s*station|pump\s*house|refuse\s+chamber|"
                           r"recycl\w*\s+(?:area|centre|center)|factory|plant\s+room|workshop|car\s*park|canopy|shelter|hostel|"
                           r"school|hospital|mall|tower|terminal|hangar|shed)\b", re.I)
ISO_ALL_ZONES = {"ZZ", "XX", "00", "000"}        # ZZ = all / several zones, XX / 000 = none: general sheets
# a discipline token in a Zone-Discipline-Type-Level-Number scheme: optional digit + the discipline code
DISC_TOKEN = re.compile(r"\d?(ST|SE|CS|AR|CV|CI|ME|MC|EL|PL|FP|LS|ID|IN|FD)")
ISO_SITE_ZONES = {"SIT", "SITE", "EXT", "GEN"}   # whole-site sheets (overall loading plans ...): not one building


def iso_parts(num):
    """ISO 19650 number (Project-Originator-Zone-Level-Type-Role-Number, project/originator optional):
    SJI-ARP-ES-01-DR-S-2221 / HS-XX-DR-A-2001 -> (zone, level, role) or None."""
    t = re.split(r"[-_\s]+", (num or "").upper())
    i = next((k for k, x in enumerate(t) if x in ISO_TYPES), None)
    if i is not None and i >= 2 and i + 1 < len(t):
        return t[i - 2], t[i - 1], t[i + 1]
    # Zone-Discipline-Type-Level-Number, project prefix optional (M.E.I.: EQB_ES103 II_WH4_0AR_CFL_01_002,
    # 000_0ST_XGN_00_004): the discipline token (0AR, 0ST, CS, ...) names the zone before it
    k = next((k for k, x in enumerate(t) if k >= 1 and k + 2 < len(t) and DISC_TOKEN.fullmatch(x)), None)
    if k is None:
        return None
    role = DISC_TOKEN.fullmatch(t[k]).group(1)
    return t[k - 1], t[k + 2], {"ST": "S", "SE": "S", "CS": "S", "AR": "A"}.get(role, role)


def iso_groups(pages):
    """Group by the ISO 19650 zone code when most numbers carry one (one zone = one stand-alone building).
    Returns ({name: [pages]}, [unrelated pages], [pages left for the title rules]) or None.
    - ZZ / XX zones (general notes, typical / standard details, overall site plans) -> unrelated.
    - Zones whose title name is the same building (Education Building ES / HS / EDU): the zone whose titles say
      "Overall" (or that never names its own code) is shared: its sheets go into each of the others.
    - A zone without one structural (role S) or steel sheet -> unrelated (architect's sheets of an RC building)."""
    numbered = [p for p in pages if not p["list"] and p.get("num")]
    parts = {id(p): iso_parts(p["num"]) for p in numbered}
    if not numbered or sum(1 for v in parts.values() if v) < 0.6 * len(numbered):
        return None
    zones, unrelated, rest = defaultdict(list), [], [p for p in pages if not p["list"] and not p.get("num")]
    for p in numbered:
        v = parts[id(p)]
        if not v:
            rest.append(p); continue
        zone, _, role = v
        p["disc"] = ISO_ROLE.get(role[:1], "other")
        if p["disc"] == "structural" and re.search(r"foundation|piling|pile cap", p.get("title") or "", re.I):
            p["disc"] = "foundation"
        p["zone"] = zone
        (unrelated if zone in ISO_ALL_ZONES or zone in ISO_SITE_ZONES else zones[zone]).append(p)

    def base_name(ps):
        votes = Counter()
        for p in sorted(ps, key=lambda p: p["disc"] != "structural"):
            t = re.sub(r"\([^)]*\)", " ", p.get("title") or "")
            t = re.sub(r"^\s*(general\s+arrangement|steel\s*works?|loading\s+plan|typical|standard)\b[\s:-]*", "", t, flags=re.I)
            t = re.sub(r"\b(basement\s*\d*|b\d|\d+\s*(st|nd|rd|th)\s+storey|lower\s+roof|upper\s+roof|roof|storey)\b.*$", "", t, flags=re.I)
            b = building_of(t)
            if b:
                votes[b] += 3 if p["disc"] == "structural" else 1
        # a zone of one big package (M.E.I. WH4: 254 sheets) is named by its titles' topic words ("Ground" from
        # GROUND FLOOR PLAN ...): a building word most titles share wins, a weak vote leaves the zone code
        nouns = Counter(m.group(1).lower() for p in ps for m in [BUILDING_NOUN.search(p.get("title") or "")] if m)
        if nouns and nouns.most_common(1)[0][1] >= max(3, 0.1 * len(ps)):
            return nouns.most_common(1)[0][0].title()
        if votes and votes.most_common(1)[0][1] >= 0.3 * sum(3 if p["disc"] == "structural" else 1 for p in ps):
            return votes.most_common(1)[0][0]
        return ""

    base = {z: base_name(ps) or z for z, ps in zones.items()}
    for z in [z for z, b in base.items() if re.fullmatch(r"(overall|site|general|whole site)( plan)?", b.strip(), re.I)]:
        unrelated += zones.pop(z); base.pop(z)          # a zone that is the whole site, not a building
    names, shared_into = {}, defaultdict(list)
    by_base = defaultdict(list)
    for z, b in base.items():
        by_base[b.lower()].append(z)
    for b, zs in by_base.items():
        if len(zs) == 1:
            z = zs[0]
            names[z] = base[z] if base[z].lower() != z.lower() else z
            continue
        titles = {z: " ".join(p.get("title") or "" for p in zones[z]).upper() for z in zs}
        shared = [z for z in zs if "OVERALL" in titles[z]] or \
                 [z for z in zs if z not in re.findall(r"\b[A-Z0-9]{2,4}\b", titles[z]) and
                  all(o in re.findall(r"\b[A-Z0-9]{2,4}\b", titles[o]) for o in zs if o != z)]
        own = [z for z in zs if z not in shared] or zs
        for z in own:
            names[z] = f"{base[z]} {z}"
        for z in shared if own != zs else []:
            shared_into[z] = own
    groups = defaultdict(list)
    for z, ps in zones.items():
        if z in shared_into:
            continue
        groups[names[z]] += ps
    for z, targets in shared_into.items():
        for t in targets:
            groups[names[t]] += zones[z]
        for p in zones[z]:
            p["building"] = "; ".join(names[t] for t in targets)
    for name, ps in list(groups.items()):
        if not any(p["disc"] == "structural" or p["steely"] for p in ps) or no_steel(ps):
            unrelated += [p for p in ps if p not in unrelated]
            del groups[name]
            continue
        for p in ps:
            p.setdefault("building", name)
    return groups, unrelated, rest


def same_building(a, b):
    wa, wb = set(a.lower().split()), set(b.lower().split())
    return bool(wa) and bool(wb) and (wa <= wb or wb <= wa)


def safe_folder(s):
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', " ", s).strip(" .")
    return re.sub(r"\s+", " ", s)[:80] or "Drawing package"


def titleblock_inner(page, tb, inset=1.5, drawings=None):
    """Inside of the title block column, between its left line and the drawing border (right / top / bottom).
    The border is the near-full-height vertical lines / rectangles in the column; falls back to tb."""
    H = page.rect.height
    xs, ys = [], []
    for dr in drawings if drawings is not None else page.get_drawings():
        for it in dr["items"]:
            if it[0] == "l" and abs(it[1].x - it[2].x) < 0.5 and abs(it[1].y - it[2].y) > 0.85 * H and it[1].x >= tb.x0 - 1:
                xs.append(it[1].x); ys += [min(it[1].y, it[2].y), max(it[1].y, it[2].y)]
            elif it[0] == "re" and it[1].height > 0.85 * H and it[1].x1 > tb.x0 + 20:
                xs += [x for x in (it[1].x0, it[1].x1) if x >= tb.x0 - 1]; ys += [it[1].y0, it[1].y1]
    right = max([x for x in xs if x > tb.x0 + 20], default=tb.x1)
    top, bot = (min(ys), max(ys)) if ys else (tb.y0, tb.y1)
    return fitz.Rect(tb.x0 + inset, top + inset, right - inset, bot - inset)


LEAN_KEYS = ("pdf", "i", "title", "title_lines", "title_y", "num", "rev", "sheet", "status")
FIXED_PAGES = []        # ((pdf, i), how) for every page that could not be copied as it is


def _raster_copy(dst, src_page, dpi=200):
    """Last resort for a page MuPDF cannot copy: the page as an image, its text kept on top (invisible, searchable)."""
    pg = dst.new_page(width=src_page.rect.width, height=src_page.rect.height)
    pg.insert_image(pg.rect, pixmap=src_page.get_pixmap(dpi=dpi))
    for b in src_page.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            for s in ln["spans"]:
                if s["text"].strip():
                    try:
                        pg.insert_text(s["origin"], s["text"], fontsize=max(s["size"], 1), render_mode=3)
                    except Exception:
                        pass
    return "raster"


def copy_page(dst, path, i):
    """Append page i of path to dst. A source with broken object references (MuPDF 'source object number out of
    range', seen on a 112-file package) is repaired first, else copied as an image + its text. Returns how."""
    try:
        dst.insert_pdf(fitz.open(path), from_page=i, to_page=i)
        return "copy"
    except Exception:
        pass
    src = fitz.open(path)
    try:
        one = fitz.open()
        one.insert_pdf(fitz.open("pdf", src.tobytes(garbage=4, clean=True)), from_page=i, to_page=i)
        dst.insert_pdf(one)
        return "repaired"
    except Exception:
        pass
    try:
        one = fitz.open("pdf", src.convert_to_pdf(from_page=i, to_page=i))
        dst.insert_pdf(one)
        return "repaired"
    except Exception:
        pass
    return _raster_copy(dst, src[i])


def lean_page(p):
    """One lean page as PDF bytes (worker-process friendly: p holds only LEAN_KEYS). Returns (key, bytes, how)."""
    dst = fitz.open()
    how = copy_page(dst, p["pdf"], p["i"])
    try:
        _lean_page(dst[-1], p)
    except Exception:
        how += ", not leaned"
    return (p["pdf"], p["i"]), dst.tobytes(garbage=3, deflate=True), how


def lean_pages_parallel(pages, progress=None):
    """{(pdf, i): lean page bytes} for every page, several processes at once (lean pages are independent)."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    jobs = list({(p["pdf"], p["i"]): {k: p.get(k) for k in LEAN_KEYS} for p in pages}.values())
    out = {}
    workers = max(1, min(len(jobs), (os.cpu_count() or 4) - 2, 12))
    if workers == 1:
        results = (lean_page(j) for j in jobs)
    else:
        pool = ProcessPoolExecutor(workers)
        results = (f.result() for f in as_completed([pool.submit(lean_page, j) for j in jobs]))
    for key, data, how in results:
        out[key] = data
        if how != "copy":
            FIXED_PAGES.append((key, how))
        if progress:
            progress(len(out), len(jobs))
    if workers > 1:
        pool.shutdown()
    return out


def lean_copy(src_pages, path, done=None):
    """Reading copy: title block blanked (text, lines, images) and replaced by drawing title / number / rev / sheet.
    done = {(pdf, i): lean page bytes} from lean_pages_parallel (else each page is made here)."""
    dst = fitz.open()
    for p in src_pages:
        data = (done or {}).get((p["pdf"], p["i"]))
        if not data:
            key, data, how = lean_page({k: p.get(k) for k in LEAN_KEYS})
            if how != "copy":
                FIXED_PAGES.append((key, how))
        dst.insert_pdf(fitz.open("pdf", data))
    dst.save(path, garbage=4, deflate=True)


def _lean_page(pg, p):
    """Blank the inside of the title block column and write title / number / rev / sheet / status into it."""
    pg.remove_rotation()
    dr = pg.get_drawings()                  # one scan of the vector paths for both the column and its frame
    box = titleblock_inner(pg, titleblock_rect(pg, dr), drawings=dr)
    H = pg.rect.height
    ty = p.get("title_y")
    if ty and box.y0 + 0.4 * H < ty < box.y1:
        # the curator's style (sample version after cleaning.pdf, M.E.I. sheet): blank the column down to the title cell; the title
        # cell and the strip under it (project no., drawing no., rev, scale) stay as drawn, nothing is retyped.
        # Removed for real (text, lines, images), not covered by a white box: the reader never sees the hidden text.
        pg.add_redact_annot(fitz.Rect(box.x0, box.y0, box.x1, ty), fill=(1, 1, 1))
        # the copyright / proprietary notice in the kept strip goes too (the sample blanks that box): its lines are found by
        # their words (or a sentence of more than 8 words), then the paragraph is followed down line by line, same left edge
        notes = fitz.Rect()
        lns = sorted(((fitz.Rect(ln["bbox"]), plain(" ".join(s["text"] for s in ln["spans"])))
                      for b in pg.get_text("dict", clip=fitz.Rect(box.x0, ty, box.x1, box.y1))["blocks"] for ln in b.get("lines", [])),
                     key=lambda x: x[0].y0)
        for r, t in lns:
            if NOTICE.search(t) or len(t.split()) > 8:
                notes |= r
                break
        grew = not notes.is_empty
        while grew:
            grew = False
            for r, t in lns:
                if not (r in notes) and abs(r.x0 - notes.x0) < 4 and notes.y1 - 3 < r.y0 < notes.y1 + 12:
                    notes |= r; grew = True
        if not notes.is_empty:
            pg.add_redact_annot(notes + (-1, -1, 1, 1), fill=(1, 1, 1))
        pg.apply_redactions(images=fitz.PDF_REDACT_IMAGE_REMOVE, graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED)
        return
    # title cell not found: blank the whole INSIDE of the title block column (its left line and the drawing border stay,
    # so the sheet keeps a closed frame; a test job 07_substation-building (lean).pdf) and write title / number / rev into it
    pg.add_redact_annot(box, fill=(1, 1, 1))
    pg.apply_redactions(images=fitz.PDF_REDACT_IMAGE_REMOVE, graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED)
    tl = p.get("title_lines") or []
    title = "\n".join(tl) if tl and " ".join(tl).strip(" :-").endswith((p.get("title") or "")) else (p.get("title") or "")
    txt = (f"DRAWING TITLE:\n{title}\n\nDRAWING NO.:\n{p.get('num') or ''}\n\nREV: {p.get('rev') or '-'}"
           f"   SHEET: {p.get('sheet') or '-'}\nSTATUS: {p.get('status') or '-'}")
    H = pg.rect.height       # 12 pt, wrapped inside the column, in the lower part of the column
    pg.insert_textbox(fitz.Rect(box.x0 + 8, box.y1 - 0.17 * H, box.x1 - 8, box.y1 - 8), txt, fontsize=12, fontname="helv")


def project_root(folder):
    """The project folder, whichever level was picked: <project>, <project>/Drawings or <project>/Drawings/Original Drawings."""
    f = Path(folder).resolve()
    if re.search(r"origin", f.name, re.I) and f.parent.name.lower() == "drawings":
        return f.parent.parent
    if f.name.lower() == "drawings":
        return f.parent
    return f


def source_items(root):
    """What in the project folder is the curator's source drawings: every PDF and every folder holding a PDF (at any depth),
    in the project folder or in Drawings/, except what the split itself writes (Drawings/Original Drawings = already
    taken in, Drawings/Organized Drawings, Models, MTO). So a re-run after a crash takes nothing of its own output."""
    dwg = root / "Drawings"
    skip = {dwg, dwg / "Original Drawings", dwg / "Organized Drawings", root / "Models", root / "MTO"}
    old = (root / "manifest.json").exists()        # an old split output written straight into the folder: not a source
    out = []
    for base in (root, dwg):
        if not base.is_dir():
            continue
        for p in sorted(base.iterdir(), key=lambda x: x.name.lower()):
            if p in skip or p.name.startswith((".", "~$")):
                continue
            if p.is_file() and p.suffix.lower() == ".pdf" and not (old and base == root):
                out.append(p)
            elif p.is_dir() and not (old and p.name == "lean") and any(p.rglob("*.pdf")):
                out.append(p)                       # the whole folder, its subfolders kept as they are
            elif base == dwg and p.is_file():
                out.append(p)                       # anything loose in Drawings/ is a source file (as before)
    return out


def prepare_project(folder):
    """the curator drops his drawings into a new project folder: loose, in a 'Drawings' folder, or in any subfolders of
    his own. Make the job tree
        <project>/Drawings/Original Drawings   his drawings, moved here with their subfolders (moved, never copied over or deleted)
        <project>/Drawings/Organized Drawings  the split output
        <project>/Models/Models Data
        <project>/MTO
    and return (source folder, output folder). A name clash keeps both ("name (2).pdf", "folder (2)").
    Every PDF under Original Drawings (any depth) is split; a re-run only adds what is new."""
    root = project_root(folder)
    dwg = root / "Drawings"
    orig, org = dwg / "Original Drawings", dwg / "Organized Drawings"
    items = source_items(root)                     # before the tree is made, so its own folders are never taken
    for d in (orig, org, root / "Models" / "Models Data", root / "MTO"):
        d.mkdir(parents=True, exist_ok=True)

    def move(p):
        if p.is_dir() and p.name.lower() in ("original drawings", "drawings"):   # his own "Original Drawings" folder:
            for c in list(p.iterdir()):                                         # its contents, not a nested copy of it
                move(c)
            try:
                p.rmdir()
            except OSError:
                pass
            return
        dst = orig / p.name
        k = 2
        while dst.exists():
            dst = orig / (f"{p.stem} ({k}){p.suffix}" if p.is_file() else f"{p.name} ({k})")
            k += 1
        shutil.move(str(p), str(dst))

    moved = 0
    for p in items:
        move(p); moved += 1
    if moved:
        print(f"job tree ready: {moved} item(s) moved into {orig}")
    return orig, org


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--job")
    ap.add_argument("--project", help="project folder: make the job tree, move the drawings into Original Drawings, split")
    ap.add_argument("--progress", action="store_true", help="print @@STEP / @@PROG / @@PLAN lines (the Split Drawings window reads them)")
    ap.add_argument("--desktop", action="store_true")
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--no-lean", action="store_true")
    ap.add_argument("--no-vectorize", action="store_true")
    a = ap.parse_args()

    def ev(*parts):            # progress line for the window; silent on the command line
        if a.progress:
            print("@@" + " ".join(str(x) for x in parts), flush=True)

    if a.job or a.project:
        orig, org = prepare_project(a.project or Path(__file__).resolve().parents[1] / "jobs" / a.job)
        a.src = a.src or [str(orig)]
        a.out = a.out or str(org)
    if not a.src or not (a.out or a.desktop):
        ap.error("give --project DIR, --job JOB, or PDF(s) with --out / --desktop")
    pdfs = []
    for s in a.src:
        p = Path(s)
        pdfs += sorted(p.rglob("*.pdf"), key=lambda x: str(x).lower()) if p.is_dir() else [p]
    if not pdfs:
        raise SystemExit(f"no PDF drawings found in {', '.join(a.src)}")
    if a.out:            # an earlier split PDF open in a viewer cannot be replaced: say so now, not after the OCR
        locked = []
        for f in Path(a.out).glob("*.pdf"):
            try:
                with open(f, "r+b"):
                    pass
            except PermissionError:
                locked.append(f.name)
        if locked:
            raise SystemExit(f"close these in your PDF viewer first, then run again: {', '.join(locked)}")

    STEPS = ["Scan pages", "Read title blocks", "Read outlined drawings (OCR)", "Group buildings", "Vectorize outlined / scanned sheets",
             "Write lean building PDFs"]
    ev("STEPS", "|".join(STEPS))
    ev("STEP", 1)
    pages, lists, heads, per_pdf = [], {}, [], []
    total_pages = sum(fitz.open(p).page_count for p in pdfs)
    for pdf in pdfs:
        doc = fitz.open(pdf)
        lst, head = drawing_list(doc)
        if head:
            heads.append(head)
        lists.update({n: (t, r, c) for n, t, r, c in lst})
        seq = [n for n, _, _, c in lst for _ in range(c)]
        list_page = None
        mine = []
        for i, page in enumerate(doc):
            page.remove_rotation()
            text = page_text(page)
            if list_page is None and lst and sum(1 for n, _, _, _ in lst if n in text) >= 3:
                list_page = i
            m = DWG.search(text)
            try:
                raw = page.read_contents()
            except Exception:
                raw = b""
            p = {"pdf": str(pdf), "i": i, "num": base_number(m.group(0)) if m else "", "text_len": len(text),
                 "secs": {x.group(0).replace(" ", "").upper() for x in SECTION.finditer(text)}, "dens": len(raw) / 1e6,
                 "hash": hashlib.md5(raw).hexdigest(), "steely": bool(STEELY.search(text)), "steelwork": bool(STEEL_WORK.search(text)),
                 "strong": len(STRONG.findall(text)), "list": False,
                 "ndraw": raw.count(b" l\n") + raw.count(b" re\n") + raw.count(b" c\n"),
                 "raster": sum(abs(fitz.Rect(x["bbox"]) & page.rect) for x in page.get_image_info()) > 0.5 * abs(page.rect)}
            # same test as vectorize_drawings.classify
            p["kind"] = ("raster" if p["raster"] and p["ndraw"] < 3000 else
                         "outlined" if p["text_len"] < 800 and p["ndraw"] > 5000 else "vector")
            # the title block is text but the drawing itself is outlines (M.E.I. structural sheets: 105 of 268): the
            # content is read by OCR before grouping (the curator: a sheet is judged by what it shows, not by its name)
            if p["kind"] == "vector" and p["ndraw"] > 5000 and body_chars(page) < 300:
                p["kind"], p["body_outlined"] = "outlined", True
            p["content"] = text
            mine.append(p)
            ev("PROG", (len(pages) + len(mine)) / total_pages, f"page {len(pages) + len(mine)}/{total_pages}")
        pages += mine
        if list_page is not None:
            mine[list_page]["list"] = True
            for k, n in enumerate(seq):
                j = list_page + 1 + k
                if j < len(mine):
                    mine[j]["num_list"] = n
        per_pdf.append((doc, mine))

    # ---------------- what this run will do, before the slow part ----------------
    by_kind = {k: [p for p in pages if p["kind"] == k] for k in ("vector", "outlined", "raster")}
    tb_ocr = [p for p in pages if not p["list"] and not a.no_ocr and p["kind"] != "vector"]   # OCR only where the title block text is outlined

    def where(ps):
        out = []
        for pdf in dict.fromkeys(p["pdf"] for p in ps):
            idx = [p["i"] + 1 for p in ps if p["pdf"] == pdf]
            runs, s = [], idx[0]
            for x, y in zip(idx, idx[1:] + [None]):
                if y != x + 1:
                    runs.append(f"{s}" if s == x else f"{s}-{x}"); s = y
            out.append(f"{Path(pdf).name} (pages {', '.join(runs)})")
        return "; ".join(out)

    # seconds per step, measured on the curator's PC (a test job: 38 A1 sheets; Job A: outlined sheets ~30 s OCR each)
    par = lambda n: max(1, min(n, max(1, (os.cpu_count() or 4) // 6), 4))  # OCR runs this many pages at once (vectorize_drawings._run_pool)
    nv = len(by_kind["outlined"]) + len(by_kind["raster"])
    body_ocr = [] if a.no_ocr else [p for p in pages if p["kind"] == "outlined" and not p["list"]]
    w = [0.04 * total_pages, 0.15 * total_pages + 10 * len(tb_ocr) / par(len(tb_ocr)),
         30 * len(body_ocr) / par(len(body_ocr)), 0.5,
         0 if a.no_vectorize else 5 * len(by_kind["outlined"]) + 70 * len(by_kind["raster"]) / par(nv),   # outlined: OCR'd in step 3
         (0.25 if not a.no_lean else 0.2) * total_pages]
    print(f"{total_pages} pages in {len(pdfs)} PDF(s):")
    if by_kind["outlined"]:
        print(f"  * {len(by_kind['outlined'])} pages have the text drawn as outlines -> OCR to become real vector: "
              f"{where(by_kind['outlined'])}")
    if by_kind["raster"]:
        print(f"  * {len(by_kind['raster'])} pages are scans -> traced to vector + OCR (the slow step): {where(by_kind['raster'])}")
    else:
        print("  * No scanned pages, so nothing needs the slow trace step.")
    print(f"  * {len(by_kind['vector'])} pages are already true vector.")
    if a.no_vectorize:
        print("  * Vectorize is switched off (--no-vectorize).")
    est = sum(w)
    print(f"  * Estimated time: about {f'{round(est / 60)} min' if est >= 90 else f'{max(10, round(est, -1)):.0f} s'}"
          + (f" ({round(w[2] / 60)} min of it OCR of outlined drawings, {par(len(body_ocr))} sheets at a time;"
             " sheets read on an earlier run are not read again)" if w[2] else "") + ".", flush=True)
    ev("PLAN", *[round(x, 1) for x in w])

    ev("STEP", 2)
    sys.path.insert(0, str(Path(__file__).parent))
    # 2a look at every title block (which ones are outlined and need OCR), 2b OCR those in parallel, 2c read them all;
    # the share of the step each part gets follows its estimated time, so the bar only moves forward
    need = [((k, p["i"]), doc[p["i"]]) for k, (doc, mine) in enumerate(per_pdf) for p in mine
            if not p["list"] and not a.no_ocr and (p["text_len"] < 800 or not p["num"])]
    share = {"a": 0.5 * len(need), "b": 0.0, "c": 0.1 * total_pages}

    def frac(part, x):
        t = sum(share.values()) or 1
        lo = {"a": 0, "b": share["a"], "c": share["a"] + share["b"]}[part]
        return (lo + x * share[part]) / t

    def planned(n, cached=0):
        left = n - cached
        share["b"] = 25.0 * left / par(left) + 0.05 * cached if n else 0.0
        w[1] = sum(share.values())
        ev("PLAN", *[round(x, 1) for x in w])
        if n:
            print(f"  * {n} title blocks have their text drawn as outlines -> OCR"
                  + (f": {cached} already read on an earlier run (saved, not read again)" if cached else "")
                  + (f", {left} to read, {par(left)} at a time (about {max(1, round(25.0 * left / par(left) / 60))} min)."
                     if left else ", nothing left to read."), flush=True)

    tbs = {}
    pre = prefetch_titleblock_ocr(
        need, tbs,
        check=lambda d, n: ev("PROG", frac("a", d / n), f"checking title block {d}/{n}"),
        planned=planned,
        progress=lambda d, n: ev("PROG", frac("b", d / n), f"OCR title block {d}/{n}")) if need else {}
    done = 0
    for k, (doc, mine) in enumerate(per_pdf):
        prev, fam_titles = None, {}
        for p in mine:
            done += 1
            ev("PROG", frac("c", done / total_pages), f"page {done}/{total_pages}")
            if p["list"]:
                continue
            tb = read_titleblock(doc[p["i"]], use_ocr=not a.no_ocr and (p["text_len"] < 800 or not p["num"]),
                                 pre_ocr=pre.get((k, p["i"])), tb=tbs.get((k, p["i"])))
            p.update({k: tb[k] for k in ("rev", "sheet", "status", "project", "ocr", "title_lines", "title_y")})
            num = base_number(tb["number"]) or p["num"]
            if p.get("num_list") and (not num or num not in lists):
                num = p["num_list"]
            p["num"] = num or prev
            prev = p["num"]
            fam = re.sub(r"\d{2}$", "", p["num"] or "")
            if tb["title"]:
                fam_titles[fam] = tb["title"]
            t = tb["title"] or lists.get(p["num"], ("",))[0] or fam_titles.get(fam, "")
            p["title"] = t or (f"Drawing {fam}xx" if fam else "")
            if not p["rev"] and p["num"] in lists:
                p["rev"] = lists[p["num"]][1]

    # ---------------- content: OCR the sheets whose drawing is outlines, so every sheet is judged by what it shows ----------------
    ev("STEP", 3)
    if body_ocr:
        read_outlined_content(per_pdf, body_ocr, ev)

    # grid labels of the sheets that may show steel (the grid test is slow on a big sheet: only these)
    docs = {mine[0]["pdf"]: doc for doc, mine in per_pdf if mine}
    for p in pages:
        if not p["list"] and (p["secs"] or p.get("strong") or STEEL_TITLE.search(p.get("title") or "")):
            p["grids"] = sorted(grid_labels(docs[p["pdf"]][p["i"]], p))

    # ---------------- groups ----------------
    ev("STEP", 4)
    unrelated, names = [], []
    iso = iso_groups(pages)
    title_pages = pages
    if iso:          # ISO 19650 numbers: the zone code decides; the title rules only for sheets without such a number
        groups_iso, unrelated, title_pages = iso
        unrelated = unrelated + [p for p in pages if p["list"]]
    for p in title_pages:
        if not p["list"] and p.get("num"):
            p["building"], p["disc"] = building_of(p["title"]), discipline(p["title"], p["num"])
    # no drawing number on the sheet (one PDF per building, no title-block numbering): the source file is the building
    nothing_read = not any(p.get("num") for p in pages if not p["list"])
    for p in title_pages:
        if nothing_read and not p["list"] and not p.get("building"):
            stem = re.sub(r"^(drawings?|dwgs?)\s*[-_:]\s*", "", Path(p["pdf"]).stem, flags=re.I).strip()
            p["building"], p["disc"] = building_of(stem) or stem, "structural"
            p["num"] = p.get("num") or ""
            p["fallback"] = True
    for p in title_pages:
        b = p.get("building")
        if not b:
            continue
        hit = next((n for n in names if same_building(n, b)), None)
        if hit is None:
            names.append(b)
        elif len(b) < len(hit):
            names[names.index(hit)] = b
    groups = defaultdict(list)
    for p in title_pages:
        b = p.get("building")
        if b:
            groups[next(n for n in names if same_building(n, b))].append(p)
        else:
            unrelated.append(p)
    if iso:
        for name, ps in groups_iso.items():
            groups[name] += ps
    ranked, dropped, not_model = [], [], []
    for gname, gps in groups.items():
        d = Counter(p["disc"] for p in gps)
        if not (d["structural"] or d["architectural"] or any(p["steely"] for p in gps)):
            unrelated += gps
            continue
        if no_steel(gps):
            unrelated += gps
            for p in gps:
                p["building"] = None
            dropped.append(f"{gname} ({len(gps)} sheet{'s' * (len(gps) > 1)})")
            continue
        units, drop = model_units(gname, gps)
        unrelated += [p for p in drop if p not in unrelated]
        not_model += drop
        for p in drop:
            p["building"] = None
        for name, ps in units.items():
            for p in ps:
                if not p.get("levels_only"):
                    p["building"] = name
            d = Counter(p["disc"] for p in ps)
            secs = set().union(*(p["secs"] for p in ps))
            score = 3 * d["structural"] + d["architectural"] + 0.5 * d["foundation"] + 0.5 * len(secs) + sum(p["dens"] for p in ps)
            ranked.append({"name": name, "slug": slug(name), "score": round(score, 1), "sheets": len(ps), "disciplines": dict(d),
                           "struct_sheets": d["structural"], "sections": len(secs), "drawings": sorted({p["num"] for p in ps}),
                           "levels_from_architect": sum(1 for p in ps if p.get("levels_only")),
                           "pages": [[Path(p["pdf"]).name, p["i"] + 1] for p in ps], "_ps": ps})
    ranked.sort(key=lambda r: -r["score"])

    # ---------------- vectorize: related non-vector sheets -> true vector (unrelated left as they are) ----------------
    ev("STEP", 5)
    if not a.no_vectorize:
        import sys as _s
        _s.path.insert(0, str(Path(__file__).parent))
        import vectorize_drawings as V
        import tempfile
        work = Path(tempfile.mkdtemp(prefix="vectorize_"))          # working copies stay out of the delivered folder
        related = defaultdict(list)
        for r in ranked:
            for p in r["_ps"]:
                if p["kind"] != "vector":
                    related[p["pdf"]].append(p["i"])
        todo, vdone = sum(len(v) for v in related.values()), [0]

        def vprog(i, kind):
            vdone[0] += 1
            ev("PROG", vdone[0] / todo, f"sheet {vdone[0]}/{todo}")

        for n, (src, idx) in enumerate(related.items()):
            vp = work / f"{n:03d}_{Path(src).stem}_vector.pdf"     # numbered: same file name in two subfolders never collides
            kinds = V.vectorize_pages(src, sorted(idx), vp, progress=vprog,
                                      force={p["i"] for r in ranked for p in r["_ps"] if p["pdf"] == src and p.get("body_outlined")})
            for r in ranked:
                for p in r["_ps"]:
                    if p["pdf"] == src:
                        p["vector"] = kinds.get(p["i"], "vector")
                        if vp.exists():          # no copy = every page was already vector: keep the source
                            p["pdf"] = str(vp)

    # ---------------- issues ----------------
    issues = []
    byhash = defaultdict(list)
    for p in pages:
        byhash[p["hash"]].append(p)
    for ps in byhash.values():
        if len(ps) > 1 and ps[0]["ndraw"] > 50:
            issues.append(f"DUPLICATE: identical pages {', '.join(f'{Path(p['pdf']).name} p{p['i'] + 1}' for p in ps)} ({ps[0].get('num')})")
    bysheet = defaultdict(list)
    for p in pages:
        if p.get("num") and p.get("sheet"):
            bysheet[(p["num"], p["sheet"])].append(p)
    for (n, s), ps in bysheet.items():
        if len({p["hash"] for p in ps}) > 1:
            issues.append(f"DUPLICATE SHEET: {n} sheet {s} appears {len(ps)} times with different content "
                          f"({', '.join('p' + str(p['i'] + 1) for p in ps)}; revs {', '.join(p.get('rev') or '?' for p in ps)})")
    revs = defaultdict(set)
    for p in pages:
        if p.get("num") and p.get("rev"):
            revs[p["num"]].add(p["rev"])
    for n, rs in revs.items():
        if len(rs) > 1:
            issues.append(f"SEVERAL REVISIONS: {n} revs {', '.join(sorted(rs))} - the latest governs, older ones superseded")
    found = Counter(p.get("num") for p in pages if p.get("num") and not p["list"])
    for n, (t, r, c) in lists.items():
        if found[n] < c:
            issues.append(f"MISSING SHEETS: {n} ({t[:50]}) list says {c} sheet(s), package has {found[n]}")
        elif found[n] > c:
            issues.append(f"EXTRA SHEETS: {n} list says {c} sheet(s), package has {found[n]}")
    # "sheet x of y" may run over one number (Job A: same number, sheets 1-3) or over a numbered set
    # (XXXX-LAY-S0-00-SLINY-001/002/003 = sheets 1/3, 2/3, 3/3): count the sheets per drawing family, not per number
    tot, fam_nums = defaultdict(set), defaultdict(set)
    for p in pages:
        if p.get("sheet") and p.get("num"):
            x, y = map(int, p["sheet"].split("/"))
            fam = re.sub(r"\d+$", "", p["num"])
            tot[(fam, y)].add(x)
            fam_nums[(fam, y)].add(p["num"])
    for (fam, y), xs in tot.items():
        miss = sorted(set(range(1, y + 1)) - xs)
        nums = sorted(fam_nums[(fam, y)])
        if miss and not any(n in lists for n in nums):
            issues.append(f"SHEET GAP: {', '.join(nums)} has sheets {sorted(xs)} of {y}; missing {miss}")
    # project prefix = before the first "-" (SJI-ARP-...); numbers joined by "_" (M.E.I. WH4_0AR_...) have no such prefix:
    # each whole number was counted as its own project (249 false OTHER PROJECT lines)
    proj = Counter(p["num"].split("-")[0] for p in pages if "-" in (p.get("num") or ""))
    key = lambda t: re.sub(r"[^A-Z]", "", (t or "").upper())[:24]        # project title, spaces / OCR noise ignored
    if len(proj) > 1:
        main_code = proj.most_common(1)[0][0]
        main_pt = Counter(key(p.get("project")) for p in pages
                          if (p.get("num") or "").startswith(main_code) and p.get("project")).most_common(1)
        main_pt = main_pt[0][0] if main_pt else ""
        for code, k in proj.items():
            if code != main_code:
                pt = next((p.get("project") for p in pages if (p.get("num") or "").startswith(code) and p.get("project")), "")
                if pt and main_pt and key(pt) == main_pt:
                    continue        # same project title: another number prefix (a zone code, an OCR slip), not another job
                issues.append(f"OTHER PROJECT: {k} sheets numbered {code}-... " + (f"({pt[:60]}) " if pt else "") +
                              f"- reference drawings from another job, not the {main_code} project; check before pricing")
    st = Counter(p.get("status") for p in pages if p.get("status"))
    if st and set(st) - {"FOR CONSTRUCTION", "AS BUILT"}:
        issues.append("STATUS: " + ", ".join(f"{k} {v} sheets" for k, v in st.items()) + " - not issued for construction")
    nonum = [f"{Path(p['pdf']).name} p{p['i'] + 1}" for p in pages if not p.get("num") and not p["list"]]
    if nonum:
        issues.append(f"NO DRAWING NUMBER: {'; '.join(nonum)}")
    if not_model:
        dk = Counter(p["disc"] for p in not_model)
        issues.append(f"NOT FOR THE MODEL: {len(not_model)} sheets show no steel member ("
                      + ", ".join(f"{v} {k}" for k, v in dk.most_common())
                      + "; RC, finishes, schedules, M&E, site) -> 99_unrelated.pdf. Architectural elevations / sections stay "
                        "with a unit only when it has no steel elevation or section (for the levels)")
    if dropped:
        issues.append(f"NO STEEL: {'; '.join(dropped)} - no steel structure or member size shown (site plan, M&E, RC) -> 99_unrelated.pdf")
    blank = [p["i"] + 1 for p in pages if p["ndraw"] < 20 and p["text_len"] < 50]
    if blank:
        issues.append(f"BLANK PAGES: {blank}")
    ocr_n = sum(1 for p in pages if p.get("ocr"))
    if ocr_n:
        issues.append(f"OUTLINED TEXT: {ocr_n} sheets had no real text; they now carry an OCR text layer (step 0a) - "
                      "OCR can misread a digit: check the numbers a model or MTO depends on against the drawing")

    # ---------------- write ----------------
    if heads:
        project = heads[0].split("Steel Structure")[0]
    else:
        pc = Counter(" ".join(p["project"].split()[:8]) for p in pages if p.get("project"))
        # no project title read: the project folder's own name (Equalbase Drawings), not the first PDF's ("COMBINE (2)")
        project = pc.most_common(1)[0][0] if pc else (project_root(a.project).name if a.project else Path(pdfs[0]).stem)
    project = safe_folder(project)
    out = Path.home() / "Desktop" / project if a.desktop else Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def write(ps, path):
        dst = fitz.open()
        for p in ps:
            how = copy_page(dst, p["pdf"], p["i"])
            if how != "copy":
                FIXED_PAGES.append(((p["pdf"], p["i"]), how))
        dst.save(path, garbage=3, deflate=True)

    ev("STEP", 6)
    for f in list(out.glob("*.pdf")) + list((out / "lean").glob("*.pdf")):      # earlier runs (lean/ was the old layout)
        f.unlink()
    if (out / "lean").is_dir() and not any((out / "lean").iterdir()):
        (out / "lean").rmdir()
    # every lean page at once in worker processes (85 % of this step), then each building PDF is put together
    done = {} if a.no_lean else lean_pages_parallel([p for r in ranked for p in r["_ps"]],
                                                    progress=lambda d, t: ev("PROG", 0.9 * d / t, f"lean page {d}/{t}"))
    for k, r in enumerate(ranked):
        # one PDF per building, and it IS the lean copy (the curator: the full copy is not needed; --no-lean = plain copy)
        r["file"], r["primary"] = f"{k + 1:02d}_{r['slug']}.pdf", k == 0
        ps = r.pop("_ps")
        if a.no_lean:
            write(ps, out / r["file"])
        else:
            lean_copy(ps, out / r["file"], done=done)
            r["lean"] = r["file"]
        ev("PROG", 0.9 + 0.1 * (k + 1) / len(ranked), f"building PDF {k + 1}/{len(ranked)}")
        ev("NOTE", f"{r['file']} written ({len(ps)} sheets)")
    if unrelated:
        write(unrelated, out / "99_unrelated.pdf")
    if FIXED_PAGES:
        fx = sorted({(Path(k[0]).name, k[1] + 1, how) for k, how in FIXED_PAGES})
        issues.append(f"DAMAGED SOURCE PDF: {len(fx)} page(s) had broken internal references and were repaired "
                      "(raster = copied as an image + its text, check it against the original): "
                      + "; ".join(f"{n} p{i} ({h})" for n, i, h in fx))
    man ={"project": project, "source": [str(p) for p in pdfs], "primary": ranked[0]["slug"] if ranked else None,
           "primary_why": (f"highest score {ranked[0]['score']} ({ranked[0]['struct_sheets']} structural sheets, "
                           f"{ranked[0]['sections']} steel sections named, {ranked[0]['sheets']} sheets)") if ranked else "no steel building found",
           "buildings": ranked, "unrelated": [[Path(p["pdf"]).name, p["i"] + 1] for p in unrelated], "issues": issues,
           # why each sheet went where it did: member sizes, steel words, grid labels, the structure its content names
           "sheets": [{**{k: p.get(k) for k in ("i", "num", "title", "rev", "sheet", "status", "building", "disc", "ocr", "grids")},
                       "file": Path(p["pdf"]).name, "sizes": sorted(p.get("secs") or [])[:12], "steel_words": p.get("strong", 0),
                       "content_ocr": bool(p.get("content_read")), "subject": subject(p)} for p in pages]}
    (out / "manifest.json").write_text(json.dumps(man, indent=1, default=list), encoding="utf-8")
    lines = [f"{r['file']}  {r['name']}: {r['sheets']} sheets {r['disciplines']}, score {r['score']}" + ("  <- PRIMARY" if r["primary"] else "")
             for r in ranked]
    lines.append(f"99_unrelated.pdf  {len(unrelated)} pages")
    (out / "split.md").write_text(f"# {project}\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    (out / "issues.md").write_text(f"# Drawing issues - {project}\n\n" + ("\n".join(f"- {i}" for i in issues) or "- none found") + "\n",
                                   encoding="utf-8")
    ev("PROG", 1, "done")
    ev("DONE", out)
    print(f"project folder: {out}")
    print("\n".join(lines))
    print(f"issues: {len(issues)} (issues.md)" + "".join(f"\n  - {i[:150]}" for i in issues[:8]))


if __name__ == "__main__":
    main()
