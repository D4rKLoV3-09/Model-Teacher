#!/usr/bin/env python3
"""The ruling sheet the curator decides from, grouped by kind of drawing situation (called by `rulings.py sheet WORK`).

Every decision (job book D01 ...) and every open ruling case is put in one of a few categories (CATS). Per category:
one line on what it covers, the counts, the tonnage of each option over all its cases, and only a few samples (more when
its cases differ, fewer when they are alike). A sample = the drawing crop (default reading drawn) beside a still of the
3D result as built (the parts involved in red, their neighbours grey; a member left out is drawn dashed where it was
read), the options with their tonnage, and the default taken. Decisions are one compact row each (no explanation).

the curator answers by category ("C4 A": every open rule of C4 that has an option A), by sample ("C4.2 B": that sample's rule)
or by rule ("R08 A"); `rulings.py answer` takes all three (the sample -> rule map is WORK/rulings/samples.json).

Tonnage of an option (when the case carries no `effect`): the read line's length x kg/m of its mark (unmarked: the
median kg/m of the steel around it), counted for the options that build it, against the default.
"""
import base64
import html as H
import io
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

# (id, title, what it covers, rule-id pattern, decision pattern on the "what" text)
CATS = [
    ("C1", "Grids, levels, scale, placing views",
     "where a view sits: grid positions, which scale, which level a plan or truss view is at, which way a section faces",
     r"^(GRID|SCALE|VIEW|LEVEL)-|^R05$", r"\bgrid|\blevel\b|scale|view.*(placed|title)|source of the sheets"),
    ("C2", "Steel or not",
     "lines with no mark, short pieces at columns, stairs, other trades (RC, equipment), parts of the package not modelled",
     r"^(R02|R03|R07|R12|SCOPE-.*|HIDDEN-.*|LAYER-.*|SYMBOL-.*|LEGEND-.*)$", r"\bRC\b|ghost|building of the package|fan|sections S1|unlabelled|isolated|frames into nothing|framing on the plans"),
    ("C3", "Which line a mark names",
     "a mark beside, along or by leader to a line; marks carried on in line, TYP, two marks on one line, marks off their drawn size",
     r"^(R01|R04|R09|R11|NAME-.*|SIZE-R01|COL-.*|MARK-.*|BOX-.*|BAY-.*)$", r"\bmark|label|read twice|duplicate|posts in the elevations|columns drawn in the elevations"),
    ("C4", "Member ends and supports",
     "an end that meets nothing read, overhangs, plan vs elevation ends, seats, column tops, ends short of or into other steel",
     r"^(R06|R08|R10|END-.*)$", r"\bend|support|float|seat|column.*top|top.*column|overlap|stopping short|crane leg|joined|runway"),
    ("C5", "Trusses and components",
     "truss geometry (chords, webs as drawn, depth, work points), purlins, repeated components",
     r"^(TRUSS|COMP)-", r"truss|purlin|diagonal"),
    ("C6", "Connections",
     "bolted / welded joints, plates, splices, cleats, stiffeners, base plates",
     r"^(CONN|PLATE|BOLT|SPLICE)-", r"connection|bolt|splice|cleat|stiffener|base plate|plate"),
    ("C7", "Sizes and the schedule",
     "a mark's section: schedule conflicts, marks not in the schedule, size written on the view vs the legend table",
     r"^(SCHED|CRANE|MTO)-|^SIZE-R0[2-9]", r"\bsize|schedule|legend|UB\s?\d|UC\s?\d"),
    ("C8", "Other (viewer, totals, other agents)",
     "viewer behaviour, tonnage reconciliations, lifting / sequence / deck calls",
     r".", r"."),
]
LEAVE = re.compile(r"leave (it|the member|the column)? ?out(?! only)|left out|not steel|drawing detail only|refused|dropped|nothing extra|mark dropped", re.I)
ROW = re.compile(r"p\d+-[A-Za-z0-9]+-L\d{3}")
CROP_PX = 300          # crop / still width (px): as big as the question needs
MAX_SAMPLES = 6        # per category


def cat_of_rule(rid):
    return next(c[0] for c in CATS if re.search(c[3], rid))


DEC_ORDER = ("C6", "C5", "C7", "C1", "C2", "C3", "C4", "C8")    # a decision about a connection / truss / size first


def cat_of_decision(d):
    t = d.get("what", "")
    cats = {c[0]: c for c in CATS}
    return next(k for k in DEC_ORDER if re.search(cats[k][4], t, re.I))


# ---------------------------------------------------------------- inputs
def _decisions(work, root):
    p = Path(work) / "agent" / "decisions.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    import subprocess
    import sys
    work = Path(work).resolve()
    job = next((q.name for q in [work, *work.parents] if q.parent.name == "jobs"), None)
    if not job:
        return []
    r = subprocess.run([sys.executable, str(root / "tools/job.py"), "decisions", job, "--json"], capture_output=True, text=True, cwd=root)
    try:
        return [{"id": s.split()[0], "what": s.split(":", 1)[0][4:], "chosen": s, "effect": ""} for s in json.loads(r.stdout)]
    except ValueError:
        return []


def _register(work):
    reg = {}
    for f in ("plans.json", "elevations.json"):
        p = Path(work) / "register" / f
        if p.exists():
            for v in json.loads(p.read_text(encoding="utf-8")).get("views", []):
                for r in v.get("rows", []):
                    reg[r["id"]] = r
    return reg


def _parts(work, parts=None):
    if parts:
        p = Path(parts)
    else:
        md = Path(work).parent / "Models Data"
        cand = sorted(md.rglob("*.parts.json"), key=lambda q: q.stat().st_mtime) if md.exists() else []
        p = cand[-1] if cand else None
    if not p or not p.exists():
        return []
    P = json.loads(p.read_text(encoding="utf-8"))
    P = P["parts"] if isinstance(P, dict) else P
    for q in P:
        pts = [pt for sol in q.get("pr") or [] for ring in sol for pt in ring]
        if not pts:
            q["_bb"] = None
            continue
        q["_bb"] = [min(p_[i] for p_ in pts) for i in range(3)] + [max(p_[i] for p_ in pts) for i in range(3)]
        try:
            q["_kgm"] = float(q["kg"]) / max(float(q["L"]), 1) * 1000
        except (KeyError, TypeError, ValueError):
            q["_kgm"] = None
    return P


# ---------------------------------------------------------------- pictures
def _jpeg(im, q=62):
    b = io.BytesIO()
    im.convert("RGB").save(b, "JPEG", quality=q, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def _items_box(reading):
    xs, ys = [], []
    for it in reading if isinstance(reading, list) else []:
        if "line" in it:
            xs += it["line"][0::2]; ys += it["line"][1::2]
        if "cross" in it:
            xs.append(it["cross"][0]); ys.append(it["cross"][1])
    return (min(xs), min(ys), max(xs), max(ys)) if xs else None


def crop(page, box, reading, hit=None, wide=False):
    """The drawing around the case, the default reading drawn on it; at most ~360 pt of sheet, CROP_PX wide.
    hit = [x0, y0, x1, y1] (pt): the line the case asks about, boxed and circled in red (the curator 2026-11-23: show which one).
    wide: a left-out case (nothing built to show in 3D): twice the sheet around it, twice as wide, no reading drawn."""
    import pymupdf
    from PIL import Image, ImageDraw
    x0, y0, x1, y1 = box
    ib = _items_box(reading)
    cx, cy = ((ib[0] + ib[2]) / 2, (ib[1] + ib[3]) / 2) if ib else ((x0 + x1) / 2, (y0 + y1) / 2)
    half = min(max(x1 - x0, y1 - y0) / 2, 180)
    if ib:
        half = min(max(half, (ib[2] - ib[0]) / 2 + 30, (ib[3] - ib[1]) / 2 + 30), 180)
    half = max(half, 45)
    if hit:
        cx, cy = (hit[0] + hit[2]) / 2, (hit[1] + hit[3]) / 2
    if wide:
        half, reading = min(max(2 * half, 150), 330), []
    b = (cx - half, cy - half * 0.75, cx + half, cy + half * 0.75)
    dpi = CROP_PX * (2 if wide else 1) / (b[2] - b[0]) * 72
    pix = page.get_pixmap(dpi=max(int(dpi), 20), clip=pymupdf.Rect(*b))
    dpi = max(int(dpi), 20)
    im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    im = Image.blend(im, Image.new("RGB", im.size, "white"), 0.30)
    g = ImageDraw.Draw(im)
    k = dpi / 72
    T = lambda x, y: ((x - b[0]) * k, (y - b[1]) * k)
    for it in reading if isinstance(reading, list) else []:
        col = tuple(it.get("c", (220, 30, 30)))
        if "line" in it:
            xa, ya, xb, yb = it["line"]
            g.line([T(xa, ya), T(xb, yb)], fill=col, width=3)
        if "cross" in it:
            x, y = T(*it["cross"])
            g.line([(x - 9, y - 9), (x + 9, y + 9)], fill=col, width=3)
            g.line([(x - 9, y + 9), (x + 9, y - 9)], fill=col, width=3)
    if hit:
        (xa, ya), (xb, yb) = T(hit[0], hit[1]), T(hit[2], hit[3])
        pad = 6 * k
        g.rectangle([min(xa, xb) - pad, min(ya, yb) - pad, max(xa, xb) + pad, max(ya, yb) + pad], outline=(230, 0, 0), width=3)
        r_ = max(abs(xb - xa), abs(yb - ya)) / 2 + 22 * k
        mx, my = (xa + xb) / 2, (ya + yb) / 2
        g.ellipse([mx - r_, my - r_, mx + r_, my + r_], outline=(230, 0, 0), width=4)
    return _jpeg(im)


def still(parts, row, ids):
    """3D still of the result as built around a read line: parts built from it red, neighbours grey; a line not built
    is drawn dashed red where it was read. Axonometric from the south-west."""
    from PIL import Image, ImageDraw
    a, b = row.get("a"), row.get("b")
    if not a or not b or None in a + b:
        return None, False
    lo = [min(a[i], b[i]) - 3500 for i in range(3)]
    hi = [max(a[i], b[i]) + 3500 for i in range(3)]
    near, hit = [], []
    for q in parts:
        bb = q.get("_bb")
        if not bb or any(bb[i + 3] < lo[i] or bb[i] > hi[i] for i in range(3)):
            continue
        (hit if set(q.get("ids") or []) & ids else near).append(q)
    near = near[:500]
    ca, sa, ce, se = math.cos(math.radians(-35)), math.sin(math.radians(-35)), math.cos(math.radians(28)), math.sin(math.radians(28))

    def P(p):
        x, y, z = p
        u = x * ca - y * sa
        w = x * sa + y * ca
        return u, -(z * ce + w * se)                    # screen y down
    segs = []
    for grp, col in ((near, (150, 150, 150)), (hit, (215, 25, 25))):
        for q in grp:
            for sol in q.get("pr") or []:
                for ring in sol:
                    segs += [(P(ring[i]), P(ring[(i + 1) % len(ring)]), col) for i in range(len(ring))]
                if len(sol) >= 2:
                    r0, r1 = sol[0], sol[-1]
                    if len(r0) == len(r1):
                        segs += [(P(r0[i]), P(r1[i]), col) for i in range(len(r0))]
    built = bool(hit)
    ra, rb = P(a), P(b)
    pts = [s[0] for s in segs] + [s[1] for s in segs] + [ra, rb]
    ux0, ux1 = min(p[0] for p in pts), max(p[0] for p in pts)
    uy0, uy1 = min(p[1] for p in pts), max(p[1] for p in pts)
    W_, H_ = CROP_PX, int(CROP_PX * 0.75)
    s = min((W_ - 16) / max(ux1 - ux0, 1), (H_ - 16) / max(uy1 - uy0, 1))
    T = lambda p: (8 + (p[0] - ux0) * s, 8 + (p[1] - uy0) * s)
    im = Image.new("RGB", (W_, H_), (252, 252, 250))
    g = ImageDraw.Draw(im)
    for p0, p1, col in sorted(segs, key=lambda t: t[2] == (215, 25, 25)):
        g.line([T(p0), T(p1)], fill=col, width=1 if col[0] == 150 else 2)
    if not built:                                     # left out: dashed where it was read
        (x0, y0), (x1, y1) = T(ra), T(rb)
        n = max(int(math.dist((x0, y0), (x1, y1)) / 8), 1)
        for i in range(0, n, 2):
            g.line([(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n), (x0 + (x1 - x0) * (i + 1) / n, y0 + (y1 - y0) * (i + 1) / n)],
                   fill=(215, 25, 25), width=3)
    g.text((6, H_ - 14), "3D as built" + ("" if built else " - red dashed: not built"), fill=(90, 90, 90))
    return _jpeg(im), built


# ---------------------------------------------------------------- tonnage
def _t(kg):
    return f"{'+' if kg >= 0 else '-'}{abs(kg) / 1000:.1f} t" if abs(kg) >= 50 else "~0 t"


def case_effect(c, rule, reg, kgm_of, parts):
    """{letter: kg against the default} when the case gives none; None when it cannot be told."""
    rid = next(iter(ROW.findall(c.get("note", "") + " " + c.get("key", ""))), None)
    row = reg.get(rid) if rid else None
    if not row or not row.get("L"):
        return None, row
    kgm = kgm_of.get(row.get("mark"))
    if kgm is None:                                   # unmarked: the steel around it
        a, b = row.get("a"), row.get("b")
        if a and b and None not in a + b:
            lo = [min(a[i], b[i]) - 4000 for i in range(3)]
            hi = [max(a[i], b[i]) + 4000 for i in range(3)]
            ks = [q["_kgm"] for q in parts if q.get("_kgm") and q.get("_bb")
                  and not any(q["_bb"][i + 3] < lo[i] or q["_bb"][i] > hi[i] for i in range(3))]
            kgm = statistics.median(ks) if ks else None
    if kgm is None:
        return None, row
    kg = row["L"] / 1000 * kgm
    builds = {k: not LEAVE.search(v) for k, v in rule["options"].items()}
    d = rule["default"]
    return {k: (builds[k] - builds[d]) * kg for k in rule["options"]}, row


# ---------------------------------------------------------------- samples
def pick(cases, n_max):
    """A few samples: one per sheet / mark first (cases that differ), fewer when they are alike."""
    if not cases:
        return []
    feats = lambda c: (c.get("sheet"), re.sub(r"\d+", "", c.get("note", ""))[:30])
    distinct = len({feats(c) for c in cases})
    n = 1 + (len(cases) >= 4) + (distinct >= 6) + (distinct >= 20)
    n = min(n, n_max, len(cases))
    out, seen = [], set()
    for c in sorted(cases, key=lambda c: -(c.get("_kg") or 0)):        # largest tonnage first, then spread
        f = feats(c)
        if f in seen:
            continue
        out.append(c); seen.add(f)
        if len(out) >= n:
            break
    return out


CSS = """
:root{--ink:#1d1d1b;--mute:#6b6b66;--line:#d9d8d2;--bg:#fbfbf8;--card:#fff;--def:#1f8a4c;--hit:#c8281e;--acc:#7a4b00}
@media (prefers-color-scheme: dark){:root:not([data-theme=light]){--ink:#ecebe6;--mute:#a3a29b;--line:#3a3a36;--bg:#1b1b19;--card:#242422;--def:#3fbf75;--hit:#ff6a5f;--acc:#e8b35a;color-scheme:dark}}
:root[data-theme=dark]{--ink:#ecebe6;--mute:#a3a29b;--line:#3a3a36;--bg:#1b1b19;--card:#242422;--def:#3fbf75;--hit:#ff6a5f;--acc:#e8b35a;color-scheme:dark}
*{box-sizing:border-box}body{font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;margin:0;padding-block:16px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));background:var(--bg);color:var(--ink)}
main{max-width:1060px;margin:auto}h1{font-size:22px;margin:4px 0}h2{font-size:18px;margin:0}
.how{color:var(--mute);margin:4px 0 14px}.toc{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px}
.toc a{border:1px solid var(--line);border-radius:14px;padding:2px 10px;color:var(--ink);text-decoration:none;font-size:13px}
.cat{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin:14px 0}
.cov{color:var(--mute);margin:2px 0 6px}.nums{font-size:13px;margin:4px 0}.nums b{font-variant-numeric:tabular-nums}
.tw{overflow-x:auto}.rules{font-size:13px;margin:6px 0;border-collapse:collapse;width:100%}.rules td{border-top:1px solid var(--line);padding:3px 6px;vertical-align:top}
.s{border-top:1px solid var(--line);padding-top:8px;margin-top:8px}.sid{font-weight:700;color:var(--acc)}
.pics{display:flex;flex-wrap:wrap;gap:8px;margin:6px 0}.pics img{width:300px;max-width:100%;border:1px solid var(--line);border-radius:4px;display:block}.pics img.wide{width:608px}
.opt{margin:2px 0}.opt.def{color:var(--def);font-weight:600}.eff{font-variant-numeric:tabular-nums;color:var(--acc);font-weight:600}
details{margin-top:8px}summary{cursor:pointer;color:var(--mute);font-size:13px}.dl{font-size:12.5px;margin:4px 0 0;padding-left:0;list-style:none}
.dl li{padding:2px 0;border-top:1px dotted var(--line)}.n{color:var(--mute);font-size:12px}
"""


def learnt_any(R, k):
    return any((r.get("basis") or r.get("answer")) and cat_of_rule(rid) == k for rid, r in R.items())


def _page_of(work):
    """The job's newest model page (jobs/<job>/Models/**/*.html outside _work, built by build_model.py), or None."""
    models = Path(work).parent
    pages = sorted((q for q in models.rglob("*.html") if "_work" not in q.parts), key=lambda q: q.stat().st_mtime, reverse=True)
    for q in pages[:6]:
        if "window.__model" in q.read_text(encoding="utf-8", errors="ignore")[-4000:]:
            return q
    return None


def model_shots(page_html, wants, n_parts):
    """Screenshots of the model page itself (the curator 2026-11-23): per sample its parts selected in the viewer, framed with the
    building around them. wants {key: [part index]} -> {key: data url}. Empty when no browser can run or the page is of
    another build (the sheet then draws its own still, labelled so)."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return {}
    out = {}
    with sync_playwright() as pw:
        try:
            br = pw.chromium.launch()
        except Exception:
            try:
                br = pw.chromium.launch(channel="chrome")
            except Exception:
                return {}
        pg = br.new_page(viewport={"width": 640, "height": 480})
        pg.goto(Path(page_html).resolve().as_uri())
        pg.wait_for_function("() => !!(window.__model && window.__model.select)", timeout=300000)
        if pg.evaluate("window.__model.parts.length") != n_parts:
            br.close()
            return {}
        pg.evaluate("""() => { const c = document.querySelector('canvas'); const keep = new Set(); for (let e = c; e; e = e.parentElement) keep.add(e);
            for (const e of document.body.querySelectorAll('*')) if (!keep.has(e) && !c.contains(e)) e.style.visibility = 'hidden'; }""")
        for k, ids in wants.items():
            pg.evaluate("""ids => { const M = window.__model; M.select(ids, [{name: 'case', ids}], 0); M.fitIds(ids, false);
                const t = M.controls.target, p = M.camera.position; p.sub(t).multiplyScalar(2.4).add(t); M.controls.update();
                if (window.__wake) window.__wake(); }""", ids)
            pg.wait_for_timeout(450)
            out[k] = "data:image/jpeg;base64," + base64.b64encode(pg.locator("canvas").first.screenshot(type="jpeg", quality=70)).decode()
        br.close()
    return out


def build(work, R, C, root, out, parts_path=None, page_html=None):
    """R = rules, C = cases (all). Writes the sheet and WORK/rulings/samples.json; returns counts."""
    work = Path(work)
    D = _decisions(work, root)
    reg = _register(work)
    parts = _parts(work, parts_path)
    kgm_of = {}
    for q in parts:
        if q.get("_kgm") and q.get("ref"):
            kgm_of.setdefault(q["ref"], q["_kgm"])
    docs, pages = {}, {}

    def page(c):
        import pymupdf
        src = c.get("pdf")
        if not src and (work / "store" / "store.json").exists():
            st = json.loads((work / "store" / "store.json").read_text(encoding="utf-8"))
            src = next(iter(st["pdfs"].values()))["path"]
        if not src or not Path(src).exists() or not c.get("sheet"):
            return None
        if src not in docs:
            docs[src] = pymupdf.open(src)
        k = (src, c["sheet"])
        if k not in pages:
            pages[k] = docs[src][int(c["sheet"]) - 1]
            pages[k].remove_rotation()
        return pages[k]

    openC = [c for c in C if c["rule"] in R and not R[c["rule"]].get("answer")]
    for c in openC:
        eff, row = case_effect(c, R[c["rule"]], reg, kgm_of, parts) if not c.get("effect") else (None, None)
        c["_eff"], c["_row"] = eff, row
        c["_kg"] = max((abs(v) for v in (eff or {}).values()), default=0)
    by_cat = defaultdict(list)
    for c in openC:
        by_cat[cat_of_rule(c["rule"])].append(c)
    dec_cat = defaultdict(list)
    for d in D:
        dec_cat[cat_of_decision(d)].append(d)
    samples, html = {}, []
    html.append(f"<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
                f"<title>Ruling Sheet</title><meta name=claude:viewport-fit-sides content=cover><style>{CSS}</style><main>")
    job = next((q.name for q in [work.resolve(), *work.resolve().parents] if q.parent.name == "jobs"), "")
    n_cases = len(openC)
    html.append(f"<h1>Ruling sheet{' &middot; ' + H.escape(job) if job else ''}</h1>"
                f"<p class=how>{len(D)} decisions and {n_cases} open cases on {len({c['rule'] for c in openC})} rules, in {len([k for k in CATS if by_cat[k[0]] or dec_cat[k[0]]])} kinds of "
                f"drawing situation. The job is built with the <b style='color:var(--def)'>default</b>. Answer by category "
                f"(<b>C4 A</b>: every open rule in C4), by sample (<b>C4.2 B</b>: that sample's rule), by rule (<b>R08 A</b>) "
                f"or change a decision by number (<b>D09 B</b>). An answered rule becomes standing for every later job.</p>")
    html.append("<nav class=toc>" + "".join(f"<a href='#{k}'>{k} {H.escape(t)} &middot; {len(by_cat[k]) + len(dec_cat[k])}</a>"
                                           for k, t, *_ in CATS if by_cat[k] or dec_cat[k] or learnt_any(R, k)) + "</nav>")
    n_shown = 0
    shots = {}                                          # token -> (part indices, drawn still): model page screenshots
    learnt = defaultdict(list)                          # standing rules mined from decisions (basis) or answered by the curator
    for rid, r in R.items():
        if r.get("basis") or r.get("answer"):
            learnt[cat_of_rule(rid)].append((rid, r))
    for k, title, covers, *_ in CATS:
        cs, ds = by_cat[k], dec_cat[k]
        if not cs and not ds and not learnt[k]:
            continue
        rids = sorted({c["rule"] for c in cs}, key=lambda r: -len([c for c in cs if c["rule"] == r]))
        html.append(f"<section class=cat id={k}><h2>{k} &middot; {H.escape(title)}</h2><p class=cov>{H.escape(covers)}</p>"
                    f"<p class=nums><b>{len(cs)}</b> open case(s) on <b>{len(rids)}</b> rule(s); <b>{len(ds)}</b> decision(s) taken</p>")
        if rids:
            rows = []
            for rid in rids:
                r = R[rid]
                rc = [c for c in cs if c["rule"] == rid]
                tot = defaultdict(float)
                known = 0
                for c in rc:
                    if c.get("_eff"):
                        known += 1
                        for L_, v in c["_eff"].items():
                            tot[L_] += v
                opts = "; ".join(f"<span class='{'opt def' if L_ == r['default'] else 'opt'}'>{L_} {H.escape(v[:90])}"
                                 + (f" <span class=eff>{_t(tot[L_])}</span>" if known and L_ != r["default"] else "")
                                 + ("</span>" if True else "") for L_, v in r["options"].items())
                rows.append(f"<tr><td><b>{rid}</b></td><td>{len(rc)}</td><td>{H.escape(r['q'])}<br>{opts}"
                            + (f" <span class=n>(tonnage over {known} of {len(rc)} cases)</span>" if known and known < len(rc) else "") + "</td></tr>")
            html.append("<div class=tw><table class=rules>" + "".join(rows) + "</table></div>")
        # samples: per rule, more when its cases differ; the category capped
        chosen = []
        for rid in rids:
            chosen += pick([c for c in cs if c["rule"] == rid], 3 if len(rids) <= 2 else 2)
        chosen = chosen[:MAX_SAMPLES]
        for i, c in enumerate(chosen, 1):
            sid = f"{k}.{i}"
            samples[sid] = c["rule"]
            r = R[c["rule"]]
            pics = []
            pg = page(c) if c.get("box") else None
            dr = c["readings"].get(r["default"])
            row = c.get("_row") or reg.get(next(iter(ROW.findall(c.get("note", "") + " " + c.get("key", ""))), ""))
            hit = [*row["a_pt"], *row["b_pt"]] if row and row.get("a_pt") else None
            left = bool(LEAVE.search(r["options"][r["default"]]))        # the default builds nothing here: no 3D to show
            if pg is not None:
                pics.append(f"<img {'class=wide ' if left else ''}alt='drawing p{c['sheet']}' "
                            f"src='{crop(pg, c['box'], dr if isinstance(dr, list) else [], hit, wide=left)}'>")
            if row and parts and not left:
                idx = [n for n, q in enumerate(parts) if row["id"] in set(q.get("ids") or [])]
                img, _ = still(parts, row, {row["id"]})
                if idx:                                          # the model page itself, these parts selected (shot below)
                    tok = f"@@SHOT{len(shots)}@@"
                    shots[tok] = (idx, img or "")
                    pics.append(f"<img alt='3D model, selected' src='{tok}'>")
                elif img:
                    pics.append(f"<img alt='3D as built' src='{img}'>")
            opts = []
            for L_, v in r["options"].items():
                rd = c["readings"].get(L_)
                txt = rd if isinstance(rd, str) else ""
                eff = (c.get("effect") or {}).get(L_) or (_t(c["_eff"][L_]) if c.get("_eff") and L_ != r["default"] else "")
                opts.append(f"<div class='{'opt def' if L_ == r['default'] else 'opt'}'><b>{L_}</b> {H.escape(v)}"
                            + (f" &mdash; {H.escape(txt)}" if txt else "") + (f" <span class=eff>{H.escape(eff)}</span>" if eff else "")
                            + (" (default, built)" if L_ == r["default"] else "") + "</div>")
            where = f"p{c['sheet']} &middot; " if c.get("sheet") else ""
            html.append(f"<div class=s><span class=sid>{sid}</span> <b>{c['rule']}</b> <span class=n>{where}{H.escape(c.get('note', ''))}</span>"
                        + (f"<div class=pics>{''.join(pics)}</div>" if pics else "") + "".join(opts) + "</div>")
            n_shown += 1
        if ds:
            html.append(f"<details><summary>{len(ds)} decision(s) taken: " + ", ".join(d["id"] for d in ds) + "</summary><ul class=dl>"
                        + "".join(f"<li><b>{d['id']}</b> {H.escape(d.get('what', '')[:110])} &rarr; {H.escape(d.get('chosen', '')[:120])}"
                                  + (f" <span class=eff>{H.escape(str(d['effect'])[:40])}</span>" if d.get("effect") else "") + "</li>" for d in ds)
                        + "</ul></details>")
        if learnt[k]:
            html.append(f"<details><summary>{len(learnt[k])} standing rule(s) learnt: " + ", ".join(r for r, _ in learnt[k]) + "</summary><ul class=dl>"
                        + "".join(f"<li><b>{rid}</b> {H.escape(r['q'])} &rarr; <b>{r.get('answer') or r['default']}</b> "
                                  f"{H.escape(r['options'][r.get('answer') or r['default']])}"
                                  + (" <span class=n>(the curator)</span>" if r.get("answer") else " <span class=n>(default, open)</span>")
                                  + (f" <span class=n>&middot; {H.escape(r.get('basis', ''))}</span>" if r.get("basis") else "") + "</li>"
                                  for rid, r in learnt[k]) + "</ul></details>")
        html.append("</section>")
    html.append("</main></html>")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = "\n".join(html)
    if shots:                                               # screenshots of the model page; without a browser: the drawn still
        ph = page_html or _page_of(work)
        got = model_shots(ph, {k: v[0] for k, v in shots.items()}, len(parts)) if ph else {}
        for tok, (_, img) in shots.items():
            alt, src = ("3D model, selected", got[tok]) if tok in got else ("3D as built (drawn: no model page shot)", img)
            doc = doc.replace(f"alt='3D model, selected' src='{tok}'", f"alt='{alt}' src='{src}'")
    out.write_text(doc, encoding="utf-8")
    (out.parent / "samples.json").write_text(json.dumps(samples, indent=1))
    cats = {k: [r for r in {c["rule"] for c in by_cat[k]}] for k, *_ in CATS}
    (out.parent / "categories.json").write_text(json.dumps(cats, indent=1))
    return {"decisions": len(D), "cases": n_cases, "samples": n_shown, "kb": round(out.stat().st_size / 1024)}
