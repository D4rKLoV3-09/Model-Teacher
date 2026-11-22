#!/usr/bin/env python3
"""data.json -> one self-contained three.js viewer page.

    python build_model.py DATA.json OUT_Rnn.html [--template viewer_template.html] [--scratch]
    (a model page is never replaced: OUT must be a new <name>_Rnn.html, see revision.py; --scratch for tests)

DATA.json is the model data block (parts with convex prisms, groups, layers, units, ...).
The fixed viewer UI lives in viewer_template.html; the MTO export code is injected from
generate-mto/port/mto_workbook.js (the MTO agent owns the workbook format); project text comes from data["meta"],
data["header"] and data["noteSections"]. Nothing project-specific is hard-coded in the template.
"""
import argparse
import base64
import gzip
import json
import math
import pathlib
import struct
import sys

REQUIRED = ["header", "meta", "parts", "groups", "layers", "units", "unitKeys", "ghosts",
            "gx", "gy", "profs", "cats", "notes", "left", "assumptions", "noteSections"]
META_KEYS = ["title", "grand_total_label", "currency", "export_prefix"]
PART_KEYS = ["t", "ax", "prof", "layer", "unit", "lu", "tw", "asm", "sub", "L", "kg", "src", "grid", "desc", "plate", "pr"]   # optional: std (standard model drawn by the page)
MAX_PAGE_MB = 16


def validate(d):
    """Return a list of problems. Empty list = safe to build."""
    errs = []
    missing = [k for k in REQUIRED if k not in d]
    if missing:
        return [f"missing top-level keys: {missing}"]
    errs += [f"meta.{k} missing" for k in META_KEYS if k not in d["meta"]]
    card = d["meta"].get("card", {})
    errs += [f"meta.card.{k} missing (project header card)" for k in ("title", "desc", "ref") if not card.get(k)]
    hr = [p for p in d["parts"] if p.get("t", "").upper().startswith("HR")]
    if hr and not any(p.get("rail") == "top" for p in hr):
        errs.append("handrail parts (HR) but no top rail flagged rail='top': kg/m run cannot be shown")
    if len(d["units"]) != len(d["unitKeys"]):
        errs.append(f"units ({len(d['units'])}) and unitKeys ({len(d['unitKeys'])}) differ in length")
    n_layer, n_unit, n_group = len(d["layers"]), len(d["units"]), len(d["groups"])
    ghost_layers = {i for i, l in enumerate(d["layers"]) if l.get("k") == "ghost"}
    for i, g in enumerate(d["ghosts"]):
        if "layer" in g and g["layer"] not in ghost_layers:
            errs.append(f"ghost[{i}]: layer {g['layer']} is not a ghost layer (k='ghost')")
    if d["ghosts"] and not ghost_layers:
        errs.append("ghosts present but no layer with k='ghost'")
    bdk = {p["layer"] for p in d["parts"] if (p.get("mto") or {}).get("kind") == "measure" and p.get("t", "")[:2] in ("BD", "SS", "EF")}
    if len(bdk) > 1:
        errs.append(f"bondek, studs and edgeform must share ONE layer (found layers {sorted(bdk)})")
    errs += [f"layer {li} ({d['layers'][li]['n']}): the bondek package layer must be hidden by default (off: true)"
             for li in bdk if 0 <= li < n_layer and not d["layers"][li].get("off")]
    covered = {li for c in d["cats"] for li in c["layers"]}
    for i, p in enumerate(d["parts"]):
        tag = f"part[{i}] {p.get('t', '?')}"
        absent = [k for k in PART_KEYS if k not in p]
        if absent:
            errs.append(f"{tag}: missing {absent}")
            continue
        if not 0 <= p["layer"] < n_layer:
            errs.append(f"{tag}: layer {p['layer']} out of range")
        elif p["layer"] not in covered:
            errs.append(f"{tag}: layer {p['layer']} is in no MTO category (export would fail)")
        if not 0 <= p["unit"] < n_unit:
            errs.append(f"{tag}: unit {p['unit']} out of range")
        for g in ("lu", "tw", "asm"):
            if not 0 <= p[g] < n_group:
                errs.append(f"{tag}: group index {g}={p[g]} out of range")
        if p["sub"] != -1 and not 0 <= p["sub"] < n_group:
            errs.append(f"{tag}: sub={p['sub']} out of range")
        if d["layers"][p["layer"]].get("k") == "ghost" if 0 <= p["layer"] < n_layer else False:
            errs.append(f"{tag}: steel part on a ghost layer")
        measured = (p.get("mto") or {}).get("kind") in ("measure", "piece")   # bondek package / bolt sets: own count or kg per piece
        if p["plate"] is None and not measured and p["prof"] not in d["profs"]:
            errs.append(f"{tag}: section '{p['prof']}' not in profs (no unit weight)")
        if not p["pr"]:
            errs.append(f"{tag}: no geometry")
        for prism in p["pr"]:
            for face in prism:
                if len(face) < 3 or any(len(v) != 3 for v in face):
                    errs.append(f"{tag}: malformed face")
                    break
        if len(errs) > 40:
            errs.append("... stopped after 40 problems")
            break
    return errs


def pack_geometry(parts):
    """Take every part's `pr` out of the data and pack it as one compact binary block (geometry v1).

    Corners are rounded to whole mm (max 0.5 mm off; the clash and gap checks run on the unrounded spec before
    this) and kept once in a vertex table; faces point into it. Layout, little-endian int32:
    header [nV, nCounts, nIdx] | vertices as deltas (x, y, z) | counts per part [nPrisms, per prism nFaces, per
    face nVerts] | vertex indices as deltas. gzip then shrinks the deltas to a few bytes a corner.
    The page (loadData) rebuilds `pr` from it, so everything after loading is unchanged."""
    vid, verts, counts, idx = {}, [], [], []
    for p in parts:
        pr = p.pop("pr")
        counts.append(len(pr))
        for prism in pr:
            counts.append(len(prism))
            for face in prism:
                counts.append(len(face))
                for v in face:
                    k = tuple(math.floor(c + 0.5) for c in v)
                    i = vid.get(k)
                    if i is None:
                        i = vid[k] = len(verts)
                        verts.append(k)
                    idx.append(i)
    dv, prev = [], (0, 0, 0)
    for v in verts:
        dv += [v[0] - prev[0], v[1] - prev[1], v[2] - prev[2]]
        prev = v
    di, last = [], 0
    for i in idx:
        di.append(i - last)
        last = i
    flat = [len(verts), len(counts), len(idx)] + dv + counts + di
    return struct.pack(f"<{len(flat)}i", *flat), len(verts), len(idx)


def add_clash_cautions(d, data_path):
    """Floating ends and overlaps of this build (X.clash.json, written by tests/clash_check.py) go into Cautious:
    one line each in the Notes / MTO Summary, and every item in the viewer's Cautious stepper (caution_joints)."""
    p = str(data_path)
    cp = pathlib.Path((p[:-len(".data.json")] if p.endswith(".data.json") else p.rsplit(".", 1)[0]) + ".clash.json")
    if not cp.exists():
        return
    c = json.loads(cp.read_text(encoding="utf-8"))
    if c.get("n_parts") != len(d["parts"]):
        print(f"clash list {cp.name} is from another build ({c.get('n_parts')} parts, model has {len(d['parts'])}): rerun clash_check.py",
              file=sys.stderr)
        return
    fl, ov = c.get("floating", []), c.get("overlaps", [])
    pt = d["parts"]
    nm = lambda i: f"{pt[i].get('t')}{' (Ref ' + pt[i]['ref'] + ')' if pt[i].get('ref') else ''} {pt[i].get('prof', '')}".strip()
    lines, cj = [], []
    if fl:
        lines.append(f"FLOATING ENDS: {len(fl)} member ends touch no steel and no RC (e.g. {'; '.join(nm(f['p'][0]) + ' end ' + f['end'] for f in fl[:3])}). "
                     "Check the support on the drawing before fabrication; step through them with the Cautious button.")
        cj += [{"n": f"Floating end {f['end']} of {nm(f['p'][0])} at {f['at']}", "p": f["p"], "c": "Floating ends"} for f in fl]
    if ov:
        lines.append(f"OVERLAPS: {len(ov)} pairs of parts overlap by more than {c.get('tol', 0.5)} mm (deepest {ov[0]['mm']} mm: "
                     f"{nm(ov[0]['p'][0])} x {nm(ov[0]['p'][1])}). Check the cut / cope at each before fabrication.")
        cj += [{"n": f"Overlap {o['mm']} mm: {nm(o['p'][0])} x {nm(o['p'][1])}", "p": o["p"], "c": "Overlaps"} for o in ov]
    if not lines:
        return
    d["cautions"] = list(d.get("cautions") or []) + lines
    d["caution_joints"] = list(d.get("caution_joints") or []) + cj
    secs = d.setdefault("noteSections", [])
    sec = next((s for s in secs if str(s.get("title", "")).lower().startswith("cautious")), None)
    if sec is None:
        sec = {"title": "Cautious", "items": []}
        at = next((k for k, s in enumerate(secs) if str(s.get("title", "")).lower().startswith("left out")), 0)
        secs.insert(at, sec)
    if "items" in sec:
        sec["items"] = list(sec["items"]) + lines
    print(f"Cautious: +{len(fl)} floating ends, +{len(ov)} overlaps (from {cp.name})")


def build(data_path, out_path, template_path, port_path, scratch=False):
    if not scratch:                                   # the curator 2026-11-23: never replace a model, save <name>_Rnn.html
        import revision
        why = revision.check_new(out_path)
        if why:
            print("NOT WRITTEN:", why, file=sys.stderr)
            return 1
    d = json.loads(pathlib.Path(data_path).read_text(encoding="utf-8"))
    errs = validate(d)
    if errs:
        print("DATA INVALID:", *errs, sep="\n  ", file=sys.stderr)
        return 1
    add_clash_cautions(d, data_path)
    geom, nv, ni = pack_geometry(d["parts"])          # parts lose `pr` here; the page rebuilds it
    d["geom"] = 1
    raw = json.dumps(d, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    b64 = base64.b64encode(gzip.compress(raw, compresslevel=9, mtime=0)).decode("ascii")
    g64 = base64.b64encode(gzip.compress(geom, compresslevel=9, mtime=0)).decode("ascii")
    tpl = pathlib.Path(template_path).read_text(encoding="utf-8")
    assert all(tpl.count(k) == 1 for k in ("@@DATA@@", "@@GEOM@@", "@@TITLE@@", "/*@@MTO_PORT@@*/")), "template placeholders damaged"
    port = pathlib.Path(port_path).read_text(encoding="utf-8")     # the MTO format belongs to generate-mto
    assert "function buildWorkbook(spec)" in port, f"{port_path} is not the MTO workbook port"
    html = tpl.replace("@@TITLE@@", d["meta"]["title"]).replace("/*@@MTO_PORT@@*/", port).replace("@@DATA@@", b64).replace("@@GEOM@@", g64)
    size_mb = len(html.encode("utf-8")) / 1e6
    pathlib.Path(out_path).write_text(html, encoding="utf-8", newline="")
    kg = sum(p["kg"] for p in d["parts"])
    print(f"built {out_path}: {len(d['parts'])} parts, {kg:,.1f} kg, {size_mb:.2f} MB (data {len(b64) / 1e6:.2f} + geometry {len(g64) / 1e6:.2f}: {nv} corners, {ni} face refs; limit {MAX_PAGE_MB} MB)")
    if size_mb > MAX_PAGE_MB:
        print("PAGE TOO LARGE for an artifact: round coordinates or drop detail", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    here = pathlib.Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--template", default=str(here / "viewer_template.html"))
    ap.add_argument("--mto-port", default=str(here.parent / "generate-mto" / "port" / "mto_workbook.js"),
                    help="ExcelJS port of build_mto.py, owned by generate-mto")
    ap.add_argument("--scratch", action="store_true", help="a test / scratch page: may be any name and replace itself")
    a = ap.parse_args()
    sys.exit(build(a.data, a.out, a.template, a.mto_port, a.scratch))
