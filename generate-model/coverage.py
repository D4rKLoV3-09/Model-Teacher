#!/usr/bin/env python3
"""Coverage tables from the drawing store (no PDF opened): which view describes each grid line and each level.

    python generate-model/coverage.py STORE --grids grids.json --datum 4.850 --out coverage.json [--md coverage.md]

grids.json: {"x": {"A": -7500, ...}, "y": {"1": 0, ...}} (the job's grid table, read once).
Views are found from their titles (big text): "ALONG GRID LINE - C", "ALONG GRID LINE C, D, E, F (ROOF TRUSS A1)",
"PLAN @ SHD+20.500m (TOS)", "PLAN @ SHD+4.875 (BOS) TO SHD+7.758M (TOS)", "SECTION S4", "ELEVATION 1", "ROOF TRUSS ...",
"... DETAIL ...". Section cut marks: a section's name found as small text on another view = where it is cut (listed).
Grid rows: drawn (titled for that line), listed (named in a multi-grid title), or NONE (a decision is needed).
Level rows: every level written on an elevation / section (SHD+x.xxx) -> the plan(s) at that level, or NONE.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import store  # noqa: E402
import view_read_store as V  # noqa: E402

CUT_SIZE = 15.0   # pt: section cut marks are the bare section name in big text (Job A: 17.7)
LVL = re.compile(V.LEVEL_RE.pattern + r"\s*M?", re.I)       # the job's level word (view_read_store LEVEL-PREFIX)
KIND = [("elevation", re.compile(r"ALONG\s+GRID\s+LINE|GRID\s+LINE\s*-|ALONG\s+G\.\s*L\b|^ELEVATION\b", re.I)),   # G.L = grid line
        ("plan", re.compile(r"\bPLAN\b", re.I)), ("section", re.compile(r"^SECTION\b", re.I)),
        ("truss", re.compile(r"\bTRUSS\b", re.I)), ("detail", re.compile(r"\bDETAIL", re.I))]


def views(sheets, min_size=12.5):
    out = []
    for sh in sheets:
        big = [w for w in sh.words if w[6] >= min_size and len(w[4]) > 3]
        for w in big:
            t = re.sub(r"\s+", " ", w[4]).strip()
            kind = next((k for k, rx in KIND if rx.search(t)), None)
            if not kind or re.search(r"STEEL STRUCTURAL PLAN|ARCHITECTURAL PLAN, ELEVATION", t):
                continue
            out.append({"sheet": sh.page, "title": t, "kind": kind, "at": [round((w[0] + w[2]) / 2), round(w[3])],
                        "levels": [float(x) for x in LVL.findall(t)]})
    return out


def grids_named(title, names):
    """Grid names a title is ALONG: 'ALONG GRID LINE C, D, E, F' -> [C, D, E, F]; 'B & G' -> [B, G]."""
    m = re.search(r"(?:GRID\s+LINE|\bG\.\s*L\.?)\s*-?\s*(.+?)(?:\(|$)", title, re.I)
    if not m:
        return []
    toks = re.split(r"\s*(?:,|&|AND)\s*", m.group(1).strip())
    return [t.strip() for t in toks if t.strip() in names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("store")
    ap.add_argument("--grids", required=True)
    ap.add_argument("--datum", type=float, default=0.0, help="SHD level of model Z = 0 (m)")
    ap.add_argument("--first-arch", type=int, default=999, help="first architectural sheet (ghost only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md")
    a = ap.parse_args()
    G = json.loads(Path(a.grids).read_text(encoding="utf-8"))
    sheets = store.sheets(a.store)
    V = views(sheets)
    steel = [v for v in V if v["sheet"] < a.first_arch]
    gx, gy = G["x"], G["y"]
    names = set(gx) | set(gy)
    grid_rows = []
    for axis, tbl in (("x", gx), ("y", gy)):
        for g in sorted(tbl, key=tbl.get):
            hits = [v for v in steel if v["kind"] in ("elevation", "truss") and g in grids_named(v["title"], names)]
            grid_rows.append({"grid": g, "axis": axis, "at_mm": tbl[g],
                              "views": [f"p{v['sheet']} {v['title']}" for v in hits],
                              "source": ("drawn" if any(len(grids_named(v["title"], names)) == 1 for v in hits)
                                         else "listed" if hits else "NONE")})
    # levels written on elevations / sections (small text included) vs plans
    lv_seen = {}
    for sh in sheets:
        if sh.page >= a.first_arch:
            continue
        for w in sh.words:
            for x in LVL.findall(w[4]):
                lv_seen.setdefault(round(float(x), 3), set()).add(sh.page)
    plans = [v for v in steel if v["kind"] == "plan" and v["levels"]]
    level_rows = []
    for L in sorted(lv_seen):
        hits = [p for p in plans if any(abs(L - x) < 0.0015 for x in p["levels"]) or
                (len(p["levels"]) == 2 and re.search(r"\bTO\b", p["title"]) and min(p["levels"]) - 0.0015 <= L <= max(p["levels"]) + 0.0015)]
        level_rows.append({"level_shd": L, "z_mm": round((L - a.datum) * 1000), "on_sheets": sorted(lv_seen[L]),
                           "plans": [f"p{p['sheet']} {p['title']}" for p in hits], "source": "plan" if hits else "NONE"})
    # sections: where is each one cut? its short name as small text on another sheet
    sec_rows = []
    for v in [v for v in steel if v["kind"] == "section" and re.search(r"SECTION\s+\S+$", v["title"], re.I)]:
        m = re.search(r"SECTION\s+(\S+)", v["title"], re.I)
        nm = m.group(1) if m else v["title"]
        cuts = []                                       # cut marks: the bare name drawn big (>= CUT_SIZE) on a steel sheet
        for sh in sheets:
            if sh.page >= a.first_arch:
                continue
            for w in sh.words:
                if w[4].strip() == nm and w[6] >= CUT_SIZE:
                    cuts.append((abs(sh.page - v["sheet"]), sh.page, round((w[0] + w[2]) / 2), round((w[1] + w[3]) / 2)))
        if not cuts:                                    # fallback: the bare name at title size on another sheet (Job A p14: S6-S8)
            for sh in sheets:
                if sh.page < a.first_arch and sh.page != v["sheet"]:
                    cuts += [(abs(sh.page - v["sheet"]), sh.page, round((w[0] + w[2]) / 2), round((w[1] + w[3]) / 2))
                             for w in sh.words if w[4].strip() == nm and w[6] >= 12.5 and len(nm) > 1]
        cuts.sort()
        near = sorted({c[1] for c in cuts if c[0] == cuts[0][0]}) if cuts else []   # names repeat per sheet group: nearest sheet
        sec_rows.append({"section": nm, "sheet": v["sheet"], "title": v["title"], "cut_marks_on": [f"p{c}" for c in near],
                         "cut_xy_pt": [list(c[2:]) for c in cuts if c[1] in near][:12], "placed": bool(near)})
    out = {"views": V, "grids": grid_rows, "levels": level_rows, "sections": sec_rows}
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    ng = sum(r["source"] == "NONE" for r in grid_rows)
    nl = sum(r["source"] == "NONE" for r in level_rows)
    ns = sum(not r["placed"] for r in sec_rows)
    print(f"views {len(V)} ({len(steel)} steel): " + ", ".join(f"{k} {sum(v['kind'] == k for v in steel)}" for k, _ in KIND))
    print(f"grid lines {len(grid_rows)}: drawn {sum(r['source'] == 'drawn' for r in grid_rows)}, listed "
          f"{sum(r['source'] == 'listed' for r in grid_rows)}, NONE {ng}")
    print(f"levels {len(level_rows)}: with a plan {len(level_rows) - nl}, NONE {nl}")
    print(f"sections {len(sec_rows)}: cut mark found {len(sec_rows) - ns}, not found {ns}")
    if a.md:
        L = ["# Coverage", "", "## Grid lines", "| Grid | at mm | Source | Views |", "|---|---|---|---|"]
        L += [f"| {r['grid']} | {r['at_mm']} | {r['source']} | {'; '.join(r['views']) or '-'} |" for r in grid_rows]
        L += ["", "## Levels", "| SHD | Z mm | Source | Plans | Written on |", "|---|---|---|---|---|"]
        L += [f"| {r['level_shd']:.3f} | {r['z_mm']} | {r['source']} | {'; '.join(r['plans']) or '-'} | "
              f"{', '.join(f'p{s}' for s in r['on_sheets'][:8])} |" for r in level_rows]
        L += ["", "## Sections", "| Section | Sheet | Cut marks on | Placed |", "|---|---|---|---|"]
        L += [f"| {r['section']} | p{r['sheet']} | {', '.join(r['cut_marks_on']) or '-'} | {'yes' if r['placed'] else 'NO'} |" for r in sec_rows]
        Path(a.md).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
