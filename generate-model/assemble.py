#!/usr/bin/env python3
"""Register rows (plans, elevations, sections, truss views) + columns + schedule -> members in 3D (mm), with the curator's rulings.
Shared by every model built on the drawing store (models/<name>/build.py calls it). No PDF opened.

    import assemble
    members, rep = assemble.members(work, sched, columns, datum, bos_z, rulings, profile)

Rules (each unsure case -> a ruling card, the job builds with the rule's answer or default):
  marks     a row's mark -> its schedule section; no section -> Left out (listed by mark).
  R01       an unmarked line in line with a marked one, touching it: the same member (default A).
  R02       any other unmarked line: A (the curator 2026-11-22: the tag is not always written, assume it from the similar member, by
            scale) = the mark of the most similar marked member in the view (same direction, same way of drawing and drawn
            depth / pen, similar length, then nearest), ASSUMED; B = left out and listed.
  R03       a short unmarked piece at a column: drawing detail (default B).
  R12       an unmarked line >= 1 m on an elevation / section with both ends on marked members (drawn framing): the mark of
            the nearest marked member of the same direction in the view, ASSUMED (default A, precedent D25 Turbine).
  R07       stair stringers (SB) on a plan: left out (default B).
  levels    a single line on a plan = the member's top at the plan level (TOS - h/2); a level single line on an elevation drawn
            at a level marker = TOS likewise; an outline (two lines) = the member drawn to size: its centre line as drawn.
  R05       a plan with a level range: each member at the level of the elevation members it meets (default B).
  R06/R09   one member drawn in several views = one member: x / y from the plan, z from the elevation, ends = union (R06 A);
            two different marks on exactly one line: the mark whose label sits closest (R09 A).
  columns   the layout plan gives each column (mark, x, y, web direction); vertical rows read on its line in the elevations give
            its bottom and top; a vertical row with a column mark away from every layout column is a column too (listed).
"""
import json
import math
import re
from pathlib import Path

import numpy as np

import view_read_store as V

LEVEL_TOL = 40.0        # mm: a level line this close to a level marker is drawn at that level (TOS)
PLAN_ELEV_OFF = 600.0   # mm: a plan copy and an elevation copy of one beam (elevations show it on the grid line)
COL_TOL = 300.0         # mm: a vertical row this close to a layout column is that column


def _dir_pt(r):
    (ax, ay), (bx, by) = r["a_pt"], r["b_pt"]
    L = math.hypot(bx - ax, by - ay) or 1e-9
    return (bx - ax) / L, (by - ay) / L, L


def _crop(rs, pad=70):
    xs = [v for r in rs for v in (r["a_pt"][0], r["b_pt"][0])]
    ys = [v for r in rs for v in (r["a_pt"][1], r["b_pt"][1])]
    return [min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad]


def _ang(r):
    (ax, ay), (bx, by) = r["a_pt"], r["b_pt"]
    return math.degrees(math.atan2(-(by - ay), bx - ax)) % 180


def _seg_dist(c, a, b):
    ax, ay = a; bx, by = b
    L2 = (bx - ax) ** 2 + (by - ay) ** 2 or 1e-9
    t = max(0.0, min(1.0, ((c[0] - ax) * (bx - ax) + (c[1] - ay) * (by - ay)) / L2))
    return math.dist(c, (ax + t * (bx - ax), ay + t * (by - ay)))


TRUSS_NAME = re.compile(r"ROOF\s+TRUSS\s+([A-Z]\d*[a-z]?)", re.I)


def view_depth(words, box, k=None, rows=None):
    """VIEW-R05: the depth of a truss view drawn with no level: the written vertical dimension at its side (a number set
    upright, e.g. 3219 = chord to chord), else scaled off the drawing (top to bottom marked level lines x k mm/pt)."""
    x0, y0, x1, y1 = box
    up = [w for w in words if re.fullmatch(r"\d{3,5}", w[4].strip()) and (w[3] - w[1]) > (w[2] - w[0])
          and x0 <= (w[0] + w[2]) / 2 <= x1 and y0 <= (w[1] + w[3]) / 2 <= y1]
    if up:
        return float(max(up, key=lambda w: int(w[4]))[4]), "written"
    lv = [r for r in rows or [] if r.get("mark") and abs(r["a_pt"][1] - r["b_pt"][1]) < 1.0]
    if k and len(lv) >= 2:
        ys = [r["a_pt"][1] for r in lv]
        return (max(ys) - min(ys)) * k, "scaled"
    return None, None


def named_hosts(words, box, views, own_title=""):
    """VIEW-R05: the trusses a no-level view names at its posts / ends (ROOF TRUSS A1 by a leader) -> the views drawing
    those trusses (their titles name them). The view sits between them, on their chords."""
    x0, y0, x1, y1 = box
    names = {m.group(1).upper() for w in words if x0 <= w[0] <= x1 and y0 <= w[1] <= y1 for m in [TRUSS_NAME.search(w[4])] if m}
    own = {m.upper() for m in TRUSS_NAME.findall(own_title)}
    names -= own
    return [v for v in views if v is not None and {m.upper() for m in TRUSS_NAME.findall(v.get("title", ""))} & names], sorted(names)


def station_by_depth(depth, hosts, axis, guess, search=4000.0, tol=0.05):
    """VIEW-R05 (the curator 2026-11-22, p3 bracing trusses B / C): a truss view with no level marker is placed where the main truss
    it connects to is as deep as it is: at the main truss's panel point (a post) nearest the guess (plan label) whose
    chord-to-chord depth matches the view's depth (within tol, >= 100 mm). hosts: [{"top": [rows], "bot": [rows], "posts":
    [coord]}] in mm (a / b = [x, y, z]); axis: the no-level view's axis ("x": it runs along x, so it stands at a y).
    -> (station, host depth there) or (None, None)."""
    f = 1 if axis == "x" else 0

    def z_at(rows, s):
        zs = [r["a"][2] + (s - r["a"][f]) / (r["b"][f] - r["a"][f]) * (r["b"][2] - r["a"][2]) for r in rows
              if abs(r["b"][f] - r["a"][f]) > 1 and min(r["a"][f], r["b"][f]) - 100 <= s <= max(r["a"][f], r["b"][f]) + 100]   # (a chord stops short of the ridge post)
        return sorted(zs)[len(zs) // 2] if zs else None
    best = None
    for h in hosts:
        for s in h["posts"]:
            if abs(s - guess) > search:
                continue
            zt, zb = z_at(h["top"], s), z_at(h["bot"], s)
            if zt is None or zb is None:
                continue
            hd = zt - zb
            if abs(hd - depth) <= max(tol * depth, 100.0) and (best is None or abs(s - guess) < abs(best[0] - guess)):
                best = (s, round(hd))
    return best or (None, None)


def place_no_level(views, words_of, top_re, bot_re, rulings=None, work=None):
    """VIEW-R05 over a register: every read view with no level marker (fit z_rel) is moved along its fixed axis to the
    panel point of the trusses it names where their depth matches its own (station_by_depth); its rows follow.
    words_of(sheet) -> the sheet's words. Returns [(view title, old at, new at, depth, host depth)]."""
    read = [v for v in views if v.get("status") == "read"]
    out = []
    for v in read:
        if not v.get("fit", {}).get("z_rel"):
            continue
        words = words_of(v["sheet"])
        d, how = view_depth(words, v["box"])
        hv, names = named_hosts(words, v["box"], [x for x in read if x is not v and x.get("axis") != v.get("axis")], v["title"])
        if not d or not hv:
            continue
        f = 1 if v["axis"] == "x" else 0
        hosts = []
        for h in hv:
            rows = [r for r in h["rows"] if r["a"][2] is not None]
            hosts.append({"top": [r for r in rows if r.get("mark") and top_re.match(r["mark"])],
                          "bot": [r for r in rows if r.get("mark") and bot_re.match(r["mark"])],
                          "posts": sorted({r["a"][f] for r in rows if abs(r["a"][2] - r["b"][2]) > 300
                                           and abs(r["a"][0] - r["b"][0]) < 5 and abs(r["a"][1] - r["b"][1]) < 5})})
        s, hd = station_by_depth(d, hosts, v["axis"], v["at"])
        if rulings is not None and work is not None:
            rulings.case(work, "VIEW-R05", v["sheet"], v["box"], readings={
                "A": f"at {round(s) if s is not None else '-'} (trusses {', '.join(names)} {hd} deep there; view {d:.0f} {how})",
                "B": f"at {v['at']} (its label on the plan)"}, note=f"p{v['sheet']} {v['title'][:40]} @ {v.get('where', '')[:40]}",
                key=f"VIEW-R05:{v['sheet']}:{v['title']}:{v.get('where')}")
        if s is None or (rulings is not None and rulings.choice("VIEW-R05") != "A"):
            continue
        dv = s - v["at"]
        for r in v["rows"]:
            for e in ("a", "b"):
                if r.get(e) is not None and r[e][f] is not None:
                    r[e][f] = round(r[e][f] + dv, 1)
        out.append((v["title"], v["at"], round(s, 1), d, hd))
        v["at"] = round(s, 1)
        v.setdefault("fit", {})["placed"] = f"VIEW-R05: depth {d:.0f} ({how}) = trusses {', '.join(names)} at {round(s)} ({hd})"
    return out


def similar(r, marked):
    """R02 A: the marked member most like r in its view: parallel to it, drawn the same way (outline of about the same drawn
    depth = the same section at the view's scale; single line of the same pen), of similar length, then the nearest."""
    tx, ty, L = _dir_pt(r)
    par = [m for m in marked if abs(tx * _dir_pt(m)[1] - ty * _dir_pt(m)[0]) < 0.035]
    hw = r.get("hw") or 0.0
    upright = abs(r["a_pt"][0] - r["b_pt"][0]) < 1.0

    def key(m):
        mw = m.get("hw") or 0.0
        if bool(r.get("outline")) != bool(m.get("outline")):
            drawn = 2
        elif r.get("outline"):
            drawn = 0 if mw and 0.75 <= hw / mw <= 1.33 else 1
        else:
            drawn = 0 if abs(hw - mw) <= 0.15 else 1
        Lm = _dir_pt(m)[2]
        length = 0 if abs(math.log(max(L, 1e-6) / max(Lm, 1e-6))) <= 0.3 else 1
        # an upright (column / post): the column in line above / below it first, then the side columns (the curator C2.1)
        inline = 0 if upright and abs(m["a_pt"][0] - r["a_pt"][0]) <= 1.5 else 1
        return (inline, drawn, length, math.dist(_mid(m), _mid(r)))
    return min(par, key=key, default=None)


def _mid(r):
    return ((r["a_pt"][0] + r["b_pt"][0]) / 2, (r["a_pt"][1] + r["b_pt"][1]) / 2)


class Assembler:
    def __init__(self, work, sched, profile, rulings, colxy, datum=4.85):
        self.W, self.sched, self.profile, self.R, self.colxy = Path(work), sched, profile, rulings, colxy
        self.datum = datum                                  # model Z = 0 at this drawing level (m); build_job passes job.json's
        self.left = {"no section for the mark": {}, "outside the building": 0}
        self.n = {}
        self.effect = {}
        self.fate = {}       # register row id -> why it is not modelled on its own (count check per view)

    def count(self, k, v=1):
        self.n[k] = self.n.get(k, 0) + v

    def kgm(self, mark):
        return self.profile(self.sched[mark])["kgm"] if mark in self.sched else 0.0

    def unmarked(self, view, kind, truss):
        R, rows = self.R, view["rows"]
        marked = [r for r in rows if r["mark"]]
        changed = True
        while changed:                                       # R01: chains of unmarked pieces in line with a marked one
            changed = False
            for r in rows:
                if r["mark"] or r.get("_rule") or str(r.get("status", "")).startswith("not a member"):
                    continue
                tx, ty, L = _dir_pt(r)
                for m in marked:
                    ux, uy, _ = _dir_pt(m)
                    if abs(tx * uy - ty * ux) > 0.035:
                        continue
                    off = abs(-(r["a_pt"][0] - m["a_pt"][0]) * uy + (r["a_pt"][1] - m["a_pt"][1]) * ux)
                    touch = min(math.dist(p, q) for p in (r["a_pt"], r["b_pt"]) for q in (m["a_pt"], m["b_pt"]))
                    if off < 1.0 and touch < 10.0:
                        pick = R.choice("R01")
                        self.count("R01")
                        R.case(self.W, "R01", view["sheet"], _crop([r, m]), {
                            "A": [{"line": [*m["a_pt"], *m["b_pt"]], "c": (40, 150, 40), "text": m["mark"]},
                                  {"line": [*r["a_pt"], *r["b_pt"]], "c": (40, 150, 40), "text": m["mark"] + " ?"}],
                            "B": [{"line": [*m["a_pt"], *m["b_pt"]], "c": (40, 150, 40), "text": m["mark"]},
                                  {"line": [*r["a_pt"], *r["b_pt"]], "c": (230, 120, 0), "text": "other member, size not given"}],
                            "C": [{"line": [*m["a_pt"], *m["b_pt"]], "c": (40, 150, 40), "text": m["mark"]}, {"cross": list(_mid(r)), "c": (120, 120, 120)}]},
                            note=f"{r['id']} ({view['title'][:40]})")
                        r["_rule"] = "R01"
                        if pick == "A":
                            r["mark"], r["status"] = m["mark"], "R01 A"
                            marked.append(r); changed = True
                        else:
                            self.count("left: R01 " + pick)
                            self.fate[r["id"]] = f"R01 {pick}: in line with {m['mark']}, not modelled"
                        break
        changed = True
        while changed:                                       # R11: in line with a marked one across a break at a support
            changed = False
            for r in rows:
                if r["mark"] or r.get("_rule") or abs(r["a_pt"][1] - r["b_pt"][1]) > 1.0 or str(r.get("status", "")).startswith("not a member"):
                    continue
                tx, ty, L = _dir_pt(r)
                mm_pt = r["L"] / max(L, 1e-6)
                for m in marked:
                    ux, uy, _ = _dir_pt(m)
                    if abs(tx * uy - ty * ux) > 0.01 or abs(m["a_pt"][1] - m["b_pt"][1]) > 1.0:
                        continue
                    off = abs(r["a_pt"][1] - m["a_pt"][1])
                    gap = min(math.dist(p, q) for p in (r["a_pt"], r["b_pt"]) for q in (m["a_pt"], m["b_pt"])) * mm_pt
                    if off < 1.0 and gap <= 1200:
                        self.count("R11")
                        self.R.case(self.W, "R11", view["sheet"], _crop([r, m]), {
                            "A": [{"line": [*m["a_pt"], *m["b_pt"]], "c": (40, 150, 40), "text": m["mark"]},
                                  {"line": [*r["a_pt"], *r["b_pt"]], "c": (40, 150, 40), "text": m["mark"] + " (next bay)"}],
                            "B": [{"line": [*m["a_pt"], *m["b_pt"]], "c": (40, 150, 40), "text": m["mark"]}, {"cross": list(_mid(r)), "c": (120, 120, 120)}]},
                            note=f"{r['id']} ({view['title'][:40]}), {gap:.0f} mm break", key=f"R11:{view['sheet']}:{view['title'][:16]}:{round(r['a_pt'][1])}")
                        r["_rule"] = "R11"
                        if self.R.choice("R11") == "A":
                            r["mark"], r["status"] = m["mark"], "R11 A"
                            marked.append(r); changed = True
                        else:
                            self.fate[r["id"]] = "R11 B: next bay, unmarked"
                        break
        if kind == "elev":
            self.framed_in(view, rows, marked)
        for r in rows:
            if r["mark"] or r.get("_rule") or str(r.get("status", "")).startswith("not a member"):
                continue                                     # (lines drawn with another pen than the members: not steel)
            nm = similar(r, [m for m in marked if m["mark"] in self.sched]) or similar(r, marked)
            near_col = kind == "plan" and r["a"][2] is not None and any(
                math.hypot(p[0] - cx, p[1] - cy) < 400 for p in (r["a"], r["b"]) for cx, cy in self.colxy)
            if r["L"] < 1000 and near_col:
                self.count("R03")
                R.case(self.W, "R03", view["sheet"], _crop([r]), {
                    "A": [{"line": [*r["a_pt"], *r["b_pt"]], "c": (40, 150, 40), "text": "column stub"}],
                    "B": [{"cross": list(_mid(r)), "c": (120, 120, 120)}],
                    "C": [{"line": [*r["a_pt"], *r["b_pt"]], "c": (230, 120, 0), "text": "member, size not given"}]}, note=f"{r['id']} ({view['title'][:40]})")
                r["_rule"] = "R03"
                self.count("left: R03 stub at a column")
                self.fate[r["id"]] = "R03 stub at a column (drawing detail)"
                continue
            self.count("R02")
            if nm and nm["mark"] in self.sched:
                self.effect["R02 A"] = self.effect.get("R02 A", 0) + r["L"] / 1e6 * self.kgm(nm["mark"])
            if r["L"] >= 1500:
                R.case(self.W, "R02", view["sheet"], _crop([r]), {
                    "A": [{"line": [*r["a_pt"], *r["b_pt"]], "c": (40, 150, 40), "text": (nm["mark"] if nm else "?") + " (as the similar member, ASSUMED)"}],
                    "B": [{"line": [*r["a_pt"], *r["b_pt"]], "c": (230, 120, 0), "text": "steel, size not given"}],
                    "C": [{"cross": list(_mid(r)), "c": (120, 120, 120)}]}, note=f"{r['id']} ({view['title'][:40]})")
            r["_rule"] = "R02"
            if R.choice("R02") == "A" and nm:
                r["mark"], r["status"] = nm["mark"], f"R02 A (ASSUMED, like {nm.get('id', nm['mark'])})"
            else:
                self.count("left: R02 unmarked line")
                self.fate[r["id"]] = "R02 unmarked line (no size given)"

    def framed_in(self, view, rows, marked, tol_pt=3.0):
        """R12 (precedent D25): an unmarked line >= 1 m on an elevation / section whose two ends both lie on marked members
        (drawn framing) -> the mark of the nearest marked member of the same direction (level / plumb / sloped)."""
        R = self.R
        cls = lambda r: 0 if min(r["ang"] if "ang" in r else _ang(r), 180 - (r["ang"] if "ang" in r else _ang(r))) < 5 else (
            1 if abs((r["ang"] if "ang" in r else _ang(r)) - 90) < 5 else 2)
        on = lambda p, m: _seg_dist(p, m["a_pt"], m["b_pt"]) <= tol_pt
        for r in rows:
            if r["mark"] or r.get("_rule") or str(r.get("status", "")).startswith("not a member") or (r.get("L") or 0) < 1000:
                continue
            hosts = [[m for m in marked if m is not r and on(p, m)] for p in (r["a_pt"], r["b_pt"])]
            if not all(hosts):
                continue
            same = [m for m in marked if cls(m) == cls(r) and m["mark"] in self.sched]
            nm = min(same, key=lambda m: math.dist(_mid(m), _mid(r)), default=None)
            if nm is None:
                continue
            self.count("R12")
            R.case(self.W, "R12", view["sheet"], _crop([r]), {
                "A": [{"line": [*r["a_pt"], *r["b_pt"]], "c": (40, 150, 40), "text": nm["mark"] + " (ASSUMED)"}],
                "B": [{"cross": list(_mid(r)), "c": (120, 120, 120)}]}, note=f"{r['id']} ({view['title'][:40]}) between "
                f"{hosts[0][0]['mark']} and {hosts[1][0]['mark']}", effect={"B": f"-{r['L'] / 1e6 * self.kgm(nm['mark']):.1f} t"})
            r["_rule"] = "R12"
            if R.choice("R12") == "A":
                r["mark"], r["status"] = nm["mark"], "R12 A (ASSUMED)"
                self.effect["R12 A"] = self.effect.get("R12 A", 0) + r["L"] / 1e6 * self.kgm(nm["mark"])
            else:
                self.count("left: R12 framed-in line")
                self.fate[r["id"]] = "R12 B: framed-in unmarked line"

    def candidates(self, views, kind, bound, datum_levels=None):
        out = []
        for v in views:
            if v["status"] != "read":
                continue
            truss = bool(re.search(r"TRUSS", v["title"], re.I))
            self.unmarked(v, kind, truss)
            lv = [round((float(s) - 4.85) * 1000) for s in v.get("fit", {}).get("levels", []) if re.fullmatch(r"\d+\.\d+", str(s))] if kind == "elev" else []
            for r in v["rows"]:
                mk = r["mark"]
                if not mk:
                    continue
                if mk.startswith("SB") and kind == "plan" and self.R.choice("R07") == "B":
                    self.count("left: R07 stair stringer on a plan")
                    self.fate[r["id"]] = "R07 stair stringer"
                    self.R.case(self.W, "R07", v["sheet"], _crop([r]), {
                        "A": [{"line": [*r["a_pt"], *r["b_pt"]], "c": (40, 150, 40), "text": mk + " level"}],
                        "B": [{"cross": list(_mid(r)), "c": (120, 120, 120)}],
                        "C": [{"line": [*r["a_pt"], *r["b_pt"]], "c": (230, 120, 0), "text": mk + " sloped"}]},
                        note=f"{r['id']} ({v['title'][:40]})", key=f"R07:{v['sheet']}:{v['title'][:20]}")
                    continue
                if mk not in self.sched and re.fullmatch(r"(UB|UC|LC|UA|EA|PFC|RHS|SHS|CHS)\s*\d.*", mk, re.I):
                    import catalogue                              # NAME-R02: a section written in the member's note
                    self.sched[mk] = catalogue.designation(mk)
                if mk not in self.sched and " " not in mk and not re.search(r"\d", mk):
                    # NAME-R02 (the curator 2026-11-23): a member named in words with no tag or size anywhere is still steel:
                    # the section of the most similar marked member in its view (by scale), ASSUMED
                    nm = similar(r, [q for q in v["rows"] if q is not r and q.get("mark") in self.sched])
                    if nm:
                        r["status"] = f"NAME-R02 {mk} sized as {nm['mark']} (most similar member, ASSUMED)"
                        self.count("NAME-R02 scaled")
                        mk = r["mark"] = nm["mark"]
                if mk not in self.sched:
                    d = self.left["no section for the mark"]
                    d[mk] = d.get(mk, 0) + 1
                    self.fate[r["id"]] = f"{mk}: no section in the schedule"
                    continue
                pf = self.profile(self.sched[mk])
                h = pf.get("h", pf.get("b", 200))
                a, b = list(r["a"]), list(r["b"])
                rng = None
                if kind == "plan":
                    zt = v["z"]
                    if zt is None:
                        lvls = [round((float(x) - self.datum) * 1000) for x in V.LEVEL_RE.findall(v["title"])]
                        zt, rng = max(lvls), (min(lvls), max(lvls))
                        self.count("R05")
                    a[2] = b[2] = zt - h / 2
                elif r.get("kind") != "outline":
                    if abs(a[2] - b[2]) < 60 and math.dist(a, b) > 300 and any(abs((a[2] + b[2]) / 2 - z) < LEVEL_TOL for z in lv):
                        a[2] = b[2] = round((a[2] + b[2]) / 2) - h / 2
                if not all(bound[0] <= p[0] <= bound[1] and bound[2] <= p[1] <= bound[3] and bound[4] <= p[2] <= bound[5] for p in (a, b)):
                    self.left["outside the building"] += 1
                    self.fate[r["id"]] = "placed outside the building"
                    continue
                out.append({"mark": mk, "a": a, "b": b, "kind": kind, "outline": r.get("kind") == "outline", "truss": truss,
                            "src": f"p{v['sheet']} {v['title'][:48]}" + (f" ({v['where']})" if v.get("where") else ""),
                            "rid": r["id"], "range": rng, "loff": r.get("label_off") or 99, "sheet": v["sheet"],
                            "a_pt": r["a_pt"], "b_pt": r["b_pt"], "view": v["title"], "where": v.get("where", ""),
                            "tat": [v.get("axis"), v.get("at")] if truss else None})
        return out

    def level_ranges(self, cand):
        """R05: a member of a level-range plan takes the level of the elevation members it meets."""
        for c in cand:
            if not c["range"]:
                continue
            lo, hi = c["range"]
            zs = [e["a"][2] for e in cand if e["kind"] == "elev" and abs(e["a"][2] - e["b"][2]) < 60 and lo - 400 <= e["a"][2] <= hi + 100
                  and min(math.hypot(p[0] - q[0], p[1] - q[1]) for p in (c["a"], c["b"]) for q in (e["a"], e["b"])) < 400]
            if zs and self.R.choice("R05") == "B":
                c["a"][2] = c["b"][2] = max(set(round(z) for z in zs), key=zs.count)

    def merge(self, cand):
        """R06 / R09: one member drawn in several views -> one member."""
        def same(p, q):
            a, b, c, d = (np.array(p["a"]), np.array(p["b"]), np.array(q["a"]), np.array(q["b"]))
            u, v_ = b - a, d - c
            Lu, Lv = np.linalg.norm(u), np.linalg.norm(v_)
            if Lu < 1 or Lv < 1:
                return False
            u, v_ = u / Lu, v_ / Lv
            if abs(abs(u @ v_) - 1) > 0.0015:
                return False
            level = abs(u[2]) < 0.01 and abs(v_[2]) < 0.01
            pe = level and p["kind"] != q["kind"] and p["mark"] == q["mark"] and abs(a[2] - c[2]) < 350
            tol, ztol = (PLAN_ELEV_OFF, 350) if pe else (160, 160)
            if p["mark"] != q["mark"]:
                tol, ztol = 50, 50
            for x in (c, d):
                w = x - a
                r = w - (w @ u) * u
                if np.hypot(r[0], r[1]) > tol or abs(r[2]) > ztol:
                    return False
            s = sorted([(c - a) @ u, (d - a) @ u])
            return min(Lu, s[1]) - max(0, s[0]) > 0.3 * min(Lu, Lv)
        par = list(range(len(cand)))

        def f(i):
            while par[i] != i:
                par[i] = par[par[i]]; i = par[i]
            return i
        buckets = {}
        for i, c in enumerate(cand):
            d = np.array(c["b"]) - np.array(c["a"]); d /= (np.linalg.norm(d) or 1)
            if d[0] < 0 or (d[0] == 0 and d[1] < 0) or (d[0] == 0 and d[1] == 0 and d[2] < 0):
                d = -d
            buckets.setdefault(tuple(np.round(d * 20).astype(int)), []).append(i)
        keys = list(buckets)
        for k in keys:
            near = [i for kk in keys if max(abs(kk[j] - k[j]) for j in range(3)) <= 1 for i in buckets[kk]]
            for i in buckets[k]:
                for j in near:
                    if j > i and f(i) != f(j) and same(cand[i], cand[j]):
                        par[f(j)] = f(i)
        grp = {}
        for i in range(len(cand)):
            grp.setdefault(f(i), []).append(cand[i])
        out, todo = [], list(grp.values())
        extra = todo                                        # R09 B: the other mark's copies become their own members
        while todo:
            g_ = todo.pop(0)
            marks = [c["mark"] for c in g_]
            mk = max(set(marks), key=marks.count)
            if len(set(marks)) > 1:
                best = min(g_, key=lambda c: c["loff"])
                other = next(c for c in g_ if c["mark"] != best["mark"])
                self.count("R09")
                self.R.case(self.W, "R09", best["sheet"], _crop([best]), {
                    "A": [{"line": [*best["a_pt"], *best["b_pt"]], "c": (40, 150, 40), "text": f"{best['mark']} only"}],
                    "B": [{"line": [*best["a_pt"], *best["b_pt"]], "c": (230, 120, 0), "text": f"{best['mark']} + {other['mark']}"}]},
                    note=f"{best['rid']} {best['mark']} vs {other['rid']} {other['mark']}")
                if self.R.choice("R09") == "A":
                    mk = best["mark"]
                else:
                    extra += [[c] for c in g_ if c["mark"] != mk]
                for c in g_:
                    if c["mark"] != mk and self.R.choice("R09") == "A":
                        self.fate[c["rid"]] = f"R09 A: one line, drawn {mk} in another view"
                g_ = [c for c in g_ if c["mark"] == mk]
            if len(g_) > 1:
                self.count("R06")
            base = max(g_, key=lambda c: (c["kind"] == "plan", math.dist(c["a"], c["b"])))
            a, b = np.array(base["a"], float), np.array(base["b"], float)
            ev = [c for c in g_ if c["kind"] == "elev"]
            if ev and base["kind"] == "plan" and abs(a[2] - b[2]) < 1:
                a[2] = b[2] = sorted(c["a"][2] for c in ev)[len(ev) // 2]
            u = (b - a) / np.linalg.norm(b - a)
            ts = [float((np.array(p) - a) @ u) for c in g_ for p in (c["a"], c["b"])]
            pick = self.R.choice("R06")
            if pick in ("B", "C") and len(g_) > 1:
                k_ = "elev" if pick == "B" else "plan"
                ts = [float((np.array(p) - a) @ u) for c in g_ if c["kind"] == k_ for p in (c["a"], c["b"])] or ts
            lo, hi = min(ts), max(ts)
            out.append({"mark": mk, "a": [round(float(v), 1) for v in a + lo * u], "b": [round(float(v), 1) for v in a + hi * u],
                        "src": "; ".join(sorted({c["src"] for c in g_}))[:300], "views": len(g_), "ids": [c["rid"] for c in g_],
                        "truss": next((c["where"] or c["view"] for c in g_ if c["truss"]), None), "outline": any(c["outline"] for c in g_),
                        "tview": next((f"p{c['sheet']} {c['view']}" for c in g_ if c["truss"]), None),
                        "tat": next((c["tat"] for c in g_ if c["truss"]), None)})
        return out


def columns(cols_layout, members, bos_z, sched, profile, rulings, work):
    """Layout columns + the vertical members read on their lines (elevations) -> columns with bottom / top.
    The vertical members on a column line are the column itself (removed from members)."""
    out, used = [], set()
    for c in cols_layout:
        on = [i for i, m in enumerate(members) if i not in used and abs(m["a"][0] - m["b"][0]) < 60 and abs(m["a"][1] - m["b"][1]) < 60
              and math.hypot(m["a"][0] - c["x"], m["a"][1] - c["y"]) < COL_TOL]
        zs = [z for i in on for z in (members[i]["a"][2], members[i]["b"][2])]
        used.update(on)
        top = max(zs) if zs else None
        out.append({"mark": c["mark"], "bp": c.get("bp"), "x": c["x"], "y": c["y"], "z0": bos_z, "z1": top, "depth_dir": c["depth_dir"],
                    "ids": [x for i in on for x in members[i]["ids"]],
                    "src": f"p1 column layout ({c['mark']}" + (f" / {c['bp']}" if c.get("bp") else "") + ")" +
                           (f"; top from {len(on)} elevation line(s): {members[on[0]]['src'][:60]}" if on else "; top: what it carries (frame)")})
    rest = [m for i, m in enumerate(members) if i not in used]
    extra = []                                               # a column mark drawn vertical away from every layout column
    for m in list(rest):
        if m["mark"].startswith("SC") and abs(m["a"][0] - m["b"][0]) < 60 and abs(m["a"][1] - m["b"][1]) < 60:
            extra.append(m)
            rest.remove(m)
    return out, rest, extra
