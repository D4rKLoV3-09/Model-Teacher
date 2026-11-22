#!/usr/bin/env python3
"""spec.json -> steel parts as convex prisms (SKILL.md section 2 rules).

    python geometry.py SPEC.json PARTS.json

Spec (what Claude writes after reading the drawings):
{
  "profiles": {"UC 152x152x23": {"type": "I", "h": 152.4, "b": 152.2, "tw": 5.8, "tf": 6.8, "kgm": 23.0}, ...},
      types: I (h b tw tf) | SHS (b t) | RHS (h b t, h along up) | CHS (d t; 16 flat wall segments) | LIPC lipped channel open to +u (h b t lip) | PL flat (b t)
  "members": [{
      "id": "TC-N", "t": "X-UB42-TC", "prof": "UC 203x203x46",
      "a": [x, y, z], "b": [x, y, z],        work points in mm (centre line runs WP to WP)
      "up": [0, 0, 1],                        section depth direction (default: global Z, or X for vertical members)
      "rank": 1,                              lower rank = host; used by "auto" ends
      "ends": ["auto", [{"host": "VR-1", "face": "far"}]],
      "arc": {"mid": [x, y, z]} | {"c": [x, y, z]},  optional: ONE bent-and-rolled member from a to b on a circle through
                                              "mid" (any point of the arc between a and b) or about centre "c" (the
                                              shorter way round); the part carries "arc" (centre, radius, normal, sweep)
      ...any other keys (layer, unit, src, grid, ...) are copied to the part unchanged
  }]
}
End at a / b:
  null                square cut at the work point
  "auto"  (default)   bear on the "near" face of every lower-rank member whose centre line passes
                      within 1 mm of the work point (diagonal at a node -> chord + vertical = true notch)
  [cut, ...]          explicit: {"host": id, "face": "near"|"far"|"+u"|"-u"|"+v"|"-v"|"web"|"+w"|"-w"|"cope"} or
                      {"plane": [nx, ny, nz, d]} (plane n.x = d); the member keeps the side its body is on
                      add "only": "web" | "flanges" | [plate indices] to cut only some plates (copes);
                      "cope" on an I host (member depth parallel to the host's): the web is split at the host's
                      inner flange faces, the band between them runs to the host web, the rest stops at the
                      flange tips (true notch, no web inside a host flange);
                      plate order: I = +u flange, -u flange, web; SHS = +u, -u, +v, -v walls;
                      LIPC = base, +v side, +v lip, -v side, -v lip
Bondek package: always the same three standard models, whatever the product, thickness or stud length (only the
quantity matters; the description carries the spec). Profiles "BONDEK", "SHEAR STUD" and "EDGEFORM" are built in:
  {"id": "D1", "t": "BDK1", "prof": "BONDEK", "a": [..], "b": [..], "wid": 3000, "desc": "Bondek II 1.0 BMT"}
        a -> b = centre line of the deck strip along the span, on top of steel; "wid" = covered width  (MTO: m2)
  {"id": "S1", "t": "SS1", "prof": "SHEAR STUD", "a": [..], "b": [..], "spacing": 300, "desc": "Shear stud 19x100"}
        a -> b = the beam's top-flange centre line; one stud part every "spacing" (default 300)       (MTO: pcs)
  {"id": "E1", "t": "EF1", "prof": "EDGEFORM", "a": [..], "b": [..], "desc": "Edgeform 1.2 BMT x 150"}
        a -> b = slab edge on top of steel                                                             (MTO: m)
Bolted joints (detail given on the drawings): spec["joints"], one line per joint (base / endplate / splice / fin /
cleat), expanded by joints.py into plates + bolt sets after the members; a typical detail is defined once in
spec["conn_types"] and each joint line only names its place; see its docstring. No joint line = welded.
"assumed": true marks the description as ASSUMED in the MTO. The "mto" override is filled in automatically.
Rolled members ("arc"): one part, drawn as short straight pieces mitred on the bisecting planes (continuous, no gap,
no overlap); ends are cut square to the tangent, or by the end cuts above with the tangent as the direction into the
body; L and kg are measured along the curved centre line. A plan curve keeps its depth vertical (bent about its weak
axis); an arc in an elevation keeps its depth in the arc plane (bent about its strong axis), unless "up" says otherwise.
Joints at generation (nodes.py, on by default; "connect": false in the spec turns it off): every end is run on to
its support with the work point on the support's centre line, chevrons meet at one point, members drawn through a
column are split, label breaks joined. A cantilever the drawing shows: "free": [true, false] + "free_why".
Welded = zero gap: each section plate is clipped by every bearing plane (half-space clip, capped),
never by moving vertices, so a corner end comes out as a true notch.
"""
import json
import math
import sys

import numpy as np

EPS = 1e-6
EXTEND = 3000.0      # mm past the work point before an end is clipped back
AUTO_TOL = 1.0       # mm: work point to host centre line for "auto" ends


# ---------------- section plates (u = depth direction, v = width), centred on the section box ----------------
def plates(pf):
    t = pf["type"]
    if t == "I":
        h, b, tw, tf = pf["h"], pf["b"], pf["tw"], pf["tf"]
        return [(h / 2 - tf, h / 2, -b / 2, b / 2), (-h / 2, -h / 2 + tf, -b / 2, b / 2),
                (-h / 2 + tf, h / 2 - tf, -tw / 2, tw / 2)]
    if t == "SHS":
        b, s = pf["b"], pf["t"]
        return [(b / 2 - s, b / 2, -b / 2, b / 2), (-b / 2, -b / 2 + s, -b / 2, b / 2),
                (-b / 2 + s, b / 2 - s, b / 2 - s, b / 2), (-b / 2 + s, b / 2 - s, -b / 2, -b / 2 + s)]
    if t == "RHS":                       # h along u (depth), b along v
        h, b, s = pf["h"], pf["b"], pf["t"]
        return [(h / 2 - s, h / 2, -b / 2, b / 2), (-h / 2, -h / 2 + s, -b / 2, b / 2),
                (-h / 2 + s, h / 2 - s, b / 2 - s, b / 2), (-h / 2 + s, h / 2 - s, -b / 2, -b / 2 + s)]
    if t == "LIPC":
        h, b, s, lip = pf["h"], pf["b"], pf["t"], pf["lip"]
        return [(-h / 2, -h / 2 + s, -b / 2, b / 2),
                (-h / 2 + s, h / 2, b / 2 - s, b / 2), (h / 2 - s, h / 2, b / 2 - s - lip, b / 2 - s),
                (-h / 2 + s, h / 2, -b / 2, -b / 2 + s), (h / 2 - s, h / 2, -b / 2 + s, -b / 2 + s + lip)]
    if t == "PL":
        return [(-pf["t"] / 2, pf["t"] / 2, -pf["b"] / 2, pf["b"] / 2)]
    if t == "L":                         # equal / unequal angle: leg h along u, leg b along v, heel at (-u, -v)
        h, b, s = pf["h"], pf["b"], pf["t"]
        return [(-h / 2, h / 2, -b / 2, -b / 2 + s), (-h / 2, -h / 2 + s, -b / 2 + s, b / 2)]
    if t == "C":                         # hot-rolled channel (PFC / TFC): web at -v, flanges open to +v
        h, b, tw, tf = pf["h"], pf["b"], pf["tw"], pf["tf"]
        return [(-h / 2, h / 2, -b / 2, -b / 2 + tw), (h / 2 - tf, h / 2, -b / 2 + tw, b / 2),
                (-h / 2, -h / 2 + tf, -b / 2 + tw, b / 2)]
    if t == "CHS":
        return []                        # round: wall segments from chs_walls(), not u/v rectangles
    if t == "DECK":       # standard deck model: 2 mm pan + 30 x 55 ribs at 200 centres, sitting on the a-b line
        w = pf["b"]
        n = max(1, int(w // DECK_PITCH))
        ribs = [(-w / 2 + (w - (n - 1) * DECK_PITCH) / 2 + k * DECK_PITCH) for k in range(n)]
        return [(0.0, 2.0, -w / 2, w / 2)] + [(2.0, DECK_RIB_H, v - 15.0, v + 15.0) for v in ribs]
    if t == "EDGE":       # standard edgeform model: 150 upstand + 50 foot, 3 mm
        return [(0.0, EDGE_H, -1.5, 1.5), (0.0, 3.0, 1.5, 50.0)]
    raise ValueError(f"unknown profile type {t}")


def chs_walls(pf, n=16):
    """CHS wall as n convex quads in (u, v): outer corners on the outside diameter."""
    ro, ri = pf["d"] / 2, pf["d"] / 2 - pf["t"]
    c = [(np.cos(2 * np.pi * i / n), np.sin(2 * np.pi * i / n)) for i in range(n + 1)]
    return [[(ri * p[0], ri * p[1]), (ro * p[0], ro * p[1]), (ro * q[0], ro * q[1]), (ri * q[0], ri * q[1])] for p, q in zip(c, c[1:])]


DECK_PITCH, DECK_RIB_H, EDGE_H = 200.0, 55.0, 150.0
STUD = {"shank": 19.0, "head": 32.0, "head_t": 10.0, "len": 100.0, "collar": 24.0, "collar_h": 4.5}   # one model for every stud (19 x 100, weld collar d1 x h)
STD_PROFILES = {"BONDEK": {"type": "DECK", "kgm": 0.0}, "SHEAR STUD": {"type": "STUD", "kgm": 0.0},
                "EDGEFORM": {"type": "EDGE", "kgm": 0.0}}
STD_MTO = {"DECK": ("m2", "Deck", "Bondek II 1.0 BMT"), "STUD": ("pcs", "Stud", "Shear stud 19x100"),
           "EDGE": ("m", "Edge", "Edgeform 1.2 BMT")}


def std_mto(m, pf):
    """Bondek package parts are counted (m2 / pcs / m), not weighed: fill the MTO override from the member."""
    uom, cat, default = STD_MTO[pf["type"]]
    desc = m.get("desc") or default
    mto = {"kind": "measure", "uom": uom, "cat": cat, "desc": f"{desc} ({uom})"}
    if pf["type"] == "DECK":
        mto["wid"] = m["wid"]
    if m.get("assumed"):
        mto["assumed"] = ["desc"]
    return dict(mto, **(m.get("mto") or {}))


def round_prism(p, z, e1, e2, s0, s1, r, n=12):
    """Round bar as an n-sided convex prism between s0 and s1 along z. Faces outward."""
    ring = [(np.cos(2 * np.pi * k / n) * e1 + np.sin(2 * np.pi * k / n) * e2) * r for k in range(n)]
    bot, top = [p + s0 * z + q for q in ring], [p + s1 * z + q for q in ring]
    faces = [bot[::-1], top] + [[bot[k], bot[(k + 1) % n], top[(k + 1) % n], top[k]] for k in range(n)]
    ctr = p + (s0 + s1) / 2 * z
    out = []
    for f in faces:
        nrm = np.cross(f[1] - f[0], f[2] - f[0])
        out.append(f if nrm @ (np.mean(f, axis=0) - ctr) > 0 else f[::-1])
    return out


def std_part(m, kind):
    """Bondek deck / stud row as ONE compact part: a bounding box for picking, clash and extents, plus "std" - the
    viewer draws the standard model from it (instanced: Bondek II section per 200 mm rib, stud with weld collar,
    shank and head at the spacing). Keeps big buildings light: no per-stud or per-rib prisms in the page."""
    a, b, x, u, v = m["_f"]
    L = float(np.linalg.norm(b - a))
    if kind == "DECK":
        w = float(m["wid"])
        r = (0.0, DECK_RIB_H, -w / 2, w / 2)
        std = {"k": "deck", "w": round(w, 1)}
        n = None
    elif kind == "EDGE":                              # edgeform: the upstand as one thin box (the 50 foot is under the slab)
        r = (0.0, EDGE_H, -1.5, 1.5)
        std = {"k": "edge"}
        n = None
    else:
        sp = float(m.get("spacing", 300))
        n = max(1, int(L // sp))
        r = (0.0, STUD["len"], -STUD["head"] / 2, STUD["head"] / 2)
        s0 = (L - (n - 1) * sp) / 2
        std = {"k": "stud", "n": n, "sp": sp, "s0": round(s0, 1)}
    std.update({"o": [round(float(c), 1) for c in a], "x": [round(float(c), 5) for c in x], "u": [round(float(c), 5) for c in u]})
    part = {k: v_ for k, v_ in m.items() if not k.startswith("_") and k not in ("a", "b", "up", "ends", "rank", "id", "spacing", "assumed", "wid")}
    part.update({"ax": [round(float(c), 5) for c in x], "L": round(L, 1), "kg": 0.0, "std": std,
                 "pr": [[[[round(float(c), 1) for c in q] for q in f] for f in box(a, x, u, v, 0.0, L, r)]]})
    if n is not None:
        part["mto"] = dict(part.get("mto") or {}, set=n, len=STUD["len"])
    part.setdefault("desc", m["prof"])
    part.setdefault("plate", None)
    return part


def stud_parts(m):
    """One stud row -> one part per stud, each the standard stud (round shank + round head, as the sample), standing on the a-b line.
    (Kept for small models that want real stud prisms; build() uses std_part.)"""
    a, b = np.asarray(m["a"], float), np.asarray(m["b"], float)
    L = np.linalg.norm(b - a)
    sp = float(m.get("spacing", 300))
    n = max(1, int(L // sp))
    along = (b - a) / L if L > EPS else np.array([1.0, 0, 0])
    z = np.array([0, 0, 1.0])
    side = np.cross(z, along)
    side = side / np.linalg.norm(side) if np.linalg.norm(side) > 1e-3 else np.array([0, 1.0, 0])
    fwd = np.cross(side, z)
    base = {k: v for k, v in m.items() if not k.startswith("_") and k not in ("a", "b", "up", "ends", "rank", "id", "spacing", "assumed", "wid")}
    out = []
    for k in range(n):
        p = a + along * ((L - (n - 1) * sp) / 2 + k * sp)
        hs, hh = STUD["shank"] / 2, STUD["head"] / 2
        prisms = [round_prism(p, z, fwd, side, 0.0, STUD["len"] - STUD["head_t"], hs),
                  round_prism(p, z, fwd, side, STUD["len"] - STUD["head_t"], STUD["len"], hh)]
        part = dict(base)
        part.update({"ax": [0.0, 0.0, 1.0], "L": STUD["len"], "kg": 0.0,
                     "pr": [[[[round(float(c), 1) for c in q] for q in f] for f in pr] for pr in prisms]})
        part.setdefault("desc", "Shear stud 19x100")
        part.setdefault("plate", None)
        out.append(part)
    return out


def half_dims(pf):
    """Half depth (u) and half width (v) of the section envelope."""
    if pf["type"] == "CHS":
        return pf["d"] / 2, pf["d"] / 2
    if pf["type"] == "SHS":
        return pf["b"] / 2, pf["b"] / 2
    if pf["type"] == "PL":
        return pf["t"] / 2, pf["b"] / 2
    if pf["type"] in STD_MTO:
        return DECK_RIB_H / 2, pf.get("b", 50.0) / 2
    return pf["h"] / 2, pf["b"] / 2


# ---------------- member frame ----------------
def frame(m):
    a, b = np.asarray(m["a"], float), np.asarray(m["b"], float)
    x = b - a
    L = np.linalg.norm(x)
    if L < EPS:
        raise ValueError(f"{m['id']}: zero-length member")
    x /= L
    up = np.asarray(m.get("up") or ([1.0, 0, 0] if abs(x[2]) > 0.9 else [0, 0, 1.0]), float)
    u = up - (up @ x) * x
    if np.linalg.norm(u) < 1e-3:
        raise ValueError(f"{m['id']}: 'up' is parallel to the member axis")
    u /= np.linalg.norm(u)
    return a, b, x, u, np.cross(x, u)


# ---------------- convex polyhedron clipping ----------------
def box(a, x, u, v, s0, s1, r):
    """Prism of one section plate r=(u0,u1,v0,v1) between s0 and s1 along x. Faces CCW seen from outside."""
    u0, u1, v0, v1 = r
    return prism(a, x, u, v, s0, s1, [(u0, v0), (u1, v0), (u1, v1), (u0, v1)])


def prism(a, x, u, v, s0, s1, poly):
    """Prism of a convex section polygon [(u, v), ...] between s0 and s1 along x."""
    P = lambda s, uu, vv: a + s * x + uu * u + vv * v
    k = len(poly)
    c = [P(s0, *p) for p in poly] + [P(s1, *p) for p in poly]
    idx = [(0, *range(k - 1, 0, -1)), tuple(range(k, 2 * k))] + [(i, (i + 1) % k, (i + 1) % k + k, i + k) for i in range(k)]
    faces = [[c[i] for i in f] for f in idx]
    # make every face outward (x, u, v may be a left-handed triad only if v was flipped by caller)
    ctr = np.mean(c, axis=0)
    out = []
    for f in faces:
        n = np.cross(f[1] - f[0], f[2] - f[0])
        out.append(f if n @ (np.mean(f, axis=0) - ctr) > 0 else f[::-1])
    return out


def clip(faces, n, d):
    """Keep n.x <= d. Returns the clipped convex polyhedron (faces) with a capping face, or [] if nothing left."""
    out, cap = [], []
    for poly in faces:
        new = []
        k = len(poly)
        for i in range(k):
            P, Q = poly[i], poly[(i + 1) % k]
            dp, dq = n @ P - d, n @ Q - d
            if dp <= EPS:
                new.append(P)
                if dp >= -EPS:
                    cap.append(P)
            if (dp < -EPS < EPS < dq) or (dq < -EPS < EPS < dp):
                X = P + (Q - P) * (dp / (dp - dq))
                new.append(X)
                cap.append(X)
        if len(new) >= 3:
            out.append(new)
    if not out:
        return []
    uniq = []
    for p in cap:
        if all(np.linalg.norm(p - q) > 1e-4 for q in uniq):
            uniq.append(p)
    if len(uniq) >= 3:
        c = np.mean(uniq, axis=0)
        e1 = uniq[0] - c
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)       # CCW about +n (outward normal of the cap)
        uniq.sort(key=lambda p: np.arctan2((p - c) @ e2, (p - c) @ e1))
        out = [f for f in out if not all(abs(n @ p - d) <= 1e-4 for p in f)]  # drop old faces lying in the cap
        out.append(uniq)
    return out


# ---------------- end cuts ----------------
def dist_point_segment(p, a, b):
    ab = b - a
    t = np.clip((p - a) @ ab / (ab @ ab), 0, 1)
    return np.linalg.norm(p - (a + t * ab))


def keep_side(n, d, body):
    """Orient plane n.x = d so the member body point is on the kept side n.x <= d."""
    return (-n, -d) if n @ body > d else (n, d)


def local_frame(m, at):
    """Frame (a, b, x, u, v) of a member where `at` meets it: the member's own frame, or for a rolled member the
    tangent frame at the point of its curve nearest to `at` (so a beam framing into a curved beam is cut on the
    curve, not on its chord)."""
    A = m.get("_arc")
    if not A or at is None:
        return m["_f"]
    a0 = np.asarray(m["a"], float)
    w0 = a0 - A["c"]
    q = np.asarray(at, float) - A["c"]
    q -= (q @ A["n"]) * A["n"]
    phi = math.atan2(np.cross(w0, q) @ A["n"], w0 @ q) % (2 * math.pi)
    if phi > A["th"]:                                     # beyond an end: the nearer end
        phi = A["th"] if phi - A["th"] < 2 * math.pi - phi else 0.0
    p = arc_point(A, a0, phi)
    x = np.cross(A["n"], p - A["c"]); x /= np.linalg.norm(x)
    u = _rolled_up(m, A, p, x)
    return p, p + x, x, u, np.cross(x, u)


def _rolled_up(m, A, p, x):
    """Depth direction of a rolled member at curve point p with tangent x (see build_rolled)."""
    if m.get("up"):
        up = np.asarray(m["up"], float)
    elif abs(A["n"][2]) > 0.7:                             # plan curve: depth vertical (along the arc normal)
        up = A["n"] if A["n"][2] > 0 else -A["n"]
    else:                                                  # elevation arc: depth in the plane
        up = A["c"] - p
        up = up if up[2] >= 0 or abs(up[2]) < 1e-9 else -up
    u = up - (up @ x) * x
    return u / np.linalg.norm(u)


def axis_dist(p, m):
    """Distance from point p to a member's centre line (its curve for a rolled member)."""
    A = m.get("_arc")
    if not A:
        return dist_point_segment(p, m["_f"][0], m["_f"][1])
    a, *_ = local_frame(m, p)
    return float(np.linalg.norm(np.asarray(p, float) - a))


def host_plane(host, face, into, at=None):
    """Plane (n, d) of one face of the host section (u = depth, v = width).
    face: near | far | +u | -u | +v | -v (envelope), web | +w | -w (web faces of an I host).
    at = the member end that bears on it (a rolled host is taken at the point of its curve nearest to it)."""
    a, _, x, u, v = local_frame(host, at)
    pf = host["_pf"]
    hu, hv = half_dims(pf)
    if face in ("web", "+w", "-w"):
        if pf["type"] != "I":
            raise ValueError(f"{host['id']}: web face asked on a {pf['type']} host")
        nrm = (v if v @ into >= 0 else -v) if face == "web" else (v if face == "+w" else -v)
        return nrm, nrm @ a + pf["tw"] / 2
    cands = [(u, hu, "+u"), (-u, hu, "-u"), (v, hv, "+v"), (-v, hv, "-v")]
    if face in ("near", "far"):
        near = max(cands, key=lambda c: c[0] @ into)       # outward normal pointing most toward the member body
        nrm = near[0] if face == "near" else -near[0]
        off = near[1]                                     # opposite faces share the same offset
    else:
        nrm, off = [(c[0], c[1]) for c in cands if c[2] == face][0]
    return nrm, nrm @ a + off                             # host centre line is a + s*x, nrm is perpendicular to x


def expand(c):
    """'cope' into an I host's web: flanges stop at the host flange tips, web runs to the host web face."""
    if c.get("face") != "cope":
        return [c]
    return [dict(c, face="near", only="flanges"), dict(c, face="web", only="web")]


def cope_band(m, host, at=None):
    """u-range of the member web that lies between the host's flanges (member u parallel to host u),
    or None. Outside it the web would run into a host flange, so it stops at the flange tips."""
    pf = host["_pf"]
    if pf["type"] != "I" or m["_pf"]["type"] != "I":
        return None
    a, _, _, u, _ = m["_f"]
    ha, _, _, hu, _ = local_frame(host, at)
    k = u @ hu
    if abs(k) < 0.999:
        return None
    inner = pf["h"] / 2 - pf["tf"]
    lo, hi = sorted(((hu @ ha + s * inner) - hu @ a) / k for s in (-1, 1))
    return lo, hi


def end_cuts(m, which, by_id, members):
    spec = (m.get("ends") or ["auto", "auto"])[which]
    a, b, x, _, _ = m["_f"]
    end_pt, into = (a, x) if which == 0 else (b, -x)
    body = (a + b) / 2                                    # the member body is on the kept side of every cut
    if m.get("_arc"):                                     # rolled member: into the body along the tangent
        into = m["_arc"]["t0"] if which == 0 else -m["_arc"]["t1"]
        body = m["_arc"]["mid"]                           # middle of the curve: always on the body side
    if spec is None:
        return []
    if spec == "auto":
        rank = m.get("rank", 9)
        spec = [{"host": h["id"], "face": "near"} for h in members
                if h is not m and h.get("rank", 9) < rank
                and axis_dist(end_pt, h) <= AUTO_TOL]
    planes = []
    for c0 in spec:
        band = cope_band(m, by_id[c0["host"]], end_pt) if c0.get("face") == "cope" else None
        near = keep_side(*host_plane(by_id[c0["host"]], "near", into, end_pt), body) if band else None
        for c in expand(c0):
            if "plane" in c:
                n, d = np.asarray(c["plane"][:3], float), float(c["plane"][3])
            else:
                n, d = host_plane(by_id[c["host"]], c["face"], into, end_pt)
            n, d = keep_side(n, d, body)
            only = c.get("only")
            if only == "web":
                only = [2]
            elif only == "flanges":
                only = [0, 1]
            # cope web: only the band between the host flanges runs to the host web, the rest stops at the flange tips
            split = (band, near) if band and c.get("face") == "web" else None
            planes.append((n, d, None if only is None else set(only), split))
    return planes


# ---------------- build ----------------
def split_u(r, ts):
    """Split plate rectangle r = (u0, u1, v0, v1) at the u values ts that fall inside it."""
    us = sorted({r[0], r[1]} | {t for t in ts if r[0] < t < r[1]})
    return [(p, q, r[2], r[3]) for p, q in zip(us, us[1:]) if q - p > EPS]


def arc_of(m):
    """Circle of a rolled member: centre, radius, plane normal, start / end tangents, sweep (rad)."""
    a, b = np.asarray(m["a"], float), np.asarray(m["b"], float)
    spec = m["arc"]
    if "mid" in spec:
        p = np.asarray(spec["mid"], float)
        n = np.cross(p - a, b - a)
        if np.linalg.norm(n) < 1e-6:
            raise ValueError(f"{m['id']}: arc mid point lies on the chord")
        n /= np.linalg.norm(n)
        # centre: equidistant from a, p, b in their plane
        A = np.array([p - a, b - a, n]); rhs = np.array([(p @ p - a @ a) / 2, (b @ b - a @ a) / 2, n @ a])
        c = np.linalg.solve(A, rhs)
        n_ = np.cross(a - c, p - c)                       # sweep direction a -> mid -> b
        n = n_ / np.linalg.norm(n_) if np.linalg.norm(n_) > 1e-9 else n
        ang = lambda q: math.atan2(np.cross(a - c, q - c) @ n, (a - c) @ (q - c))
        th = ang(b) % (2 * math.pi)
        if not 0 < ang(p) % (2 * math.pi) < th:
            n, th = -n, (2 * math.pi - th)
    else:
        c = np.asarray(spec["c"], float)
        n = np.cross(a - c, b - c)
        if np.linalg.norm(n) < 1e-6:
            raise ValueError(f"{m['id']}: arc centre on the chord (half circle): give 'mid' instead")
        n /= np.linalg.norm(n)
        th = math.atan2(np.linalg.norm(np.cross(a - c, b - c)), (a - c) @ (b - c))
    r = float(np.linalg.norm(a - c))
    if abs(np.linalg.norm(b - c) - r) > max(1.0, 1e-3 * r):
        raise ValueError(f"{m['id']}: a and b are not on one circle about the arc centre")
    t0 = np.cross(n, a - c); t0 /= np.linalg.norm(t0)
    t1 = np.cross(n, b - c); t1 /= np.linalg.norm(t1)
    A = {"c": c, "r": r, "n": n, "th": th, "t0": t0, "t1": t1, "chord": float(np.linalg.norm(b - a))}
    A["mid"] = arc_point(A, a, th / 2)
    return A


def arc_point(A, a, phi):
    """Point at angle phi from a on the member's circle (rotation about the normal)."""
    c, n = A["c"], A["n"]
    w = a - c
    return c + w * math.cos(phi) + np.cross(n, w) * math.sin(phi) + n * (n @ w) * (1 - math.cos(phi))


def _plate_prisms(pf, a, x, u, v, s0, s1, cuts):
    """Every plate (and CHS wall) of one straight run, clipped by cuts [(n, d, only, split)]."""
    prisms = []
    for poly in (chs_walls(pf) if pf["type"] == "CHS" else []):       # round: plain plane cuts, no cope bands
        faces = prism(a, x, u, v, s0, s1, poly)
        for n, d, only, split in cuts:
            faces = clip(faces, *(split[1] if split else (n, d))) if faces else faces
        if faces:
            prisms.append(faces)
    for k, r in enumerate(plates(pf)):
        bands = [r]
        for *_, only, split in cuts:
            if split and (only is None or k in only):
                (lo, hi), _ = split
                bands = [nb for b in bands for nb in split_u(b, (lo, hi))]
        for br in bands:
            faces = box(a, x, u, v, s0, s1, br)
            for n, d, only, split in cuts:
                if only is not None and k not in only:
                    continue
                if split:
                    (lo, hi), near = split
                    if not (lo - EPS <= br[0] and br[1] <= hi + EPS):
                        n, d = near
                faces = clip(faces, n, d)
                if not faces:
                    break
            if faces:
                prisms.append(faces)
    return prisms


def build_rolled(m, by_id, members):
    """ONE rolled part: pieces along the arc, each mitred on the bisecting plane of its joints (tangent normal)."""
    pf, A = m["_pf"], m["_arc"]
    a = np.asarray(m["a"], float)
    seg = max(4, min(72, int(math.ceil(math.degrees(A["th"]) / 2.5)), int(math.ceil(A["r"] * A["th"] / 150.0))))
    seg = min(seg, 72)
    phis = [A["th"] * k / seg for k in range(seg + 1)]
    P = [arc_point(A, a, f) for f in phis]
    T = [np.cross(A["n"], p - A["c"]) / A["r"] for p in P]           # tangents (direction of travel)
    cuts = [end_cuts(m, 0, by_id, members), end_cuts(m, 1, by_id, members)]
    reach = 4 * max(half_dims(pf)) + 600.0                 # mm along the curve that an end cut can still reach
    prisms, L = [], 0.0
    for k in range(seg):
        p, q = P[k], P[k + 1]
        x = (q - p) / np.linalg.norm(q - p)
        u = _rolled_up(m, A, (p + q) / 2, x)
        v = np.cross(x, u)
        ln = float(np.linalg.norm(q - p))
        ext = 0.5 * ln + 400.0
        cs = []
        if k > 0:                                          # keep the part beyond the joint at P[k]
            cs.append((-T[k], float(-T[k] @ P[k]), None, None))
        elif not cuts[0]:
            cs.append((-T[0], float(-T[0] @ P[0]), None, None))   # square to the tangent at a
        if k < seg - 1:
            cs.append((T[k + 1], float(T[k + 1] @ P[k + 1]), None, None))
        elif not cuts[1]:
            cs.append((T[-1], float(T[-1] @ P[-1]), None, None))
        # an end cut acts on every piece near that end (a member meeting its host at an angle runs past the first
        # short piece before its far flange clears the host)
        if cuts[0] and A["r"] * phis[k] <= reach:
            cs += cuts[0]
        if cuts[1] and A["r"] * (A["th"] - phis[k + 1]) <= reach:
            cs += cuts[1]
        s0 = -(EXTEND if (k == 0 and cuts[0]) else ext)
        s1 = ln + (EXTEND if (k == seg - 1 and cuts[1]) else ext)
        prisms += _plate_prisms(pf, p, x, u, v, s0, s1, cs)
        lo, hi = s0, s1                                    # centre-line length of this piece between its cuts
        for n_, d_, _o, split in cs:
            k_ = float(n_ @ x)
            if abs(k_) < 1e-12:
                continue
            t_ = (d_ - float(n_ @ p)) / k_
            if k_ > 0:
                hi = min(hi, t_)
            else:
                lo = max(lo, t_)
        L += max(0.0, hi - lo)
    return prisms, L


def build_member(m, by_id, members, profiles):
    pf = m["_pf"]
    a, b, x, u, v = m["_f"]
    if m.get("_arc"):
        prisms, cut_len = build_rolled(m, by_id, members)
    else:
        L = np.linalg.norm(b - a)
        cuts = [end_cuts(m, 0, by_id, members), end_cuts(m, 1, by_id, members)]
        s0 = -EXTEND if cuts[0] else 0.0
        s1 = L + EXTEND if cuts[1] else L
        prisms = _plate_prisms(pf, a, x, u, v, s0, s1, cuts[0] + cuts[1])
        cut_len = None
    V = np.array([p for pr in prisms for f in pr for p in f])
    if cut_len is None:
        s = V @ x
        cut_len = float(s.max() - s.min())
    part = {k: val for k, val in m.items() if not k.startswith("_") and k not in ("a", "b", "up", "ends", "rank", "id", "wid", "spacing", "assumed", "arc")}
    if m.get("_arc"):
        A = m["_arc"]
        part["arc"] = {"c": [round(float(c), 1) for c in A["c"]], "r": round(A["r"], 1),
                       "n": [round(float(c), 5) for c in A["n"]], "deg": round(math.degrees(A["th"]), 3),
                       "a": [round(float(c), 1) for c in m["a"]], "b": [round(float(c), 1) for c in m["b"]]}
        part.setdefault("roll", f"bent and rolled, R {A['r']:,.0f}")
    part.update({
        "ax": [round(float(c), 5) for c in x],
        "L": round(cut_len, 1),
        "kg": round(pf["kgm"] * cut_len / 1000.0, 2),
        "pr": [[[[round(float(c), 1) for c in p] for p in f] for f in pr] for pr in prisms],
    })
    part.setdefault("desc", m["prof"])
    part.setdefault("plate", None)
    return part


def build(spec):
    profiles, members = dict(STD_PROFILES, **spec.get("profiles", {})), spec["members"]
    if spec.get("connect", True):                      # the curator's rules at generation time: no floating end, concentric nodes
        import nodes
        spec["_connect"] = nodes.connect(members, profiles)
    by_id = {}
    for m in members:
        if m["id"] in by_id:
            raise ValueError(f"duplicate member id {m['id']}")
        pf = profiles[m["prof"]]
        if pf["type"] in STD_MTO:                      # bondek package: standard model, free end cuts, counted not weighed
            if pf["type"] == "DECK":
                pf = dict(pf, b=float(m["wid"]))
            m["mto"] = std_mto(m, pf)
            m.setdefault("ends", [None, None])
            m.setdefault("rank", 9)
        m["_pf"] = pf
        m["_f"] = frame(m)
        if m.get("arc"):
            m["_arc"] = arc_of(m)
        by_id[m["id"]] = m
    for m in members:
        for e in (m.get("ends") or []):
            for c in (e if isinstance(e, list) else []):
                if "host" in c and c["host"] not in by_id:
                    raise ValueError(f"{m['id']}: unknown host {c['host']}")
    parts, idx = [], {}
    for m in members:
        if m.get("host_only"):
            continue
        t_ = m["_pf"]["type"]
        idx[m["id"]] = len(parts)
        parts.append(std_part(m, t_) if t_ in ("STUD", "DECK", "EDGE") else build_member(m, by_id, members, profiles))
    if spec.get("_connect"):                           # an end cut by several node members may touch none of them: rebuild
        import nodes                                   # it with its main support's cut only (then its near face)
        spec["_connect"]["repaired"] = nodes.contact_repair(members, parts, idx, by_id, lambda m: build_member(m, by_id, members, profiles))
    if spec.get("joints"):                             # bolted joints with their detail given (joints.py)
        import joints
        parts += joints.apply(spec["joints"], parts, idx, by_id, spec.get("conn_types"))
    return parts


if __name__ == "__main__":
    spec = json.load(open(sys.argv[1], encoding="utf-8"))
    parts = build(spec)
    json.dump(parts, open(sys.argv[2], "w", encoding="utf-8"), separators=(",", ":"))
    if spec.get("_connect"):                           # the joined spec goes back to its file: make_data / clash read it
        spec["members"] = [{k: v for k, v in m.items() if not k.startswith("_")} for m in spec["members"]]
        json.dump(spec, open(sys.argv[1], "w", encoding="utf-8"), separators=(",", ":"))
    c = spec.get("_connect")
    if c:
        print(f"joints: {c['run_on']} ends run on to their support, {c['nodes']} shared nodes, {c['merged']} pieces joined, "
              f"{c['butts']} in-line butts, {c['split']} split at columns, {c['col_tops']} column tops set; FLOATING ends: {len(c['free'])}")
        for f_ in c["free"][:8]:
            print("  FLOATING", f_)
    print(f"{len(parts)} parts, {sum(p['kg'] for p in parts):,.1f} kg -> {sys.argv[2]}")
