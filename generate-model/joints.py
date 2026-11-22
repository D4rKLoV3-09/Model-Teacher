#!/usr/bin/env python3
"""Bolted connections from their drawing detail: ONE spec line per joint -> plates + bolt sets at the right place.

the curator's rule: a joint is welded or bolted. No connection on the drawings = welded, zero gap, nothing here.
A bolted joint whose detail is given is modelled as drawn with these kinds (geometry.py calls apply() after
the members are built). Every part carries "conn" (the detail ref, e.g. "BP3"), so the gate can count it.

Typical details: read a detail ONCE into spec["conn_types"] and give each place it applies one short line - the
detail is never re-read per joint:
  "conn_types": {"EP1": {"kind": "endplate", "plate": [500, 200, 20], "bolts": {"d": 20, "n": [4, 2], "p": [110, 110]},
                         "sheet": "p27"}},
  "joints": [{"conn": "EP1", "member": "R1", "end": "a", "host": "C1"}, {"conn": "EP1", "member": "R7", "end": "b", "host": "C9"}]
A joint line may override any key of its type (e.g. a longer plate at one place).
spec["joints"] = [
  {"conn": "BP3", "kind": "base", "member": "C-A1", "end": "a",            base / cap plate on the member end
   "plate": [L, W, t],                                                       L along the member's u, W along v
   "bolts": {"d": 24, "n": [2, 2], "p": [300, 200], "embed": 500}},          anchor bolts (no "bolts" = cap plate);
                                                                             the member end = plate top (BOS + t)
  {"conn": "S1", "kind": "splice", "member": "C-A1", "end": "b",            cover-plate splice of two I members in
   "fl": [L, W, t], "web": [L, H, t],                                        line at that end (both flanges outside,
   "fl_bolts": {"d": 20, "n": [4, 2], "p": [70, 100], "gap": 80},            web both sides); n = [along, across],
   "web_bolts": {"d": 20, "n": [4, 2], "p": [70, 70], "gap": 80}},           along = total over both members
  {"conn": "S2", "kind": "endplate", "member": "B1", "end": "b",            end plate: the member is cut back t and
   "plate": [H, W, t], "u0": 0, "host": "C-A2" | "pair": true,                the plate fills it; "pair" = end-plate
   "bolts": {"d": 20, "n": [4, 2], "p": [90, 100]}},                         splice (two plates, grip 2t); "host"
                                                                             gives the grip (t + host flange)
  {"conn": "F1", "kind": "fin", "member": "B2", "end": "a", "side": "+v",   fin plate on the web, welded to the host
   "plate": [H, L, t], "bolts": {"d": 20, "n": [3, 1], "p": [70, 60]}},     face at the member end
  {"conn": "PC1", "kind": "cleat", "member": "P-12", "host": "R-3",         purlin cleat: vertical plate beside the
   "side": "-v", "plate": [H, W, t], "bolts": {"d": 12, "n": [2, 1], "p": [50, 60]}}   purlin web, standing on the
]                                                                           rafter top where the two cross
Optional on every joint: "t" (tag, default PLT / BOLT), "layer" (default "plate" / "bolt"), "src", "grade", "desc".
Bolts: hex head 1.5d across flats x 0.65d, nut 0.8d, shank to 0.5d past the nut; one part per bolt group, MTO piece
row "Set of n bolt(s)" with kg per bolt from those dimensions. Holes are not cut (convex prisms): the clash check
skips the shanks (part["shank"]). Heads and nuts are still checked.
"""
import math

import numpy as np

import geometry as G

STEEL = 7.85e-6          # kg/mm3


def _r(v):
    return [round(float(c), 1) for c in v]


def _pr(prisms):
    return [[[_r(p) for p in f] for f in pr] for pr in prisms]


def _np(prisms):
    return [[np.array(f, float) for f in pr] for pr in prisms]


def _common(j, kind_t, layer):
    out = {"conn": j["conn"], "t": j.get("t" if kind_t == "PLT" else "bolt_t", kind_t), "layer": j.get("layer" if kind_t == "PLT" else "bolt_layer", layer)}
    mp = j.get("_mp") or {}                                # the member the joint sits on: same assembly, level, grid
    for k in ("unit", "lu", "grid", "asm", "sub", "tw", "level"):
        if k in j or k in mp:
            out[k] = j.get(k, mp.get(k))
    out.setdefault("grid", "")
    out["src"] = j.get("src") or f"Detail {j['conn']}" + (f" ({j['sheet']})" if j.get("sheet") else "")
    if kind_t == "PLT" and "grade" in j:
        out["grade"] = j["grade"]
    out["plate"] = None
    return out


def plate_part(j, desc, o, x, u, v, s0, s1, r, dims):
    """Plate box: along x from s0 to s1, section r = (u0, u1, v0, v1) about o. dims = (L, W, t) for the MTO."""
    L, W, t = (float(d) for d in dims)
    part = _common(j, "PLT", "plate")
    part.update({"prof": f"PL{t:g}", "desc": j.get("desc", desc), "ax": _r(x), "L": round(L, 1),
                 "kg": round(L * W * t * STEEL, 2), "plate": {"L": round(L, 1), "W": round(W, 1), "t": t},
                 "pr": _pr([G.box(o, x, u, v, s0, s1, r)])})
    return part


def bolt_part(j, pts, z, d, grip, anchor=False, embed=0.0, desc=None):
    """One bolt group. pts = points on the head-side contact face, z = into the joint. Anchor: nut on top, rod below."""
    z = np.asarray(z, float) / np.linalg.norm(z)
    e1 = np.cross(z, [0, 0, 1.0]) if abs(z[2]) < 0.9 else np.cross(z, [1.0, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(z, e1)
    k, m, s = 0.65 * d, 0.8 * d, 1.5 * d
    R = s / math.sqrt(3)                                   # hex circumradius from across-flats
    if anchor:
        sh0 = -(m + 0.5 * d)
        L = 5 * math.ceil((grip + embed - sh0) / 5)       # rod length to 5 mm, the extra goes into the embedment
        sh1 = sh0 + L
    else:
        L = 5 * math.ceil((grip + m + 0.5 * d) / 5)
        sh0, sh1 = 0.0, L
    prisms, shank = [], []
    for p in pts:
        p = np.asarray(p, float)
        if not anchor:
            prisms.append(G.round_prism(p, z, e1, e2, -k, 0.0, R, n=6))
        shank.append(len(prisms))
        prisms.append(G.round_prism(p, z, e1, e2, sh0, sh1, d / 2, n=8))
        nut = (-m, 0.0) if anchor else (grip, grip + m)
        prisms.append(G.round_prism(p, z, e1, e2, nut[0], nut[1], R, n=6))
    hexa = 3 * math.sqrt(3) / 2 * R * R
    kg = (math.pi * d * d / 4 * L + hexa * ((0 if anchor else k) + m)) * STEEL
    n = len(pts)
    what = f"M{d:g} x {L:g} anchor bolt + nut" if anchor else f"M{d:g} x {L:g} Gr 8.8 bolt + nut"
    part = _common(j, "BOLT", "bolt")
    part.update({"prof": f"M{d:g}x{L:g}", "desc": desc or what, "ax": _r(z), "L": round(L, 1), "kg": round(kg * n, 2),
                 "qty_in": n, "shank": shank, "pr": _pr(prisms),
                 "mto": {"kind": "piece", "cat": "Bolt", "desc": f"{what} (set of {n})", "kgm": round(kg, 3), "set": n}})
    part["grade"] = j.get("bolt_grade", "G8.8")
    return part


def grid(o, e1, e2, n, p, gap=None):
    """Bolt positions about o: n = [along e1, along e2], p = pitches. gap = clear centre gap along e1 (splices)."""
    n1, n2 = int(n[0]), int(n[1])
    p1, p2 = float(p[0]), float(p[1])
    if gap is not None:
        half = [gap / 2 + k * p1 for k in range(max(1, n1 // 2))]
        a1 = sorted([-h for h in half] + half)
    else:
        a1 = [(k - (n1 - 1) / 2) * p1 for k in range(n1)]
    a2 = [(k - (n2 - 1) / 2) * p2 for k in range(n2)]
    return [o + s * e1 + t * e2 for s in a1 for t in a2]


# ---------------- member end helpers ----------------
def _end(part, m, end):
    """Axis x, outward sign, end-face position along x (from a) for a built member part."""
    a, b, x, u, v = m["_f"]
    V = np.array([p for pr in part["pr"] for f in pr for p in f], float)
    s = (V - a) @ x
    return (1.0, float(s.max())) if end == "b" else (-1.0, float(s.min()))


def cut_back(part, m, end, t):
    """Shorten a built member by t at its end (room for an end plate): clip every prism, recompute L and kg."""
    a, b, x, u, v = m["_f"]
    sg, se = _end(part, m, end)
    n, d = (x, x @ a + se - t) if sg > 0 else (-x, -(x @ a + se + t))
    prisms = [G.clip(pr, n, d) for pr in _np(part["pr"])]
    prisms = [p for p in prisms if p]
    part["pr"] = _pr(prisms)
    V = np.array([p for pr in prisms for f in pr for p in f])
    L = float(((V - a) @ x).max() - ((V - a) @ x).min())
    part["L"] = round(L, 1)
    part["kg"] = round(m["_pf"]["kgm"] * L / 1000.0, 2)
    return sg, se


def _tf(pf):
    return pf.get("tf", pf.get("t", 10.0))


# ---------------- kinds ----------------
def base(j, part, m):
    a, b, x, u, v = m["_f"]
    sg, se = _end(part, m, j.get("end", "a"))
    L, W, t = (float(q) for q in j["plate"])
    o = a + se * x
    w = sg * x                                             # outward
    out = [plate_part(j, "Base plate" if j.get("bolts") else "Cap plate", o, w, u, np.cross(w, u), 0.0, t,
                      (-L / 2, L / 2, -W / 2, W / 2), (L, W, t))]
    bl = j.get("bolts")
    if bl:
        pts = grid(o, u, np.cross(w, u), bl["n"], bl["p"])
        out.append(bolt_part(j, pts, w, float(bl["d"]), t, anchor=True, embed=float(bl.get("embed", 500))))
    return out


def endplate(j, part, m, by_id, pair=False):
    a, b, x, u, v = m["_f"]
    H, W, t = (float(q) for q in j["plate"])
    sg, se = cut_back(part, m, j.get("end", "b"), t)
    w = sg * x
    o = a + (se - sg * t) * x                              # inner face of the plate (member side)
    u0 = float(j.get("u0", 0.0))
    vv = np.cross(w, u)
    out = [plate_part(j, "End plate", o, w, u, vv, 0.0, t, (u0 - H / 2, u0 + H / 2, -W / 2, W / 2), (H, W, t))]
    bl = j.get("bolts")
    if bl:
        if pair:
            grip = 2 * t
        elif j.get("host"):
            grip = t + _tf(by_id[j["host"]]["_pf"])
        else:
            grip = float(bl.get("grip", 2 * t))
        pts = grid(o + u0 * u, u, vv, bl["n"], bl["p"])
        out.append(bolt_part(j, pts, w, float(bl["d"]), grip))
    return out


def splice(j, part, m):
    a, b, x, u, v = m["_f"]
    pf = m["_pf"]
    if pf["type"] != "I":
        raise ValueError(f"{j['conn']}: cover-plate splice needs an I member ({m['id']} is {pf['type']}); use endplate with pair")
    h, tw, tf = pf["h"], pf["tw"], pf["tf"]
    P = b if j.get("end", "b") == "b" else a
    s_p = (P - a) @ x
    out = []
    if j.get("fl"):
        L, W, t = (float(q) for q in j["fl"])
        bl = j.get("fl_bolts")
        for sgn in (1, -1):
            r = (h / 2, h / 2 + t, -W / 2, W / 2) if sgn > 0 else (-h / 2 - t, -h / 2, -W / 2, W / 2)
            out.append(plate_part(j, "Splice flange plate", a, x, u, v, s_p - L / 2, s_p + L / 2, r, (L, W, t)))
            if bl:
                o = P + sgn * (h / 2 + t) * u
                out.append(bolt_part(j, grid(o, x, v, bl["n"], bl["p"], bl.get("gap")), -sgn * u, float(bl["d"]), t + tf))
    if j.get("web"):
        L, H, t = (float(q) for q in j["web"])
        for sgn in (1, -1):
            r = (-H / 2, H / 2, tw / 2, tw / 2 + t) if sgn > 0 else (-H / 2, H / 2, -tw / 2 - t, -tw / 2)
            out.append(plate_part(j, "Splice web plate", a, x, u, v, s_p - L / 2, s_p + L / 2, r, (L, H, t)))
        bl = j.get("web_bolts")
        if bl:
            o = P + (tw / 2 + t) * v
            out.append(bolt_part(j, grid(o, x, u, bl["n"], bl["p"], bl.get("gap")), -v, float(bl["d"]), 2 * t + tw))
    return out


def fin(j, part, m):
    a, b, x, u, v = m["_f"]
    pf = m["_pf"]
    tw = pf.get("tw", pf.get("t", 10.0))
    H, L, t = (float(q) for q in j["plate"])
    sg, se = _end(part, m, j.get("end", "a"))
    sd = -1.0 if j.get("side", "+v") == "-v" else 1.0
    s0, s1 = (se - L, se) if sg > 0 else (se, se + L)
    r = (-H / 2, H / 2, tw / 2, tw / 2 + t) if sd > 0 else (-H / 2, H / 2, -tw / 2 - t, -tw / 2)
    out = [plate_part(j, "Fin plate", a, x, u, v, s0, s1, r, (L, H, t))]
    bl = j.get("bolts")
    if bl:
        o = a + (s0 + s1) / 2 * x + sd * (tw / 2 + t) * v
        out.append(bolt_part(j, grid(o, u, x, bl["n"], bl["p"]), -sd * v, float(bl["d"]), t + tw))
    return out


def _web(pf, u, v, side):
    """Web normal and the offset of the web's outer face on that side (purlin cleat contact)."""
    t = pf["type"]
    if t == "I":
        return v, (1.0 if side != "-v" else -1.0), pf["tw"] / 2, pf["tw"]
    if t == "C":
        return v, -1.0, pf["b"] / 2, pf["tw"]
    if t == "LIPC":
        return u, -1.0, pf["h"] / 2, pf["t"]
    raise ValueError(f"cleat on a {t} purlin: give 'off' (web face offset) and 'web_t'")


def cleat(j, part, m, by_id):
    a, b, x, u, v = m["_f"]
    host = by_id[j["host"]]
    ha, hb, hx, hu, hv = host["_f"]
    pf = m["_pf"]
    if "off" in j:
        nrm = v if j.get("side", "-v").endswith("v") else u
        sgn = -1.0 if j.get("side", "-v").startswith("-") else 1.0
        off, web_t = float(j["off"]), float(j.get("web_t", 6.0))
    else:
        nrm, sgn, off, web_t = _web(pf, u, v, j.get("side", "-v"))
    # crossing: closest points of the two axes
    w0 = a - ha
    A, B, C = x @ x, x @ hx, hx @ hx
    D, E = x @ w0, hx @ w0
    den = A * C - B * B
    sp = (B * E - C * D) / den if abs(den) > 1e-9 else 0.0
    sh = (A * E - B * D) / den if abs(den) > 1e-9 else 0.0
    q, p0 = a + sp * x, ha + sh * hx
    up = hu if hu[2] >= 0 else -hu
    top = p0 + G.half_dims(host["_pf"])[0] * up            # rafter top face at the crossing
    Z = np.array([0, 0, 1.0])
    zz = Z - (Z @ x) * x - (Z @ nrm) * nrm
    zz /= np.linalg.norm(zz)
    H, W, t = (float(c) for c in j["plate"])
    h0 = (top - q) @ zz
    o = q + h0 * zz
    side = sgn * nrm
    r = (0.0, H, off, off + t) if sgn > 0 else (0.0, H, -off - t, -off)
    out = [plate_part(j, "Purlin cleat", o, np.cross(zz, nrm), zz, nrm, -W / 2, W / 2, r, (H, W, t))]
    bl = j.get("bolts")
    if bl:
        c = o + float(bl.get("bz", H / 2)) * zz + side * (off + t)
        out.append(bolt_part(j, grid(c, zz, x, bl["n"], bl["p"]), -side, float(bl["d"]), t + web_t))
    return out


def apply(joints, parts, idx, by_id, types=None):
    """Add the joint parts (and cut members back for end plates). idx = member id -> part index.
    types = spec["conn_types"]: a typical detail read ONCE ({"EP1": {"kind": ..., "plate": ..., "bolts": ...}});
    each joint line then only says where it goes ({"conn": "EP1", "member": "R1", "end": "a", "host": "C1"})."""
    types = types or {}
    out = []
    for j in joints:
        if not j.get("conn"):
            raise ValueError(f"joint without conn ref: {j}")
        j = dict(types.get(j["conn"], {}), **j)
        mid = j.get("member")
        if mid not in idx:                                # split at generation (nodes.py: "<id>.1", "<id>.x1" ...): the piece
            pieces = sorted(k for k in idx if str(k).startswith(f"{mid}."))   # that holds the joint's end
            if pieces:
                mid = pieces[-1] if j.get("end", "b") == "b" else pieces[0]
                j = dict(j, member=mid)
        if mid not in idx:
            raise ValueError(f"{j['conn']}: unknown member {mid}")
        if "kind" not in j:
            raise ValueError(f"{j['conn']}: no kind - define the detail once in conn_types or give it on the joint")
        part, m = parts[idx[mid]], by_id[mid]
        j["_mp"] = part
        k = j["kind"]
        if k == "base":
            out += base(j, part, m)
        elif k == "endplate":
            out += endplate(j, part, m, by_id, pair=bool(j.get("pair")))
            if j.get("pair") and j.get("other"):          # the other member of an end-plate splice gets its plate too
                o = dict(j, member=j["other"], end=j.get("other_end", "a"), _mp=parts[idx[j["other"]]])
                out += endplate(o, parts[idx[j["other"]]], by_id[j["other"]], by_id, pair=True)[:1]
        elif k == "splice":
            out += splice(j, part, m)
        elif k == "fin":
            out += fin(j, part, m)
        elif k == "cleat":
            out += cleat(j, part, m, by_id)
        else:
            raise ValueError(f"{j['conn']}: unknown joint kind {k} (base | endplate | splice | fin | cleat)")
    return out
