#!/usr/bin/env python3
"""Clash and gap check on the real prisms (SAT), SKILL.md section 3.

    python tests/clash_check.py PARTS_OR_DATA.json [SPEC.json] [--tol 0.5]

Also writes X.clash.json beside X.data.json (every overlap and floating end, by part index): build_model.py puts them
into the page's Cautious list (Notes, MTO Summary, and the Cautious stepper in the viewer).

Penetration: every pair of convex prisms from different parts whose overlap on every separating axis
(face normals of both + edge x edge) exceeds tol mm. Reported per pair of parts with the depth.
Gap (needs the spec): every member end that names a host must touch it; the SAT separation between
the member and its host is a lower bound of the true gap, so a reported gap is a real one. A cut marked
"sep": true only keeps a member off something and is not checked.
Parts counted by measure (mto kind "measure": bondek deck, studs, edgeform) are indicative solids and are left out.
Exit code 0 = no penetration over tol (gaps are reported, they do not fail the run).
"""
import itertools
import json
import sys

import numpy as np


def prisms(parts):
    out = []        # (part index, vertices (n,3), face normals, edges)
    for i, p in enumerate(parts):
        skip = set(p.get("shank") or [])               # bolt shanks pass through their plates (holes are not cut)
        for k, pr in enumerate(p["pr"]):
            if k in skip:
                continue
            V = np.array([v for f in pr for v in f], float)
            N, E = [], []
            for f in pr:
                f = np.array(f, float)
                n = np.cross(f[1] - f[0], f[2] - f[0])
                if np.linalg.norm(n) > 1e-9:
                    N.append(n / np.linalg.norm(n))
                for k in range(len(f)):
                    e = f[(k + 1) % len(f)] - f[k]
                    if np.linalg.norm(e) > 1e-6:
                        E.append(e / np.linalg.norm(e))
            out.append((i, np.unique(V.round(3), axis=0), dedup(N), dedup(E)))
    return out


def dedup(vs):
    keep = []
    for v in vs:
        if all(abs(abs(v @ k) - 1) > 1e-6 for k in keep):
            keep.append(v)
    return np.array(keep) if keep else np.zeros((0, 3))


def sat(A, B):
    """Signed SAT result: >0 = minimum overlap (penetration depth), <0 = separation lower bound."""
    axes = [A[2], B[2]]
    if len(A[3]) and len(B[3]):
        C = np.cross(A[3][:, None, :], B[3][None, :, :]).reshape(-1, 3)
        n = np.linalg.norm(C, axis=1)
        axes.append(C[n > 1e-6] / n[n > 1e-6, None])
    ax = np.vstack(axes)
    pa, pb = A[1] @ ax.T, B[1] @ ax.T
    ov = np.minimum(pa.max(0), pb.max(0)) - np.maximum(pa.min(0), pb.min(0))
    return float(ov.min())


_PR = None


def _init(pr):
    global _PR
    _PR = pr


def _sat_chunk(pairs):
    return [(j, k, sat(_PR[j], _PR[k])) for j, k in pairs]


def candidates(pr, tol):
    """Prism pairs of different parts whose boxes overlap by more than tol (numpy sweep on x)."""
    lo = np.array([p[1].min(0) for p in pr]); hi = np.array([p[1].max(0) for p in pr])
    own = np.array([p[0] for p in pr])
    order = np.argsort(lo[:, 0]); los = lo[order, 0]
    out = []
    for n, k in enumerate(order):
        m = np.searchsorted(los, hi[k, 0] - tol, side="left")       # boxes starting before k ends (on x)
        js = order[n + 1:m]
        if not len(js):
            continue
        ok = np.all(lo[js] < hi[k] - tol, 1) & np.all(lo[k] < hi[js] - tol, 1) & (own[js] != own[k])
        out.extend((int(j), int(k)) for j in js[ok])
    return out


def penetrations(pr, tol=0.5, workers=None):
    """{(part a, part b): deepest SAT overlap} over tol for every pair of prisms of different parts.
    Runs the SAT tests on all CPU cores (the curator: faster builds); same result as one core."""
    import os
    cand = candidates(pr, tol)
    workers = workers or min(os.cpu_count() or 1, 16)
    if workers > 1 and len(cand) > 4000:
        from concurrent.futures import ProcessPoolExecutor
        size = max(len(cand) // (workers * 4), 500)
        chunks = [cand[i:i + size] for i in range(0, len(cand), size)]
        with ProcessPoolExecutor(workers, initializer=_init, initargs=(pr,)) as ex:
            res = [r for part in ex.map(_sat_chunk, chunks) for r in part]
    else:
        _init(pr)
        res = _sat_chunk(cand)
    hits = {}
    for j, k, d in res:
        if d > tol:
            key = tuple(sorted((pr[j][0], pr[k][0])))
            hits[key] = max(hits.get(key, 0), d)
    return hits


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tol = float(sys.argv[sys.argv.index("--tol") + 1]) if "--tol" in sys.argv else 0.5
    d = json.load(open(args[0], encoding="utf-8"))
    parts = d["parts"] if isinstance(d, dict) else d
    pr = [p for p in prisms(parts) if (parts[p[0]].get("mto") or {}).get("kind") != "measure"]
    hits = penetrations(pr, tol)                      # box sweep on x, SAT on all cores
    name = lambda i: f"{parts[i].get('t')}{'/' + parts[i]['ref'] if parts[i].get('ref') else ''} @ {parts[i].get('grid', '')}"
    print(f"{len(parts)} parts, {len(pr)} prisms: {len(hits)} penetrating pairs over {tol} mm")
    for (a, b), dep in sorted(hits.items(), key=lambda kv: -kv[1])[:60]:
        print(f"  {dep:7.1f} mm  {name(a)}  x  {name(b)}")
    if len(args) > 1:                                 # host contact check
        spec = json.load(open(args[1], encoding="utf-8"))
        ms = [m for m in spec["members"] if not m.get("host_only")]
        idx = {m["id"]: i for i, m in enumerate(ms)}
        by_part = {}
        for p in pr:
            by_part.setdefault(p[0], []).append(p)
        gaps = []
        for i, m in enumerate(ms):
            for e in (m.get("ends") or []):
                best = None                                   # an end at a node is supported when it touches ANY of its hosts
                for c in (e if isinstance(e, list) else []):
                    h = c.get("host")
                    if h is None or h not in idx or c.get("sep"):     # "sep": separation cut, no contact meant
                        continue
                    sep = max(sat(a, b) for a in by_part[i] for b in by_part[idx[h]])
                    if best is None or sep > best[0]:
                        best = (sep, h, c.get("face"))
                if best is not None and best[0] < -tol:
                    gaps.append((-best[0], m["id"], best[1], best[2]))
        print(f"host contacts: {len(gaps)} member ends not touching their host (gap > {tol} mm)")
        for g in sorted(gaps, reverse=True)[:40]:
            print(f"  {g[0]:7.1f} mm  {g[1]} -> {g[2]} ({g[3]})")
        # concentric: an end that names a host has its work point on the host's centre line
        ecc = []
        for m in ms:
            for k, e in enumerate(m.get("ends") or []):
                for c in (e if isinstance(e, list) else [])[:1]:      # the end's main support (others only cut it at the node)
                    h = c.get("host")
                    if h in idx and not c.get("sep") and not m.get("ecc"):
                        d_ = axis_dist(m["a"] if k == 0 else m["b"], ms[idx[h]])
                        bear = half_size(spec, ms[idx[h]]) + half_size(spec, m) - 5   # bearing ON the host (purlin on a rafter)
                        if 1.0 < d_ < bear:
                            ecc.append((d_, m["id"], "ab"[k], h))
        print(f"eccentric ends: {len(ecc)} work points off their host's centre line (> 1 mm)")
        for e_ in sorted(ecc, reverse=True)[:20]:
            print(f"  {e_[0]:7.1f} mm  {e_[1]} {e_[2]} -> {e_[3]}")
    # steel never floats: every end of every member touches steel or a shown RC ghost (a flagged cantilever excepted)
    ghosts = d.get("ghosts", []) if isinstance(d, dict) else []
    fl = free_ends(parts, pr, ghosts, tol)
    print(f"free ends: {len(fl)} member ends touch nothing (no steel, no RC ghost within {tol} mm)")
    for f_ in fl[:40]:
        print(f"  FLOATING {name(f_[0])} end {f_[1]} at {f_[2]}")
    if isinstance(d, dict):                           # for the Cautious list of the page (build_model.py)
        src = args[0]
        out = src[:-len(".data.json")] if src.endswith(".data.json") else src.rsplit(".", 1)[0]
        json.dump({"n_parts": len(parts), "tol": tol,
                   "overlaps": [{"p": [a, b], "mm": round(dep, 1)} for (a, b), dep in sorted(hits.items(), key=lambda kv: -kv[1])],
                   "floating": [{"p": [f_[0]], "end": f_[1], "at": [round(float(c)) for c in f_[2]]} for f_ in fl]},
                  open(out + ".clash.json", "w", encoding="utf-8"))
    return 1 if hits else 0


def half_size(spec, m):
    pf = spec.get("profiles", {}).get(m.get("prof"), {})
    return min(v for v in (pf.get("h"), pf.get("b"), pf.get("d"), 200) if v) / 2


def axis_dist(p, h):
    a, b, p = np.array(h["a"], float), np.array(h["b"], float), np.array(p, float)
    ab = b - a
    t = np.clip((p - a) @ ab / max(ab @ ab, 1e-9), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def box_solid(c, half, axes, owner):
    corners = np.array([c + sx * half[0] * axes[0] + sy * half[1] * axes[1] + sz * half[2] * axes[2]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    ax = np.array(axes)
    return (owner, corners, ax, ax)


def free_ends(parts, pr, ghosts, tol, probe_len=40.0, only=None):
    """Ends of linear members whose end probe (the section box over the last 40 mm plus tol beyond) meets nothing.
    only = part indices to test (the others still count as support), default all."""
    solids = list(pr)
    for g in ghosts:
        rz = g.get("rz", 0.0)
        axes = [np.array([np.cos(rz), np.sin(rz), 0.0]), np.array([-np.sin(rz), np.cos(rz), 0.0]), np.array([0, 0, 1.0])]
        half = [g["s"][0] / 2, g["s"][1] / 2, g["s"][2] / 2] if "r" not in g else [g["r"], g["r"], g["s"][2] / 2]
        solids.append(box_solid(np.array(g["c"], float), half, axes, -1))
    lo = np.array([s[1].min(0) for s in solids]); hi = np.array([s[1].max(0) for s in solids])
    vert_ = [p for p in parts if p.get("ax") and abs(p["ax"][2]) > 0.9 and p.get("pr")]
    zmin = min((min(q[2] for pr_ in p["pr"] for f in pr_ for q in f) for p in vert_), default=0)   # the lowest column foot = base
    out = []
    for i, p in enumerate(parts):
        if only is not None and i not in only:
            continue
        mto = p.get("mto") or {}
        if mto.get("kind") in ("measure", "piece") or p.get("plate") or not p.get("ax") or p.get("L", 0) < 150 or p.get("std"):
            continue
        V = np.array([q for pr_ in p["pr"] for f in pr_ for q in f], float)
        x = np.array(p["ax"], float); x /= np.linalg.norm(x)
        u = np.cross(x, [0, 0, 1.0]) if abs(x[2]) < 0.9 else np.cross(x, [1.0, 0, 0]); u /= np.linalg.norm(u)
        v = np.cross(x, u)
        s = V @ x
        free = p.get("free") or [False, False]
        for k, (sel, sgn) in enumerate(((s <= s.min() + probe_len, -1), (s >= s.max() - probe_len, 1))):
            if free[k]:
                continue
            R = V[sel]
            if abs(x[2]) > 0.9 and ((sgn < 0) == (x[2] > 0)) and R[:, 2].min() <= zmin + 100:
                continue                                       # a column foot at the base level stands on the base
            s0, s1 = (s.min() - tol, s.min() + probe_len) if sgn < 0 else (s.max() - probe_len, s.max() + tol)
            cu, cv = R @ u, R @ v
            c = x * (s0 + s1) / 2 + u * (cu.min() + cu.max()) / 2 + v * (cv.min() + cv.max()) / 2
            half = [(s1 - s0) / 2, (cu.max() - cu.min()) / 2 + tol, (cv.max() - cv.min()) / 2 + tol]
            pb = box_solid(c, half, [x, u, v], i)
            plo, phi = pb[1].min(0), pb[1].max(0)
            cand = np.where(np.all(lo <= phi, axis=1) & np.all(hi >= plo, axis=1))[0]
            if not any(solids[j][0] != i and sat(pb, solids[j]) > -tol for j in cand):
                end_pt = (c + x * (half[0] * sgn)).round().astype(int).tolist()
                out.append((i, "ab"[k], end_pt))
    return out


if __name__ == "__main__":
    sys.exit(main())
