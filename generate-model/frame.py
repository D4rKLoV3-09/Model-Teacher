#!/usr/bin/env python3
"""Joints of a read frame, decided from what the drawing shows (replaces nodes.py's search: no 8 m reach, no sideways moves).

    import frame
    rep = frame.connect(members, profiles, rulings, work)

1. Split: a beam drawn through a column (the column's axis crosses it inside its length, the column spans its level) is two
   members meeting at the column axis.
2. Each end of every member that is not a column looks for the member it frames into, along its OWN line only:
   the closest member whose centre line passes within (its half depth + the end member's half depth + GAP) of the end's
   line, between BACK behind the end and REACH ahead of it (a drawn gap is drawing convention, but only a short one).
   Work point (concentric): on the host's centre line where the end's own line meets it. A level member keeps its level
   (TOS-flush into a deeper beam or onto a column side: z is the member's own). An inclined member (brace, web) goes to
   the 3D point of its own line closest to the host axis.
   The end cut: the host's near face ("cope" into a level I beam of the same level, "web" into a column it frames square into).
3. A column with no top read runs up to the top of the highest member landing on it; a column bottom stays at its base.
4. An end that meets nothing within REACH: R08. A (the curator 2026-11-22: a member never floats, it is connected to something): the
   end runs on along its own line up to RUN_ON to the first member it meets (counted, ASSUMED); what still meets nothing
   is then treated as D. B: the member is left out; C: kept as a cantilever; D: only an isolated line is left out, one loose
   end stays listed floating (Cautious). Each such end is a ruling card.
Report: counts + the list of left-out members (with their register ids for the cards).
"""
import math

import numpy as np

REACH = 1500.0      # mm ahead of a drawn end (drawn gaps at supports are shorter than this on GA drawings)
BACK = 400.0        # mm back from a drawn end that overshoots its support
GAP = 120.0         # mm extra clearance between centre-line distance and the two half depths
LEVEL = 30.0
INLINE = 60.0     # mm: a parallel member is in line (a butt) only this close
OVERHANG = 2000.0  # mm: an end this far past a support on its own line is a drawn overhang (R10)
RUN_ON = 6000.0    # mm: R08 A runs a loose end on along its own line this far to the member it meets
BASE_TOL = 300.0   # mm: an upright member's foot this close above the lowest column foot stands on the base (RC)


def _half(pf):
    t = pf["type"]
    if t == "CHS":
        return pf["d"] / 2
    if t in ("SHS",):
        return pf["b"] / 2
    if t == "PL":
        return pf["b"] / 2
    return max(pf.get("h", 200), pf.get("b", 200)) / 2


def _closest(p, d, A, B):
    """Own line p + s d vs host segment A-B: (s, t in [0,1], distance, point on host)."""
    e = B - A
    Le = float(np.linalg.norm(e))
    if Le < 1e-6:
        return None
    e = e / Le
    w = p - A
    bb = float(d @ e)
    den = 1 - bb * bb
    if den < 1e-6:                                           # in line / parallel: the host's nearer end
        t = float(np.clip(w @ e, 0, Le))
        H = A + t * e
        s = float((H - p) @ d)
        dist = float(np.linalg.norm((H - p) - s * d))
        return s, t / Le, dist, H
    s = (bb * float(w @ e) - float(w @ d)) / den
    t = float(np.clip(float(w @ e) + bb * s, 0, Le))
    H = A + t * e
    P = p + s * d
    return s, t / Le, float(np.linalg.norm(P - H)), H


def _face(m, h, d, profiles):
    """The host face a member end bears on, from the host's orientation (welded, zero gap):
    in line with the host (either section): no cut, the square end butts on its square end;
    column host (vertical I): arriving along its depth -> its flange face ("near"), from the side -> its web ("web");
    level I host, a level member square to it within its depth -> coped into the web ("cope"), deeper -> onto the web;
    anything else: the near face."""
    ph, pm = profiles[h["prof"]], profiles[m["prof"]]
    ax = np.subtract(h["b"], h["a"]).astype(float); ax = ax / (np.linalg.norm(ax) or 1)
    if abs(float(d @ ax)) > 0.95:
        return None
    hv = abs(h["a"][2] - h["b"][2]) > 0.98 * math.dist(h["a"], h["b"])
    if ph["type"] == "I" and hv:                             # a column (any member framing in)
        up = np.array(h.get("up") or [1, 0, 0], float)
        dh = np.array([d[0], d[1], 0.0]); dh = dh / (np.linalg.norm(dh) or 1)
        return "near" if abs(float(dh @ up)) > 0.7 else "web"
    if ph["type"] == "I" and abs(h["a"][2] - h["b"][2]) < LEVEL and abs(m["a"][2] - m["b"][2]) < LEVEL:
        dz = abs((m["a"][2] + m["b"][2]) / 2 - h["a"][2])
        return "cope" if pm["type"] == "I" and dz + pm.get("h", 0) / 2 <= ph["h"] / 2 + 5 else "web"   # angles / channels: onto the web
    return "near"


def connect(members, profiles, rulings=None, work=None):
    rep = {"split": 0, "joined": 0, "col_tops": 0, "no_support": []}
    iscol = lambda m: m.get("layer") == "col"
    vert = lambda m: abs(m["a"][2] - m["b"][2]) > 0.98 * math.dist(m["a"], m["b"])
    level = lambda m: abs(m["a"][2] - m["b"][2]) < LEVEL
    cols = [m for m in members if iscol(m)]
    # 1. split beams drawn through columns
    out = []
    for m in members:
        if iscol(m) or not level(m) or m.get("node") is False or m.get("comp"):   # a component's chord is not cut
            out.append(m)
            continue
        a, b = np.array(m["a"], float), np.array(m["b"], float)
        L = float(np.linalg.norm(b - a))
        u = (b - a) / L
        cuts = []
        for c in cols:
            ca, cb = np.array(c["a"], float), np.array(c["b"], float)
            if not (min(ca[2], cb[2]) - 50 <= a[2] <= max(ca[2], cb[2]) + 300):
                continue
            s = float((ca[:2] - a[:2]) @ u[:2]) / max(float(u[:2] @ u[:2]), 1e-9)
            off = float(np.linalg.norm(a[:2] + s * u[:2] - ca[:2]))
            if 600 < s < L - 600 and off < _half(profiles[c["prof"]]) + 50:
                cuts.append(s)
        if not cuts:
            out.append(m)
            continue
        pts = [0.0] + sorted(cuts) + [L]
        for k, (s0, s1) in enumerate(zip(pts, pts[1:])):
            q = dict(m, id=f"{m['id']}.{k + 1}", a=[round(float(v), 1) for v in a + s0 * u], b=[round(float(v), 1) for v in a + s1 * u])
            out.append(q)
        rep["split"] += len(cuts)
    members[:] = out
    # 2. ends
    live = [m for m in members if m.get("node") is not False]
    A = np.array([m["a"] for m in live], float)
    B = np.array([m["b"] for m in live], float)
    HS = np.array([_half(profiles[m["prof"]]) for m in live])
    landing, butts = {}, []
    upright = lambda m: abs(m["a"][2] - m["b"][2]) > 0.98 * math.dist(m["a"], m["b"]) and math.dist(m["a"], m["b"]) > 1000
    feet = [min(c["a"][2], c["b"][2]) for c in cols] or [min(min(m["a"][2], m["b"][2]) for m in live)] if live else [0.0]
    z_base = min(feet)                                     # the base: the lowest column foot (BOS)
    run_on = rulings is not None and rulings.choice("R08") == "A"
    rep["run_on"] = []
    for i, m in enumerate(live):
        if iscol(m):
            continue
        hm = HS[i]
        for k in (0, 1):
            if (m.get("free") or [False, False])[k] or (m.get("fixed") or [False, False])[k]:
                continue                                    # (fixed: joined inside its component, components.py)
            p, q = (A[i], B[i]) if k == 0 else (B[i], A[i])
            d = p - q
            Ld = float(np.linalg.norm(d))
            if Ld < 1:
                continue
            d = d / Ld

            def search(reach):
                best = None
                for j, h in enumerate(live):
                    if j == i:
                        continue
                    r = _closest(p, d, A[j], B[j])
                    if r is None:
                        continue
                    s, t, dist, H = r
                    lim = HS[j] + 50 if iscol(h) else HS[j] + 0.5 * hm + GAP     # into a column: its line passes through the column
                    eh = B[j] - A[j]
                    if abs(float(d @ (eh / (np.linalg.norm(eh) or 1)))) > 0.95:
                        lim = INLINE                                              # parallel: only a member truly in line
                    if not (-BACK <= s <= reach) or dist > lim:                  # its line passes through the host (a deep girder
                        continue                                                  # between twin columns does not frame into them)
                    if s < -BACK + 1 and not iscol(h):
                        continue
                    key = (abs(s) + 0.5 * dist + 120 * h.get("rank", 2) - (200 if iscol(h) else 0), j)   # a chord / column before a web
                    if best is None or key < best[0]:
                        best = (key, j, s, H)
                return best

            best = search(REACH)
            if best is None and upright(m) and p[2] < q[2] and p[2] <= z_base + BASE_TOL:
                # an upright member's foot at the base (a column read from an elevation, not on the layout plan: the curator C4.1,
                # SC6 on grid C1 / 1): it stands on its RC pedestal like every layout column - stated, never floating
                f_ = list(m.get("free") or [False, False]); f_[k] = True
                m["free"], m["free_why"] = f_, "column foot at the base on its RC pedestal (RC not read yet)"
                rep.setdefault("feet_on_rc", []).append((m, "ab"[k]))
                continue
            if best is None:
                # R10: the member runs over a support and stops within OVERHANG beyond it = a drawn overhang (cantilever end)
                over = None
                for j, h in enumerate(live):
                    if j == i:
                        continue
                    r = _closest(p, d, A[j], B[j])
                    if r and -OVERHANG <= r[0] < -1 and r[2] <= HS[j] + hm + GAP and (q - p) @ d < r[0]:
                        over = (r[0], h)
                        break
                if over and (rulings is None or rulings.choice("R10") == "A"):
                    f_ = list(m.get("free") or [False, False]); f_[k] = True
                    m["free"], m["free_why"] = f_, f"R10 A: drawn overhang {abs(over[0]):.0f} mm past {over[1]['ref']} ({over[1]['id']})"
                    rep.setdefault("overhangs", []).append((m, "ab"[k], round(abs(over[0]))))
                    continue
                best = search(RUN_ON) if run_on and not (upright(m) and p[2] > q[2]) else None   # a column top is never run on
                if best is None:
                    rep["no_support"].append((m, "ab"[k]))
                    continue
                rep["run_on"].append((m, "ab"[k], round(best[2]), live[best[1]]["id"]))
            _, j, s, H = best
            h = live[j]
            if level(m):
                P = np.array([H[0], H[1], p[2]]) if iscol(h) else np.array([*(p[:2] + s * d[:2]), p[2]])
                if not iscol(h):                              # onto the host's centre line in plan, own level kept
                    a2, b2 = A[j][:2], B[j][:2]
                    e2 = b2 - a2
                    den = d[0] * e2[1] - d[1] * e2[0]
                    if abs(den) > 0.05 * float(np.linalg.norm(e2)) * float(np.linalg.norm(d[:2]) or 1):   # >= ~3 deg apart
                        tt = ((a2[0] - p[0]) * e2[1] - (a2[1] - p[1]) * e2[0]) / den
                        if -BACK <= tt <= REACH:                  # the work point stays within the reach along its own line
                            P = np.array([p[0] + tt * d[0], p[1] + tt * d[1], p[2]])
            else:
                P = p + s * d                                 # an inclined member: the point of its own line nearest the host axis
                if np.linalg.norm(P - H) <= HS[j]:
                    P = H
            face = _face(m, h, d, profiles)
            if m.get("ends") is None:
                m["ends"] = [None, None]
            m["ends"][k] = [{"host": h["id"], "face": face}] if face else None
            if face is None:
                butts.append((m, k, h))
            m["a" if k == 0 else "b"] = [round(float(v), 1) for v in P]
            if k == 0:
                A[i] = P
            else:
                B[i] = P
            landing.setdefault(h["id"], []).append(m)
            rep["joined"] += 1
    # in-line butts follow their host's final end (the host's own end may have run on to its support after)
    for m, k, h in butts:
        p = np.array(m["a"] if k == 0 else m["b"], float)
        e = min((np.array(h["a"], float), np.array(h["b"], float)), key=lambda q: float(np.linalg.norm(q[:2] - p[:2])))
        m["a" if k == 0 else "b"] = [round(float(e[0]), 1), round(float(e[1]), 1), m["a" if k == 0 else "b"][2]]
    # stubs left inside a joint (both ends landed on one point): dropped, counted
    stubs = [m for m in members if math.dist(m["a"], m["b"]) < 150]
    if stubs:
        ids = {m["id"] for m in stubs}
        members[:] = [m for m in members if m["id"] not in ids]
        for m in members:
            for k in (0, 1):
                e = (m.get("ends") or [None, None])[k]
                if isinstance(e, list) and any(c.get("host") in ids for c in e):
                    m["ends"][k] = None
    rep["stubs"] = len(stubs)
    rep["stub_rows"] = [i for m in stubs for i in (m.get("ids") or [])]
    for c in cols:                                       # column feet at the base stand on RC (not read yet): stated, not floating
        kk = 0 if c["a"][2] <= c["b"][2] else 1
        f_ = list(c.get("free") or [False, False]); f_[kk] = True
        c["free"], c["free_why"] = f_, "column base at BOS on its RC pedestal (RC not read yet)"
    # 3. column tops with nothing read above: the top of the highest member landing on them
    for c in cols:
        if c.get("top_read"):
            kk = 1 if c["b"][2] >= c["a"][2] else 0
            f_ = list(c.get("free") or [False, False]); f_[kk] = True
            c["free"], c["free_why"] = f_, (c.get("free_why", "") + "; " if c.get("free_why") else "") + "top as drawn in the elevations"
            continue
        land = landing.get(c["id"], [])
        if land:
            top = max(max(m["a"][2], m["b"][2]) + profiles[m["prof"]].get("h", 0) / 2 for m in land)
            kk = "b" if c["b"][2] >= c["a"][2] else "a"
            c[kk] = [c[kk][0], c[kk][1], round(top, 1)]
            f_ = list(c.get("free") or [False, False]); f_[0 if kk == "a" else 1] = True
            c["free"], c["free_why"] = f_, (c.get("free_why", "") + "; " if c.get("free_why") else "") + \
                "top level with the members framing into it (cap), nothing drawn above"
            rep["col_tops"] += 1
        else:
            rep["no_support"].append((c, "top"))
    # COL-TOP (the curator 2026-11-23): a column stands alone when its foot is on the ground or on a platform: its top may end with
    # nothing framing into it - stated (free top as drawn), never floating, never left out
    loose = {(id(m), e) for m, e in rep["no_support"]}
    keep = []
    for m, e in rep["no_support"]:
        top = "b" if m["b"][2] >= m["a"][2] else "a"
        foot = "a" if top == "b" else "b"
        if (upright(m) or m.get("layer") == "col") and e in (top, "top") and (id(m), foot) not in loose:
            k_ = 0 if top == "a" else 1
            f_ = list(m.get("free") or [False, False]); f_[k_] = True
            m["free"], m["free_why"] = f_, (m.get("free_why", "") + "; " if m.get("free_why") else "") +                 "column standing on its foot: top as drawn, nothing framing into it"
            rep.setdefault("free_tops", []).append((m, e))
            continue
        keep.append((m, e))
    rep["no_support"] = keep
    return rep


def r08_leave_out(no_support, pick):
    """Members R08 leaves out of the model, from frame's no_support [(member, end)] and the R08 pick:
    B = every member with an end meeting nothing; D (precedent D26) and A (after its run-on, connect() did it) = only members
    neither of whose ends meets anything (an isolated line); C keeps them all. Columns (no key) are never left out here."""
    if pick == "B":
        return [(m, e) for m, e in no_support if m.get("key")]
    if pick in ("D", "A"):
        ends = {}
        for m, e in no_support:
            if m.get("key"):
                ends.setdefault(id(m), (m, set()))[1].add(e)
        return [(m, "ab") for m, es in ends.values() if {"a", "b"} <= es]
    return []

