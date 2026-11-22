#!/usr/bin/env python3
"""Joints made at generation time (the curator's three rules), run by geometry.build on every spec before any cutting:

  1. Steel never floats: every end is joined to the member it frames into (or to RC shown as a ghost / the base).
  2. Concentric work points: the end's work point goes ON the support's centre line (corner = column x beam centre
     line; chevron / V braces landing on one beam meet at ONE point on its centre line; X-brace = crossing point).
  3. A gap drawn between a member and its support is drawing convention: the reader's end is where the line stops,
     not where the steel stops. Each end runs along its own line to the first member it meets (ahead up to REACH,
     or back if it overshoots), wherever that is.

connect(members, profiles) works on the spec members in place, in this order:
  a. pieces of one mark drawn in line with a gap and nothing between (a label break) become one member;
     two different marks in line butt at the middle of the gap ("caut" set: different sections in line);
  b. a member drawn through a column (the column spans its level) is split at the column;
  c. every end runs on to its support; the work point goes onto the support's centre line;
  d. ends landing on one support within NODE_TOL of each other share one work point (chevron, V, K);
  e. a column top with nothing above runs up to the member over it, or comes down to the highest member framing in;
  f. ends get the cut that matches: "auto" when the support ranks lower and the point is on its axis, else an
     explicit face (column: web between the flanges when the member fits, else flange face / tips; level beam into a
     level I girder: cope; anything else: near face).
Skipped: plates and bondek / studs / edgeform, members with "node": false, ends flagged in "free" (a cantilever the
drawing shows: "free": [true, false], "free_why": "<sheet + reason>"), ends that already carry a plane cut.
Column feet at the base level stand on RC. Returns a report; report["free"] lists every end still without support -
geometry prints them and the gate fails on them.
"""
import math

import numpy as np

REACH = 8000.0       # mm an end may run on to find its support (the verifier looks 8 m ahead)
REACH_UP = 30000.0   # mm a column top may run up its own line to the member it carries
RISE = 3000.0        # mm another member may lift a column top to meet it (its line above the column's top)
BACK = 600.0         # mm an end may come back when the drawn line overshoots its support
NODE_TOL = 400.0     # mm: ends on one support closer than this along it share one work point
LEVEL = 30.0         # mm: |dz| under this = a level member
CUT_TOL = 25.0       # mm: other members passing this close to a node also cut the end (read work points are a few mm apart)
SKIP_TYPES = ("PL", "DECK", "STUD", "EDGE")


def _half(pf):
    t = pf["type"]
    if t == "CHS":
        return pf["d"] / 2, pf["d"] / 2
    if t == "SHS":
        return pf["b"] / 2, pf["b"] / 2
    if t == "PL":
        return pf["t"] / 2, pf["b"] / 2
    return pf.get("h", 200) / 2, pf.get("b", 200) / 2


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def connect(members, profiles):
    rep = {"merged": 0, "butts": 0, "split": 0, "run_on": 0, "nodes": 0, "col_tops": 0, "free": []}
    live = [m for m in members if profiles[m["prof"]]["type"] not in SKIP_TYPES and m.get("node", True) and not m.get("host_only")]
    vert = lambda m: abs(m["a"][2] - m["b"][2]) > 0.98 * math.dist(m["a"], m["b"])
    iscol = lambda m: vert(m) and (m.get("rank", 2) <= 0 or m.get("layer") in ("col", "column"))   # truss verticals are not columns
    level = lambda m: abs(m["a"][2] - m["b"][2]) < LEVEL
    free = lambda m, k: bool((m.get("free") or [False, False])[k])
    planecut = lambda m, k: isinstance((m.get("ends") or [None, None])[k], list) and any("plane" in c for c in m["ends"][k])
    cols = lambda: [m for m in live if iscol(m)]

    # a. pieces of one mark in line -> one member; different marks in line -> butt at mid-gap
    LA0 = np.array([m["a"] for m in live], float)
    LB0 = np.array([m["b"] for m in live], float)
    HS0 = np.array([max(_half(profiles[m["prof"]])) for m in live])

    def in_line_pairs():
        L_ = [m for m in live if not vert(m) and not m.get("arc")]
        if not L_:
            return
        P = np.array([m["a"] for m in L_] + [m["b"] for m in L_], float)            # end points
        Q = np.array([m["b"] for m in L_] + [m["a"] for m in L_], float)            # the other end of the same member
        D = P - Q
        D /= np.maximum(np.linalg.norm(D, axis=1), 1e-9)[:, None]                  # out of the member at that end
        n_ = len(L_)
        for i in range(2 * n_):
            g = P - P[i]
            along = g @ D[i]
            lat = np.linalg.norm(g - along[:, None] * D[i], axis=1)
            cand = np.where((along >= 150) & (along <= 1500) & (lat <= 60) & (D @ D[i] < -0.9995))[0]   # a label break, not a butt
            for j in cand:
                if j % n_ == i % n_ or j < i:
                    continue
                if _col_between(P[i], P[j], cols(), profiles) or _crossed(P[i], P[j], L_[i % n_], L_[j % n_], LA0, LB0, HS0, live):
                    continue                                      # something stands in the gap: a joint, not a label break
                yield L_[i % n_], i // n_, L_[j % n_], j // n_
    gone, renamed = set(), {}
    for x, kx, y, ky in list(in_line_pairs()):
        if id(x) in gone or id(y) in gone:
            continue
        if x.get("ref") == y.get("ref") and x["prof"] == y["prof"]:
            far = y["b"] if ky == 0 else y["a"]
            x["a" if kx == 0 else "b"] = list(far)
            gone.add(id(y))
            renamed[y["id"]] = x["id"]
            rep["merged"] += 1
        else:
            px, py = (x["a"] if kx == 0 else x["b"]), (y["a"] if ky == 0 else y["b"])
            mid = [round((px[i] + py[i]) / 2, 1) for i in range(3)]
            for m_, k_ in ((x, kx), (y, ky)):
                m_["a" if k_ == 0 else "b"] = list(mid)
                e = list(m_.get("ends") or ["auto", "auto"])
                e[k_] = None
                m_["ends"] = e
                m_["caut"] = f"different sections butt in line ({x.get('ref')} {x['prof']} / {y.get('ref')} {y['prof']})"
                m_.setdefault("_butt", set()).add(k_)
            rep["butts"] += 1
    if gone:
        members[:] = [m for m in members if id(m) not in gone]
        live[:] = [m for m in live if id(m) not in gone]
        _rehost(members, renamed)

    # b0. two members drawn through each other (X bracing, crossing beams): the lighter is split at every crossing
    #     (all crossings of a round at once; it used to stop after the first one per round, 2 in a whole model)
    for _round in range(6):
        L2 = [m for m in live if not vert(m) and not m.get("arc")]      # a rolled member is never split
        if len(L2) < 2:
            break
        A_ = np.array([m["a"] for m in L2], float); B_ = np.array([m["b"] for m in L2], float)
        lo_, hi_ = np.minimum(A_, B_) - 30, np.maximum(A_, B_) + 30
        cuts = {}                                                        # id(weak) -> (weak, [points])
        for ii, x in enumerate(L2):
            js = np.where(np.all(lo_[ii + 1:] <= hi_[ii], 1) & np.all(hi_[ii + 1:] >= lo_[ii], 1))[0] + ii + 1
            for jj in js:
                y = L2[int(jj)]
                cp = _crossing(x, y)
                if cp is None:
                    continue
                wx, wy = (profiles[x["prof"]].get("kgm", 0), profiles[y["prof"]].get("kgm", 0))
                weak = x if wx < wy or (wx == wy and x["id"] > y["id"]) else y
                cuts.setdefault(id(weak), (weak, []))[1].append(cp[0] if weak is x else cp[1])
        if not cuts:
            break
        for weak, pts in cuts.values():
            a0 = np.array(weak["a"], float); u = np.array(weak["b"], float) - a0; L_ = float(np.linalg.norm(u)); u /= L_
            ss = []
            for P in sorted(pts, key=lambda q: float((np.asarray(q) - a0) @ u)):
                s_ = float((np.asarray(P) - a0) @ u)
                if not ss or s_ - ss[-1][0] > 50:                         # two crossings within 50 mm: one point
                    ss.append((s_, [round(float(v), 1) for v in P]))
            pts_ = [weak["a"]] + [q for _, q in ss] + [weak["b"]]
            e0 = weak.get("ends") or ["auto", "auto"]
            n_ = len(pts_) - 1
            pieces = []
            for k in range(n_):
                e = dict(weak, id=f"{weak['id']}.x{k + 1}", a=list(pts_[k]), b=list(pts_[k + 1]),
                         ends=[e0[0] if k == 0 else "auto", e0[1] if k == n_ - 1 else "auto"])
                if weak.get("free"):
                    e["free"] = [weak["free"][0] if k == 0 else False, weak["free"][1] if k == n_ - 1 else False]
                pieces.append(e)
            k_ = members.index(weak)
            members[k_:k_ + 1] = pieces
            j_ = live.index(weak)
            live[j_:j_ + 1] = pieces
            _rehost(members, {weak["id"]: pieces[0]["id"]})
            rep["split"] += n_ - 1

    # b. members drawn through a column are split at it
    for m in list(live):
        if vert(m) or m.get("arc"):
            continue
        cuts = []
        for c in cols():
            h = _col_cross(m["a"], m["b"], c, profiles, top_tol=_half(profiles[m["prof"]])[0] + 200)
            if h and h[1] + 300 < h[0] < h[2] - h[1] - 300:
                cuts.append((h[0], c))
        if not cuts:
            continue
        cuts.sort(key=lambda t: t[0])
        a, b = np.array(m["a"], float), np.array(m["b"], float)
        L = float(np.linalg.norm(b - a))
        pts = [a] + [a + (b - a) * s / L for s, _ in cuts] + [b]
        i = members.index(m)
        pieces = []
        for k in range(len(pts) - 1):
            e = dict(m, id=f"{m['id']}.{k + 1}", a=[round(float(v), 1) for v in pts[k]], b=[round(float(v), 1) for v in pts[k + 1]],
                     ends=list(m.get("ends") or ["auto", "auto"]))
            e["ends"] = [e["ends"][0] if k == 0 else "auto", e["ends"][1] if k == len(pts) - 2 else "auto"]
            e["_atcol"] = {}
            if k > 0:
                e["_atcol"][0] = cuts[k - 1][1]["id"]                # this end sits on the column it was split at
            if k < len(pts) - 2:
                e["_atcol"][1] = cuts[k][1]["id"]
            if m.get("free"):
                e["free"] = [m["free"][0] if k == 0 else False, m["free"][1] if k == len(pts) - 2 else False]
            pieces.append(e)
        members[i:i + 1] = pieces
        j = live.index(m)
        live[j:j + 1] = pieces
        rep["split"] += 1

    # c. every end runs on along its line to the first member it meets; work point onto that member's centre line
    A = np.array([m["a"] for m in live], float)
    B = np.array([m["b"] for m in live], float)
    E = B - A
    Ls = np.linalg.norm(E, axis=1)
    Eu = E / np.maximum(Ls, 1e-9)[:, None]
    HS = np.array([max(_half(profiles[m["prof"]])) for m in live])
    base_z = min((min(c["a"][2], c["b"][2]) for c in cols()), default=None)
    # a column (drawn a -> b upward) may rise to a member framing into its line above its top
    UPX = np.array([RISE if (iscol(m) and Eu[n_][2] > 0) else 0.0 for n_, m in enumerate(live)])
    # knees, on the drawing as read: two ends of non-parallel members meeting within 100 mm with no third member there
    # (a beam tip carried by a knee brace) - they stay at that point and are mitred against each other
    knees, in_knee = [], set()
    END = [(n_, kk, (A[n_] if kk == 0 else B[n_])) for n_ in range(len(live)) for kk in (0, 1)]
    for a_ in range(len(END)):
        i1, k1, p1 = END[a_]
        if (i1, k1) in in_knee or iscol(live[i1]) or live[i1].get("arc"):
            continue
        for b_ in range(a_ + 1, len(END)):
            i2, k2, p2 = END[b_]
            if i2 == i1 or (i2, k2) in in_knee or iscol(live[i2]) or np.linalg.norm(p1 - p2) > 100:
                continue
            if abs(Eu[i1] @ Eu[i2]) > 0.95:
                continue
            P = (p1 + p2) / 2
            dd = _seg_dists(P, A, B)
            dd[[i1, i2]] = np.inf
            if (dd <= HS + 5).any():
                continue                                          # a third member is there: an ordinary node
            knees.append((live[i1], k1, live[i2], k2, [round(float(v), 1) for v in P]))
            in_knee.update({(i1, k1), (i2, k2)})
            break
    for m, k, h, kh, P in knees:
        m["a" if k == 0 else "b"] = list(P)
        h["a" if kh == 0 else "b"] = list(P)
    landing = {}                                                  # support index -> [(member, end, s along support)]
    caps = {}                                                     # column -> the member that caps its top
    host_of = {}
    explicit = lambda m, k: isinstance((m.get("ends") or [None, None])[k], list)      # joined by the builder: keep
    col_top = lambda m, k: iscol(m) and (m["b"] if k == 1 else m["a"])[2] >= (m["a"] if k == 1 else m["b"])[2]
    order = [(i, k) for i, m in enumerate(live) for k in (0, 1) if not col_top(m, k)] +             [(i, k) for i, m in enumerate(live) for k in (0, 1) if col_top(m, k)]       # column tops last: they only rise
    for i, k in order:
        m = live[i]
        if True:
            if free(m, k) or planecut(m, k) or explicit(m, k) or k in m.get("_butt", ()) or m.get("arc") or (i, k) in in_knee:
                continue                                          # (a rolled member's ends follow its curve; a knee stays put)
            p, q = (A[i], B[i]) if k == 0 else (B[i], A[i])
            d = _unit(p - q)
            if base_z is not None and p[2] <= base_z + 100 and (iscol(m) or d[2] < 0):
                continue                                          # a foot at the base level stands on the base / RC
            atcol = (m.get("_atcol") or {}).get(k)
            if atcol is not None:                                 # split at this column: it is the support
                j = next(n_ for n_, o in enumerate(live) if o["id"] == atcol)
                host = live[j]
                sv = float((p - A[j]) @ Eu[j])
                host_of[(id(m), k)] = host
                landing.setdefault(j, []).append((m, k, sv))
                continue
            if _butted(i, p, d, A, B, Eu):
                continue                                          # already meets an in-line neighbour end to end
            if not iscol(m):
                jc = _in_contact(i, p, A, B, HS, m, profiles, prefer=lambda j: iscol(live[j]))
                if jc is not None and not iscol(live[jc]) and _seg_dists(p, A[jc:jc + 1], B[jc:jc + 1])[0] <= 1.0:
                    continue                                      # placed by the builder on a member's centre line (a generated
                                                                  # truss node): left as built - its "auto" cut is the right one
                if jc is not None:                                # already in place: keep it there, but join it to what it
                    host_of[(id(m), k)] = live[jc]                # touches so it gets the right face (web / cope / near)
                    landing.setdefault(jc, []).append((m, k, float((p - A[jc]) @ Eu[jc])))
                    continue
            skip = {n_ for n_, o in enumerate(live) if any(host_of.get((id(o), kk)) is m for kk in (0, 1))} if col_top(m, k) else set()
            best = _first_hit(i, p, d, A, Eu, Ls, HS, live, iscol, back=0.0 if vert(m) else min(BACK, 0.3 * Ls[i]),   # an overshoot, never most of the member
                              reach=REACH_UP if (iscol(m) and d[2] > 0) else REACH, up_ext=UPX, skip=skip)   # framing in is not a cap
            if best is not None and col_top(m, k) and best[0] < -1:
                best = None                                       # a column top never comes down
            if best is None:
                continue
            t, j, sv, P2 = best
            host = live[j]
            if level(m) and iscol(host):
                P2 = np.array([P2[0], P2[1], p[2]])               # a level member stays at its level on the column axis
            if iscol(m):
                P2 = np.array([p[0], p[1], _z_on(host, p[0], p[1], P2[2])])   # a column end moves only along its own line
            if iscol(host):
                top_k = "b" if host["b"][2] >= host["a"][2] else "a"
                if level(m):
                    need = P2[2] + _half(profiles[m["prof"]])[0]  # a beam into the side: the column rises flush with its top
                else:
                    need = P2[2]                                  # a chord / brace landing on the top: it caps the column
                    if need > host[top_k][2] + 1:
                        caps[id(host)] = m
                if need > host[top_k][2] + 1:
                    host[top_k] = [host[top_k][0], host[top_k][1], round(float(need), 1)]
                    A[j], B[j] = np.array(host["a"], float), np.array(host["b"], float)
                    E_ = B[j] - A[j]; Ls[j] = np.linalg.norm(E_); Eu[j] = E_ / Ls[j]
                    rep["col_tops"] += 1
            if np.linalg.norm(P2 - p) > 0.5:
                rep["run_on"] += 1
            m["a" if k == 0 else "b"] = [round(float(v), 1) for v in P2]
            host_of[(id(m), k)] = host
            landing.setdefault(j, []).append((m, k, sv))

    # d. ends on one support within NODE_TOL share one work point: a brace goes to the node a level member makes there
    #    (beam-column node), braces with no level member meet at their mean point (chevron, V, K)
    for j, ends_ in landing.items():
        host = live[j]
        o = np.array(host["a"], float)
        u = _unit(np.subtract(host["b"], host["a"]))
        Lh = float(np.linalg.norm(np.subtract(host["b"], host["a"])))
        pts = sorted([(float((np.array(m["a" if k == 0 else "b"], float) - o) @ u), m, k) for m, k, _ in ends_]
                     + [(0.0, None, 0), (Lh, None, 1)], key=lambda e: e[0])       # the host's own ends are nodes too
        group = []
        for e in pts + [None]:
            if e is not None and (not group or e[0] - group[-1][0] <= NODE_TOL):
                group.append(e)
                continue
            incl = [g for g in group if g[1] is not None and not level(g[1]) and not vert(g[1])]
            fixed = [g for g in group if g[1] is None or level(g[1]) or vert(g[1])]
            if incl and (fixed or len(incl) > 1):
                s_ = (sum(g[0] for g in fixed) / len(fixed)) if fixed else sum(g[0] for g in incl) / len(incl)
                P = o + u * s_
                for _, m, k in incl:
                    m["a" if k == 0 else "b"] = [round(float(v), 1) for v in P]
                rep["nodes"] += 1
            group = [e] if e is not None else []

    # e. column tops: run up to the member over the column, or come down to the highest member framing in
    for c in cols():
        k = 1 if c["b"][2] > c["a"][2] else 0
        if free(c, k) or (id(c), k) in host_of:
            continue
        top = np.array(c["a" if k == 0 else "b"], float)
        zs = [m_["a" if kk == 0 else "b"][2] + _half(profiles[m_["prof"]])[0] for m_ in live for kk in (0, 1)
              if host_of.get((id(m_), kk)) is c and level(m_)]     # flush with the top of the highest member framing in
        if zs and max(zs) < top[2] - 1 and not _touch_above(c, live, profiles):
            c["a" if k == 0 else "b"] = [float(top[0]), float(top[1]), round(max(zs), 1)]
            rep["col_tops"] += 1

    # hosts move too (their own ends ran on): re-project every joined end onto its host's final centre line until still
    for _ in range(4):
        moved = 0.0
        for m in live:
            for k in (0, 1):
                h = host_of.get((id(m), k))
                if h is None:
                    continue
                p = np.array(m["a"] if k == 0 else m["b"], float)
                a_, b_ = np.array(h["a"], float), np.array(h["b"], float)
                ab = b_ - a_
                t_ = float(np.clip((p - a_) @ ab / max(float(ab @ ab), 1e-9), 0, 1))
                P = a_ + t_ * ab
                if iscol(h) and level(m):
                    P[2] = p[2]
                if iscol(m):                                      # a column end moves only along its own line
                    P = np.array([p[0], p[1], _z_on(h, p[0], p[1], p[2])])
                d_ = float(np.linalg.norm(P - p))
                if d_ > 0.05:
                    m["a" if k == 0 else "b"] = [round(float(v), 1) for v in P]
                    moved = max(moved, d_)
        if moved < 0.5:
            break

    # drop stubs left inside a joint (both ends run to one node)
    stubs = {m["id"] for m in live if math.dist(m["a"], m["b"]) < 100}
    if stubs:
        members[:] = [m for m in members if m["id"] not in stubs]
        live[:] = [m for m in live if m["id"] not in stubs]
        host_of = {kk: h for kk, h in host_of.items() if h["id"] not in stubs}
        rep["stubs"] = sorted(stubs)
        _rehost(members, {s_: None for s_ in stubs})

    for c in live:
        cap = caps.get(id(c))
        if cap is not None and (id(c), 1) not in host_of and (id(c), 0) not in host_of:
            top_k = 1 if c["b"][2] >= c["a"][2] else 0
            host_of[(id(c), top_k)] = cap

    # mitre: inclined ends meeting at one point (chevron / V / K) are cut on the plane bisecting each pair, each keeps its side
    pts_ = {}
    for m in live:
        if vert(m) or level(m):
            continue
        for k in (0, 1):
            if (id(m), k) in host_of:
                pts_.setdefault(tuple(round(v, 1) for v in (m["a"] if k == 0 else m["b"])), []).append((m, k))
    mitres = {}
    for P, grp in pts_.items():
        if len(grp) < 2:
            continue
        for m, k in grp:
            dm = _unit(np.subtract(m["b"] if k == 0 else m["a"], P))
            for o, ko in grp:
                if o is m:
                    continue
                do = _unit(np.subtract(o["b"] if ko == 0 else o["a"], P))
                nrm = _unit(dm - do)                               # points from the other's side to this one's
                if np.linalg.norm(dm - do) < 1e-6:
                    continue
                mitres.setdefault((id(m), k), []).append({"plane": [round(float(c), 6) for c in nrm] + [round(float(nrm @ np.array(P)), 2)]})

    for m, k, h, kh, P in knees:                                  # the knee mitre
        dm = _unit(np.subtract(m["b"] if k == 0 else m["a"], P))
        dh = _unit(np.subtract(h["b"] if kh == 0 else h["a"], P))
        if np.linalg.norm(dm - dh) > 1e-6:
            nm = _unit(dm - dh)
            for o, ko, nn in ((m, k, nm), (h, kh, -nm)):
                e = list(o.get("ends") or ["auto", "auto"])
                e[ko] = [{"plane": [round(float(c), 6) for c in nn] + [round(float(nn @ np.array(P)), 2)]}]
                o["ends"] = e
        rep.setdefault("knees", 0)
        rep["knees"] += 1

    # f. the cut that matches each joined end
    LA = np.array([m["a"] for m in live], float)
    LB = np.array([m["b"] for m in live], float)
    HSL = np.array([max(_half(profiles[m["prof"]])) for m in live])
    for m in live:
        ends = list(m.get("ends") or ["auto", "auto"])
        for k in (0, 1):
            host = host_of.get((id(m), k))
            if host is None or isinstance(ends[k], list):
                continue
            p = m["a"] if k == 0 else m["b"]
            on_axis = _axis_dist(p, host) <= 1.0
            if False:
                pass
            else:                                                 # always explicit: auto would take flange-tip planes
                cuts = [{"host": host["id"], "face": _face(m, k, host, profiles, iscol, level)}]
                body = _unit(np.subtract(m["b"] if k == 0 else m["a"], p))
                near = np.where(_seg_dists(np.array(p, float), LA, LB) <= CUT_TOL)[0]
                for o in (live[n_] for n_ in near):                # the other members meeting at this node, on this side
                    if o is host or o is m or (o.get("rank", 2) >= m.get("rank", 2) and not iscol(m)):
                        continue                                  # (a column top capped at a node: every member there cuts it)
                    da, db = math.dist(p, o["a"]), math.dist(p, o["b"])
                    if min(da, db) <= 1.0:                        # o ends here: only if it lies on the side of m's body
                        far = o["b"] if da <= 1.0 else o["a"]
                        if _unit(np.subtract(far, p)) @ body < -0.2:      # only a member clearly on the far side is skipped
                            continue
                    cuts.append({"host": o["id"], "face": _face(m, k, o, profiles, iscol, level)})
                ends[k] = cuts + mitres.get((id(m), k), [])
        m["ends"] = ends
        m.pop("_butt", None)
        m.pop("_atcol", None)

    # a free end of a member carried along its length (it passes over / through / beside a support) is a drawn cantilever:
    # flagged free with the reason, listed in the report as a cantilever, not floating
    rep["cantilevers"] = []
    for m in live:
        for k in (0, 1):
            if free(m, k) or planecut(m, k) or (id(m), k) in host_of:
                continue
            p = np.array(m["a"] if k == 0 else m["b"], float)
            dd = _seg_dists(p, LA, LB)
            dd[live.index(m)] = np.inf
            if (dd <= HSL + max(_half(profiles[m["prof"]])) * 0.25).any():
                continue                                          # touches something at the end: not free
            if base_z is not None and p[2] <= base_z + 100:
                continue
            if _carried(m, k, live, LA, LB, HSL, profiles, exclude=host_of.get((id(m), 1 - k))):
                f_ = list(m.get("free") or [False, False]); f_[k] = True
                m["free"] = f_
                m["free_why"] = (m.get("free_why", "") + " " if m.get("free_why") else "") + "cantilever past its support (drawn)"
                rep["cantilevers"].append(f"{m['id']} {'ab'[k]} {m.get('ref', '')}")

    # report what is still free (the gate fails on these)
    for m in live:
        for k in (0, 1):
            if free(m, k) or planecut(m, k) or (id(m), k) in host_of:
                continue
            p = m["a"] if k == 0 else m["b"]
            if base_z is not None and p[2] <= base_z + 100:
                continue
            dd = _seg_dists(np.array(p, float), LA, LB)
            me = live.index(m)
            dd[me] = np.inf
            if (dd <= HSL + max(_half(profiles[m["prof"]])) * 0.25).any():
                continue
            rep["free"].append(f"{m['id']} {'ab'[k]} {m.get('ref', '')} at {[round(v) for v in p]}")
    return rep


def _crossed(p, q, x, y, LA, LB, HS, live):
    """Any other member passing through the gap p -> q (closer to the gap's line than its own half size)?"""
    p, q = np.asarray(p, float), np.asarray(q, float)
    for f in (0.25, 0.5, 0.75):
        c = p + (q - p) * f
        dd = _seg_dists(c, LA, LB)
        for j in np.where(dd <= HS + 5)[0]:
            if live[j] is not x and live[j] is not y:
                return True
    return False


def _crossing(x, y, tol=None):
    """x and y pass through each other away from both ends (closest approach under their half sizes, >= 300 mm in from
    every end, at an angle): (point on x, point on y), else None."""
    a, b, c, d = (np.array(v, float) for v in (x["a"], x["b"], y["a"], y["b"]))
    u, v = b - a, d - c
    Lu, Lv = np.linalg.norm(u), np.linalg.norm(v)
    if Lu < 700 or Lv < 700:
        return None
    u, v = u / Lu, v / Lv
    if abs(u @ v) > 0.95:
        return None
    w0 = a - c
    B_, D_, E_ = u @ v, u @ w0, v @ w0
    den = 1 - B_ * B_
    s = (B_ * E_ - D_) / den
    t = (E_ - B_ * D_) / den
    if not (300 < s < Lu - 300 and 300 < t < Lv - 300):
        return None
    P, Q = a + s * u, c + t * v
    if np.linalg.norm(P - Q) > (tol or 30.0):                   # same plane (read noise a few mm)
        return None
    return P, Q


def _rehost(members, renamed):
    """Cuts naming a merged member follow it to the member it merged into; cuts naming a dropped stub are removed."""
    for m in members:
        ends = m.get("ends")
        if not ends:
            continue
        for k, e in enumerate(ends):
            if not isinstance(e, list):
                continue
            new = []
            for c in e:
                h = c.get("host")
                while h in renamed and renamed[h] is not None and renamed[h] != h:
                    h = renamed[h]
                if h in renamed and renamed[h] is None:
                    continue
                if h == m["id"]:
                    continue
                new.append(dict(c, host=h) if "host" in c else c)
            ends[k] = new if new else "auto"


def _z_on(h, x, y, z0):
    """Height of member h's centre line where it passes the vertical line (x, y) (closest point in plan)."""
    a, b = np.array(h["a"], float), np.array(h["b"], float)
    ab = b - a
    L2 = ab[0] ** 2 + ab[1] ** 2
    if L2 < 1e-6:
        return float(z0)
    t = float(np.clip(((x - a[0]) * ab[0] + (y - a[1]) * ab[1]) / L2, 0, 1))
    return float(a[2] + t * ab[2])


def _seg_dists(p, A, B):
    """Distance from point p to every segment A[i]-B[i] (numpy)."""
    AB = B - A
    L2 = np.maximum(np.einsum("ij,ij->i", AB, AB), 1e-9)
    t = np.clip(np.einsum("ij,ij->i", p - A, AB) / L2, 0, 1)
    return np.linalg.norm(p - (A + t[:, None] * AB), axis=1)


def _axis_dist(p, h):
    a, b = np.array(h["a"], float), np.array(h["b"], float)
    ab = b - a
    t = np.clip((np.array(p) - a) @ ab / max(float(ab @ ab), 1e-9), 0, 1)
    return float(np.linalg.norm(np.array(p) - (a + t * ab)))


def _touched(p, m, live, profiles):
    """An end already in contact (another member's body within its half size of the point)."""
    hm = max(_half(profiles[m["prof"]]))
    for o in live:
        if o is m:
            continue
        if _axis_dist(p, o) <= max(_half(profiles[o["prof"]])) + hm * 0.25:
            return True
    return False


def _touch_above(c, live, profiles):
    top = c["b"] if c["b"][2] > c["a"][2] else c["a"]
    return any(o is not c and _axis_dist(top, o) <= max(_half(profiles[o["prof"]])) + 5 for o in live)


def _col_cross(p, q, c, profiles, reach_top=False, top_tol=0.0):
    """Column c crossed by the plan line p -> q at its level: (s of the column axis from p, column half extent along
    the line, plan length) or None."""
    pf = profiles[c["prof"]]
    cx, cy = c["a"][0], c["a"][1]
    z0, z1 = sorted((c["a"][2], c["b"][2]))
    zm = (p[2] + q[2]) / 2
    if not (z0 + 1 < zm < z1 - 1 + top_tol) and not (reach_top and z0 - 1 <= zm <= z1 + max(_half(pf)) + 1):
        return None
    dx, dy = q[0] - p[0], q[1] - p[1]
    L = math.hypot(dx, dy)
    if L < 1:
        return None
    tx, ty = dx / L, dy / L
    hu, hv = _half(pf)
    if abs((cx - p[0]) * ty - (cy - p[1]) * tx) > max(hu, hv):
        return None
    return (cx - p[0]) * tx + (cy - p[1]) * ty, min(hu, hv), L


def _col_between(p, q, cols, profiles):
    for c in cols:
        h = _col_cross(p, q, c, profiles, reach_top=True)
        if h and -h[1] <= h[0] <= h[2] + h[1]:
            return True
    return False


def _carried(m, k, live, LA, LB, HSL, profiles, exclude=None):
    """Member m crosses a support part-way along its body - away from both ends, at an angle - so its free end k is a
    drawn cantilever. The member its other end sits on does not count."""
    a, b = np.array(m["a"], float), np.array(m["b"], float)
    p = a if k == 0 else b
    q = b if k == 0 else a
    L = float(np.linalg.norm(q - p))
    if L < 600:
        return False
    u = (q - p) / L
    hm = max(_half(profiles[m["prof"]]))
    me = live.index(m)
    for f in np.linspace(0.15, 0.85, 15):                    # the body, clear of both ends
        x = p + (q - p) * f
        dd = _seg_dists(x, LA, LB)
        dd[me] = np.inf
        for j in np.where(dd <= HSL + hm + 5)[0]:
            o = live[j]
            if o is exclude:
                continue
            e = LB[j] - LA[j]
            if abs(float(e @ u)) / max(float(np.linalg.norm(e)), 1e-9) < 0.85:   # crosses at an angle (> ~30 deg)
                return True
    return False


def _in_contact(i, p, A, B, HS, m, profiles, prefer=None):
    """Index of the member whose section envelope the end already lies in / on - a column first, else the nearest axis."""
    dd = _seg_dists(p, A, B)
    dd[i] = np.inf
    inside = np.where(dd <= HS + 0.5)[0]
    if not len(inside):
        return None
    if prefer is not None:
        cols_in = [j for j in inside if prefer(j)]
        if cols_in:
            return int(min(cols_in, key=lambda j: dd[j]))
    return int(inside[np.argmin(dd[inside])])


def _butted(i, p, d, A, B, Eu):
    """p already sits on the end of a member in line with this one (a butt / splice joint)."""
    for P in (A, B):
        near = np.where(np.linalg.norm(P - p, axis=1) <= 1.0)[0]
        if any(j != i and abs(Eu[j] @ d) > 0.9995 for j in near):
            return True
    return False


def _first_hit(i, p, d, A, Eu, Ls, HS, live, vert, back=BACK, reach=REACH, up_ext=None, skip=()):
    """First member met by the ray p + t d (t in [-BACK, REACH]): (t, index, s along it, point on its axis)."""
    if up_ext is None:
        up_ext = np.zeros(len(A))
    w0 = p - A                                                # (n, 3)
    b_ = Eu @ d
    dd, ee = w0 @ d, np.einsum("ij,ij->i", Eu, w0)
    den = 1 - b_ * b_
    ok = np.abs(den) > 1e-6
    t = np.where(ok, (b_ * ee - dd) / np.where(ok, den, 1), np.nan)
    s = np.where(ok, (ee - b_ * dd) / np.where(ok, den, 1), np.nan)
    s_raw = s.copy()
    s = np.clip(s, 0, Ls + up_ext)                          # a column's line runs on above its top (it can rise)
    P1 = p + t[:, None] * d
    P2 = A + s[:, None] * Eu
    dist = np.linalg.norm(P1 - P2, axis=1)
    tol = HS + HS[i] * 0.5 + 20
    hit = ok & (dist <= tol) & (t >= -back) & (t <= reach)
    hit[i] = False
    for j_ in skip:
        hit[j_] = False
    if not hit.any():
        return None
    idx = np.where(hit)[0]
    pool = idx                                               # nearest along the line wins (an overshoot comes back)
    key = np.abs(t[pool]) + np.where(t[pool] < -1, 100.0, 0.0) - np.array([60.0 if vert(live[j]) else 0.0 for j in pool])         + np.where(s_raw[pool] > Ls[pool] + 1, 1000.0, 0.0)  # a column wins a near tie; its projected extension loses
    j = int(pool[np.argmin(key)])
    return float(t[j]), j, float(s[j]), P2[j]


def _face(m, k, host, profiles, vert, level):
    """The face an end bears on. Any I host: arriving on its web side -> on to the web between the flanges when the
    member fits ("web"), else the flange tips ("near"); a level beam into a level I girder -> "cope"; else "near"."""
    hp, mp = profiles[host["prof"]], profiles[m["prof"]]
    p, q = (m["a"], m["b"]) if k == 0 else (m["b"], m["a"])
    if hp["type"] != "I":
        return "near"
    a, b = np.array(host["a"], float), np.array(host["b"], float)
    x = _unit(b - a)
    lvl = lambda o: abs(o["a"][2] - o["b"][2]) <= max(LEVEL, 0.18 * math.dist(o["a"], o["b"]))   # within ~10 deg of level
    if lvl(host) and lvl(m) and mp["type"] == "I":
        return "cope"
    default_up = [1.0, 0, 0] if abs(x[2]) > 0.9 else [0, 0, 1.0]       # as geometry.frame
    up = np.array(host.get("up") or default_up, float)
    u = _unit(up - (up @ x) * x)
    v = np.cross(x, u)
    d = _unit(np.subtract(p, q))
    if abs(d @ v) > abs(d @ u):                                       # arrives on the web side
        half_w = _half(mp)[1] if mp["type"] in ("I", "C", "LIPC") else max(_half(mp))
        off = abs((np.array(p, float) - a) @ u)
        return "web" if half_w + off <= hp["h"] / 2 - hp.get("tf", 10) - 1 else "near"
    return "near"


# ---------------- contact repair (at generation, on the cut solids) ----------------
def _solid(pr):
    V = np.array([p for f in pr for p in f], float)
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
    return V, np.array(N), np.array(E)


def _sep(A, B):
    """SAT: > 0 = separation (mm), <= 0 = touching / overlapping."""
    axes = [A[1], B[1]]
    if len(A[2]) and len(B[2]):
        C = np.cross(A[2][:, None, :], B[2][None, :, :]).reshape(-1, 3)
        n = np.linalg.norm(C, axis=1)
        axes.append(C[n > 1e-6] / n[n > 1e-6, None])
    ax = np.vstack(axes)
    pa, pb = A[0] @ ax.T, B[0] @ ax.T
    return float(-(np.minimum(pa.max(0), pb.max(0)) - np.maximum(pa.min(0), pb.min(0))).min())


def _end_probe(part, k, probe_len=40.0, tol=0.5):
    """The member's section box over its last 40 mm at end k (plus tol beyond), as a solid (as the clash check probes)."""
    V = np.array([q for pr in part["pr"] for f in pr for q in f], float)
    x = np.array(part["ax"], float); x /= np.linalg.norm(x)
    u = np.cross(x, [0, 0, 1.0]) if abs(x[2]) < 0.9 else np.cross(x, [1.0, 0, 0]); u /= np.linalg.norm(u)
    v = np.cross(x, u)
    s = V @ x
    if k == 0:
        sel, s0, s1 = s <= s.min() + probe_len, s.min() - tol, s.min() + probe_len
    else:
        sel, s0, s1 = s >= s.max() - probe_len, s.max() - probe_len, s.max() + tol
    R = V[sel]
    cu, cv = R @ u, R @ v
    c = x * (s0 + s1) / 2 + u * (cu.min() + cu.max()) / 2 + v * (cv.min() + cv.max()) / 2
    h = [(s1 - s0) / 2, (cu.max() - cu.min()) / 2 + tol, (cv.max() - cv.min()) / 2 + tol]
    P = np.array([c + a * h[0] * x + b * h[1] * u + d * h[2] * v for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)])
    return P, np.array([x, u, v]), np.array([x, u, v])


def _touches(part, host_part, k=None, tol=0.5):
    """End k of part (or the whole part when k is None) touches the host's solids."""
    S = [_solid(pr) for pr in host_part["pr"]]
    if k is not None and part.get("ax"):
        pb = _end_probe(part, k)
        return any(_sep(pb, h) <= tol for h in S)
    return any(_sep(_solid(pr), h) <= tol for pr in part["pr"] for h in S)


def _which_end(part, m, k):
    """Index (0 = low end along the part's axis) of member end k on the part."""
    x = np.array(part["ax"], float)
    return 0 if (np.array(m["a" if k == 0 else "b"], float) - np.array(m["b" if k == 0 else "a"], float)) @ x < 0 else 1


def contact_repair(members, parts, idx, by_id, rebuild):
    """An end that touches none of its node members after cutting is rebuilt with simpler cuts until it touches one:
    the main support alone, its near face, the mitre alone, each other node member alone (near / web). An "auto" end at
    a generated node tries the lower-rank members through its point one by one (a web foot bearing on the continuous
    chord). Returns the ends repaired."""
    fixed = []
    ids = list(idx)
    A = np.array([by_id[i]["a"] for i in ids], float)
    B = np.array([by_id[i]["b"] for i in ids], float)
    for m in members:
        if m["id"] not in idx or m.get("node") is False:
            continue
        part = parts[idx[m["id"]]]
        if not part.get("ax") or part.get("plate") or (part.get("mto") or {}).get("kind") in ("measure", "piece"):
            continue
        for k in (0, 1):
            if (m.get("free") or [False, False])[k]:
                continue
            e = (m.get("ends") or ["auto", "auto"])[k]
            p = np.array(m["a" if k == 0 else "b"], float)
            if isinstance(e, list):
                hosts = [c["host"] for c in e if "host" in c and c["host"] in idx and not c.get("sep")]
                if not hosts:
                    continue
                trials = [[e[0]], [dict(e[0], face="near")]] if "host" in e[0] else []
                planes = [c for c in e if "plane" in c]
                if planes:
                    trials.append(planes)
                trials += [[{"host": h, "face": f}] for h in hosts[1:] for f in ("near", "web")]
            elif e == "auto":
                near = [ids[j] for j in np.where(_seg_dists(p, A, B) <= 1.0)[0] if ids[j] != m["id"]
                        and by_id[ids[j]].get("rank", 2) < m.get("rank", 2)]
                if not near:
                    continue
                hosts = near
                trials = [[{"host": h, "face": f}] for h in sorted(near, key=lambda h: by_id[h].get("rank", 2)) for f in ("near", "web")]
            else:
                continue
            ka = _which_end(part, m, k)
            if any(_touches(part, parts[idx[h]], ka) for h in hosts):
                continue
            for trial in trials:
                m["ends"][k] = trial
                try:
                    new = rebuild(m)
                except Exception:
                    continue
                h0 = next((c["host"] for c in trial if "host" in c), None)
                ok = any(_touches(new, parts[idx[h]], _which_end(new, m, k)) for h in ([h0] if h0 else hosts))
                if ok:
                    parts[idx[m["id"]]] = new
                    part = new
                    fixed.append(f"{m['id']} {'ab'[k]}")
                    break
            else:
                m["ends"][k] = e                              # nothing better: as it was (the checks report it)
                parts[idx[m["id"]]] = rebuild(m)
                part = parts[idx[m["id"]]]
    return fixed
