#!/usr/bin/env python3
"""Sections from the Consteel 2025 catalogue (generate-mto/references/consteel_catalogue.md) and the member schedule
printed on the drawings (read from the drawing store). Shared by every model.

    import catalogue
    sched = catalogue.schedule(sheets)            # {"B4": "UB 254x146x31", "SC3": "UC 356x406x393", ...} + report
    pf = catalogue.profile("UB 254x146x31")       # geometry.py profile dict: type, h, b, tw, tf, kgm, _m2m, _src

Schedule: every table headed MARKING / MEMBER SIZE on any sheet; a row's marks are split on "/" (B3 / BR5 / TC1 / BC1
share one size); the latest sheet wins when two disagree (reported).
Profile: the drawing's designation (UB / UC D x B x kg, L / UA a x b x t, C / PFC h x b x t, LC h x b x lip x t) is matched to the
catalogue row nearest in kg/m and size; the catalogue's kg/m, D, B are used; flange / web thickness are derived from the
area for the 3D shape only (the catalogue gives kg/m, D, B). A designation not in the catalogue raises (never invented).
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAT = (ROOT / "generate-mto/references/consteel_catalogue.md").read_text(encoding="utf-8").splitlines()
_ROW = re.compile(r"([\d.]+) \(([\d.]+)x([\d.]+)\)(?: ([\d.]+))?")


def _i_rows(depth):
    line = next((l for l in CAT if l.startswith(f"- **{depth}**")), None)
    return [(float(a), float(b), float(c)) for a, b, c, _ in _ROW.findall(line)] if line else []


_M2M = {}
for l in CAT[:12]:
    if "(" in l and "·" in l:
        for a, b, c, d in _ROW.findall(l):
            if d:
                _M2M[(float(a), float(b), float(c))] = float(d)


def _table(head):
    """Lines of one catalogue section (under its ### heading)."""
    i = next((k for k, l in enumerate(CAT) if l.startswith(head)), None)
    out = []
    for l in CAT[i + 1:] if i is not None else []:
        if l.startswith("###"):
            break
        out.append(l)
    return out


def _hollow(kind, dims, t):
    """RHS / SHS / CHS kg/m: hot finished first, then cold formed; nearest thickness. (t, kg/m, source) or None."""
    for fin in ("hot finished", "cold formed"):
        for l in _table(f"### {kind} - {fin}"):
            if l.startswith(f"- {dims}:"):
                vals = dict(re.findall(r"t([\d.]+)=([\d.]+)", l))
                if vals:
                    tt = min(vals, key=lambda k: abs(float(k) - t))
                    return float(tt), float(vals[tt]), (f"Consteel 2025 {kind} {dims} t{tt} ({fin}) {vals[tt]}"
                                                        + ("" if float(tt) == t else f" (drawing t{t:g}: nearest)"))
    return None


def profile(name):
    n = re.sub(r"\s+", " ", name.upper().replace("X", "x")).strip()
    m = re.match(r"(UB|UC) ?(\d+)x(\d+)x([\d.]+)", n)
    if m:
        fam, d, b, kg = m.group(1), int(m.group(2)), int(m.group(3)), float(m.group(4))
        rows = _i_rows(d) or _i_rows({356: 356, 254: 254}.get(d, d))
        if not rows:
            raise KeyError(f"{name}: no catalogue group {d}")
        kgm, D, B = min(rows, key=lambda r: abs(r[0] - kg) / kg + abs(r[2] - b) / b)
        A = kgm / 7.85e-3
        tw = A / (3.2 * B + D)
        m2 = _M2M.get((kgm, D, B))
        return {"type": "I", "h": D, "b": B, "tw": round(tw, 1), "tf": round(1.6 * tw, 1), "kgm": kgm, "_cat": fam, "_m2m": m2,
                "_src": f"Consteel 2025 {fam} {kgm} ({D}x{B})" + ("" if m2 else "; m2/m computed")}
    m = re.match(r"(?:L|UA|EA) ?(\d+)x(\d+)x([\d.]+)", n)
    if m:
        a, b, t = m.groups()
        line = next((l for l in CAT if l.startswith(f"- {a}x{b}:")), None)
        if not line:
            raise KeyError(f"{name}: no angle {a}x{b} in the catalogue")
        vals = dict(re.findall(r"t([\d.]+)=([\d.]+)", line))
        tt = min(vals, key=lambda k: abs(float(k) - float(t)))
        return {"type": "L", "h": float(a), "b": float(b), "t": float(tt), "kgm": float(vals[tt]), "_cat": "L",
                "_src": f"Consteel 2025 angle {a}x{b} t{tt} {vals[tt]}" + ("" if float(tt) == float(t) else f" (drawing t{t}: nearest)")}
    m = re.match(r"LC ?(\d+)x(\d+)x(\d+)x([\d.]+)", n)
    if m:
        h, b, lip, t = map(float, m.groups())
        line = next((l for l in CAT if re.search(rf"\b{int(h)}x{int(b)}x{int(lip)}\b", l) and "t" in l), None)
        kgm = None
        if line:
            vals = dict(re.findall(r"t([\d.]+)=([\d.]+)", line))
            if vals:
                tt = min(vals, key=lambda k: abs(float(k) - t))
                kgm = float(vals[tt])
        if kgm is None:
            kgm = round((h + 2 * b + 2 * lip) * t * 7.85e-3, 2)
        return {"type": "LIPC", "h": h, "b": b, "t": t, "lip": lip, "kgm": kgm, "_cat": "C",
                "_src": f"lipped channel {int(h)}x{int(b)}x{int(lip)} t{t} {kgm} kg/m" + ("" if line else " (computed: not in the catalogue)")}
    m = re.match(r"(RHS|SHS|CHS) ?([\d.]+)(?:x([\d.]+))?x([\d.]+)", n)
    if m:
        kind, a_, b_, t = m.group(1), m.group(2), m.group(3), float(m.group(4))
        if kind == "SHS" and b_ and b_ != a_:
            kind = "RHS"
        if kind != "CHS":
            b_ = b_ or a_
        dims = a_ if kind == "CHS" else f"{a_}x{b_}"
        hit = _hollow(kind, dims, t) or (_hollow("SHS", f"{a_}x{a_}", t) if kind == "RHS" and b_ == a_ else None)
        if not hit:
            raise KeyError(f"{name}: no {kind} {dims} in the catalogue")
        tt, kgm, src = hit
        base = {"CHS": {"type": "CHS", "d": float(a_)}, "SHS": {"type": "SHS", "b": float(a_)},
                "RHS": {"type": "RHS", "h": float(a_), "b": float(b_ or a_)}}[kind]
        return {**base, "t": tt, "kgm": kgm, "_cat": kind, "_src": src}
    m = re.match(r"(?:C|PFC) ?(\d+)x(\d+)x([\d.]+)", n)
    if m:
        h, b, t = map(float, m.groups())
        line = next((l for l in CAT if re.search(rf"\b{int(h)}x{int(b)}\b", l) and "PFC" in l.upper()), None)
        kgm = None
        mk = re.search(r"([\d.]+)\s*kg", name)
        if mk:
            kgm = float(mk.group(1))
        tab = {k: float(v) for k, v in re.findall(r"(\d+x\d+) ([\d.]+)", " ".join(_table("### PFC")))}
        pk = tab.get(f"{int(h)}x{int(b)}")
        if kgm is None and pk and (t > 25 or abs(t - pk) / pk < 0.15):   # PFC 150x75x17.9: the third number is kg/m, not t
            kgm, line = pk, f"PFC {int(h)}x{int(b)} {pk}"
            t = round(kgm / 7.85e-3 / (h + 3 * b), 1)
        if kgm is None:
            kgm = round((h + 2 * b) * t * 7.85e-3, 2)
        return {"type": "C", "h": h, "b": b, "tw": t, "tf": round(1.5 * t, 1), "kgm": kgm, "_cat": "C",
                "_src": f"channel {int(h)}x{int(b)}x{t:g} {kgm} kg/m" + ("" if line else " (drawing kg/m)" if mk else " (computed)")}
    raise KeyError(f"{name}: designation not understood")


def schedule(sheets):
    """Member schedule tables (MARKING / MEMBER SIZE) on every sheet -> {mark: size}; the latest sheet wins, conflicts listed."""
    out, src, conflicts = {}, {}, []
    for sh in sheets:
        heads = [w for w in sh.words if w[4].strip().upper() == "MARKING"]
        for h in heads:
            sz = next((w for w in sh.words if w[4].strip().upper() == "MEMBER SIZE" and abs(w[1] - h[1]) < 3 and 0 < w[0] - h[2] < 200), None)
            if not sz:
                continue
            col = sorted((w for w in sh.words if abs(w[0] - h[0]) < 10 and w[1] > h[3]), key=lambda w: w[1])
            y = h[3]
            for w in col:
                if w[1] - y > 30:
                    break
                y = w[3]
                txt = w[4].strip()
                if not re.fullmatch(r"[A-Z]{1,3}\d{0,3}[a-zA-Z]?(\s*/\s*[A-Z]{1,3}\d{0,3}[a-zA-Z]?)*", txt):
                    break
                cy = (w[1] + w[3]) / 2
                cell = [v for v in sh.words if abs((v[1] + v[3]) / 2 - cy) < 3 and h[2] - 5 <= v[0] <= sz[2] + 40
                        and re.search(r"\d\s*[xX]\s*\d", v[4])]       # under its own MEMBER SIZE heading, reads like a size
                if not cell:
                    continue
                size = re.sub(r"\s*kg/m\.?|\s*kg\.?$", "", " ".join(v[4] for v in sorted(cell, key=lambda v: v[0]))).strip()
                for mk in re.split(r"\s*/\s*", txt):
                    if mk in out and out[mk] != size and src[mk] != sh.page:
                        conflicts.append((mk, out[mk], size, src[mk], sh.page))
                    if mk not in out or sh.page >= src[mk]:
                        out[mk], src[mk] = size, sh.page
        # SCHED-LIST: a "MEMBER SIZE:" heading over lines "SC1 = UB 533x210x82.2 kg/m" (mark, '=', size on one line)
        heads = [(w[0], w) for w in sh.words if re.fullmatch(r"MEMBER\s+SIZES?\s*:", w[4].strip().upper())]   # one phrase
        heads += [(v[0], w) for w in sh.words if re.fullmatch(r"SIZES?\s*:", w[4].strip().upper())          # or two words
                  for v in sh.words if v[4].strip().upper() == "MEMBER" and abs(v[1] - w[1]) < 3 and 0 < w[0] - v[2] < 20]
        for x0, h in heads:
            col = sorted((w for w in sh.words if abs(w[0] - x0) < 8 and w[1] > h[3] - 1), key=lambda w: w[1])
            y = h[3]
            for w in col:
                if w[1] - y > 25:
                    break
                line = sorted((v for v in sh.words if abs((v[1] + v[3]) / 2 - (w[1] + w[3]) / 2) < 3 and v[0] >= w[0] - 1
                               and v[0] - w[2] < 260), key=lambda v: v[0])
                txt = " ".join(v[4] for v in line)
                mm = re.fullmatch(r"([A-Z]{1,3}\d{0,3}[a-zA-Z]?(?:\s*/\s*[A-Z]{1,3}\d{0,3}[a-zA-Z]?)*)\s*=\s*(.+)", txt.strip())
                if not mm:
                    break
                y = w[3]
                size = re.sub(r"\s*kg/m\.?|\s*kg\.?$", "", mm.group(2)).strip()
                size = re.sub(r"(\d)\s*mm\b", r"\1", size)
                if not re.search(r"\d\s*[xX]\s*\d", size):
                    continue                                # a rod / bar ("Ø20 STEEL ROD"): no section to model
                for mk in re.split(r"\s*/\s*", mm.group(1)):
                    if mk in out and out[mk] != size and src[mk] != sh.page:
                        conflicts.append((mk, out[mk], size, src[mk], sh.page))
                    if mk not in out or sh.page >= src[mk]:
                        out[mk], src[mk] = size, sh.page
    # a mark listed twice with two sizes in one table (Job A BR6: B4 / BR6 and BR6 / B17): both kept for a ruling
    return out, {"marks": len(out), "conflicts": conflicts}


def designation(size):
    """'UC 305x305x158' from a schedule cell ('UC 305x305x158 kg/m', 'UB914x419x343.3 kg/m', 'C 100x50x5/8.5-10.2 kg/m')."""
    s = size.upper().replace("X", "x")
    m = re.match(r"(UB|UC|LC|UA|EA|L|PFC|C)\s*([\d.]+x[\d.]+(?:x[\d.]+){0,2})", s)
    return f"{m.group(1)} {m.group(2)}" if m else size
