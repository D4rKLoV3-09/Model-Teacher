#!/usr/bin/env python3
"""Model revisions (the curator 2026-11-23): a model is never replaced. Each build saves a new file named <name>_R01.html,
<name>_R02.html, ... beside the earlier ones; the MTO of that build takes the same number (<name>_MTO_R02.xlsx).

    import revision
    page, xlsx = revision.next_set(PAGE_DIR / "Job A_Turbine_Building.html", MTO_DIR / "01_turbine-building_MTO.xlsx")
    python generate-model/revision.py next PATH [PATH ...]      print the next revision's paths (one number for all)
    python generate-model/revision.py latest PATH                print the newest saved revision of PATH (or nothing)

A path may be given with or without its _Rnn ending; the base name is what counts.
"""
import re
import sys
from pathlib import Path

REV = re.compile(r"^(?P<base>.+?)_R(?P<n>\d{2,})$")


def base(path):
    """Job A_Turbine_Building_R03.html -> Job A_Turbine_Building.html (as a Path)."""
    p = Path(path)
    m = REV.match(p.stem)
    return p.with_name((m.group("base") if m else p.stem) + p.suffix)


def number(path):
    m = REV.match(Path(path).stem)
    return int(m.group("n")) if m else None


def revisions(path):
    """Saved revisions of path's base name, oldest first: [(n, Path)]."""
    b = base(path)
    if not b.parent.exists():
        return []
    out = []
    for p in b.parent.glob(f"{b.stem}_R*{b.suffix}"):
        m = REV.match(p.stem)
        if m and m.group("base") == b.stem:
            out.append((int(m.group("n")), p))
    return sorted(out)


def latest(path):
    r = revisions(path)
    return r[-1][1] if r else None


def next_set(*paths):
    """One revision number for every output of a build (page, MTO, ...): one past the highest already saved for any of
    them -> the new paths, none of which exists."""
    n = max([0] + [k for p in paths for k, _ in revisions(p)]) + 1
    return tuple(base(p).with_name(f"{base(p).stem}_R{n:02d}{base(p).suffix}") for p in paths) if len(paths) > 1 else \
        base(paths[0]).with_name(f"{base(paths[0]).stem}_R{n:02d}{base(paths[0]).suffix}")


def check_new(path):
    """The reason a build may not write path (it must be a new _Rnn file), or None."""
    p = Path(path)
    if number(p) is None:
        return f"{p.name}: a model is saved as a new revision <name>_Rnn{p.suffix} (generate-model/revision.py next_set)"
    if p.exists():
        return f"{p.name} exists: a model is never replaced, save the next revision ({next_set(p).name})"
    return None


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "next":
        r = next_set(*sys.argv[2:])
        print("\n".join(str(p) for p in (r if isinstance(r, tuple) else (r,))))
    elif len(sys.argv) == 3 and sys.argv[1] == "latest":
        print(latest(sys.argv[2]) or "")
    else:
        print(__doc__)
        sys.exit(2)
