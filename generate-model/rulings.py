#!/usr/bin/env python3
"""the curator's rulings: every unsure call in ANY kind of work (reading a drawing, an MTO row, a crane pick, a truss reading,
an erection step, a deck layout) is one rule per kind of case, answered once by the curator with a letter and applied by every
later job (rules/rulings.json, in the repo). The job builds with the default; the ruling sheet shows the decisions taken
(job book, D01 ...) and every open case with its options and their effect, so the curator changes either by number or letter.

    python generate-model/rulings.py add-rule MTO-R01 --q "..." --opt "A=..." --opt "B=..." --default A   a new kind of case

    python generate-model/rulings.py sheet  WORK [--out WORK/rulings/rulings.html]   the job's ruling sheet (cards)
    python generate-model/rulings.py answer R01 A [--note "..."]                     record the curator's letter
    python generate-model/rulings.py answer WORK C4 A | WORK C4.2 B                   by category / by sample of the sheet
    python generate-model/rulings.py list                                            rules, defaults, answers

In a script:
    import rulings
    pick = rulings.choice("R01")             # the curator's answer, else the rule's default (the cautious option)
    rulings.case(WORK, "R01", sheet=25, box=[x0, y0, x1, y1], readings={"A": [...], "B": [...]}, note="p25-P2-L016")
A reading is a list of drawn items in sheet points: {"line": [x0, y0, x1, y1], "text": "B8"} or {"cross": [x, y]}.
The sheet (ruling_sheet.py) groups decisions and cases by kind of drawing situation (C1-C8): per category a count, the
tonnage of each option and a few samples (drawing crop + 3D still as built); answered rules are not shown again.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULES = ROOT / "rules" / "rulings.json"

DEFAULT_RULES = {
    "R01": {"q": "A line drawn with no mark runs on in line from a marked member (touching it, same direction). What is it?",
            "options": {"A": "The same member: the mark carries on along the line (same section)",
                        "B": "A different member whose section is not given: leave it out and list it",
                        "C": "Not steel (outline of something else)"}, "default": "A"},
    "R02": {"q": "A line drawn with no mark frames between members (not in line with a marked one). What is it?",
            "options": {"A": "Steel: same section as the nearest marked member parallel to it in the view",
                        "B": "Steel, section not given: leave it out and list it",
                        "C": "Not steel (handrail, grating edge, equipment, opening)"}, "default": "B"},
    "R03": {"q": "A short piece (under 1 m) with no mark at a column, between the column and a beam. What is it?",
            "options": {"A": "A stub / bracket of the column (part of the column assembly, section of the beam it carries)",
                        "B": "Drawing detail only (the beam runs to the column; nothing extra)",
                        "C": "A separate member, section not given: leave it out and list it"}, "default": "B"},
    "R04": {"q": "A bracing mark with (TYP) on a plan: which braces does it cover?",
            "options": {"A": "Every brace of the same pattern on that plan, also in bays of a different size",
                        "B": "Only braces of the same length as the labelled one"}, "default": "A"},
    "R05": {"q": "A plan titled with a level range (… BOS TO … TOS): at which level are its members?",
            "options": {"A": "At the upper (TOS) level of the title",
                        "B": "Each at the level of the elevation / section members it meets (TOS level when none)"}, "default": "B"},
    "R06": {"q": "A member is drawn both on a plan and on an elevation, with different end points. Which ends are right?",
            "options": {"A": "The longer of the two (union)", "B": "The elevation's ends", "C": "The plan's ends"}, "default": "A"},
    "R07": {"q": "Stair stringers (SB marks) drawn on a plan: how are they modelled?",
            "options": {"A": "As drawn on the plan, level at the plan's level",
                        "B": "Left out of this model and listed (the stair's slope comes from its section, not read yet)",
                        "C": "Sloped between the two levels written at the stair"}, "default": "B"},
    "R08": {"q": "A read member has an end that meets nothing within reach (its support is not in the reading). What to do?",
            "options": {"A": "Keep it: run the end on to the nearest steel or RC below / beside it",
                        "B": "Leave the member out and list it (until its support is read)",
                        "C": "Keep it as a cantilever (free end)",
                        "D": "Leave it out only when neither end meets anything (an isolated line); one loose end: keep it, that end listed "
                             "floating (Cautious)"}, "default": "D", "basis": "precedent D26 / D32 (Turbine)"},
    "SCALE-R01": {"q": "A view's grid bubbles all agree on one scale that differs from the scale written under its title. Which scale?",
                  "options": {"A": "The bubbles' scale (the drawing is placed by its grids)", "B": "The written scale (the view is refused)"}, "default": "A"},
    "COL-R01": {"q": "A column label whose leader reaches no symbol, and a column symbol no label reaches, on the column layout. Pair them?",
                "options": {"A": "Yes: the label goes to the nearest symbol nobody else names", "B": "No: the symbol is left out and listed"}, "default": "A"},
    "R10": {"q": "A member runs over a support and stops a short way past it (e.g. a truss chord 500 past its eave column). What is the end?",
            "options": {"A": "A drawn overhang: kept as a cantilever end", "B": "An unsupported end: the member is left out (R08)"}, "default": "A"},
    "NAME-R01": {"q": "A member named in words (CRANE RUNWAY BEAM TYP.) whose leader is not followed to a line. Which line is it?",
                 "options": {"A": "The nearest long level line running past the label (and the lines in line with it, TYP)", "B": "Left out and listed"},
                 "default": "A"},
    "R11": {"q": "An unmarked level line continues a marked one in the next bay, broken only at a support (a gap up to 1.2 m at a column). What is it?",
            "options": {"A": "The same member in the next bay (same mark)", "B": "Left out and listed (size not given)"}, "default": "A"},
    "R12": {"q": "An unmarked line (1 m or longer) on an elevation or section has both ends on marked members (drawn framing). What is it?",
            "options": {"A": "Steel: the section of the nearest marked member of the same direction in the view (ASSUMED)",
                        "B": "Left out and listed (size not given)"}, "default": "A", "basis": "precedent D25 (Turbine)"},
    "VIEW-R05": {"q": "A truss view has no level marker and no grid of its own across it (p3 bracing trusses SECTION FOR ROOF TRUSS "
                          "B1 TO B5 / C1 & C5). Where along the main trusses does it stand?",
                 "options": {"A": "Where the main trusses it names (ROOF TRUSS A / A1 / A2 at its posts) are as deep as it is (its written "
                                   "depth, else scaled): at that panel point, nearest its plan label",
                             "B": "At its label on the roof plan"}, "default": "A"},
    "R09": {"q": "Two different marks are read on exactly the same line in two views (e.g. B4 on the plan, BR2 on a section). Which is it?",
            "options": {"A": "One member: the mark whose label sits closest to the line", "B": "Two members (both kept)"}, "default": "A"},
}


def load():
    R = json.loads(RULES.read_text(encoding="utf-8")) if RULES.exists() else {"rules": {}}
    for k, v in DEFAULT_RULES.items():
        R["rules"].setdefault(k, {**v, "answer": None})
    return R


def save(R):
    RULES.write_text(json.dumps(R, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def rule(rid, q, options, default):
    """Register a kind of case (any agent: ids by area, e.g. MTO-R01, LIFT-R01, TRUSS-R01, SEQ-R01, PPT-R01). Kept if it exists."""
    R = load()
    if rid not in R["rules"]:
        assert default in options, f"{rid}: default {default} not an option"
        R["rules"][rid] = {"q": q, "options": options, "default": default, "answer": None}
        save(R)
    return choice(rid)


def choice(rule):
    r = load()["rules"][rule]
    return r.get("answer") or r["default"]


def answered(rule):
    return bool(load()["rules"][rule].get("answer"))


def _cases(p):
    """The job's cases; an empty / half-written file (a run stopped mid-write) reads as none."""
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    except ValueError:
        return []


def _write(p, C):
    """Write via a temp file + rename, so a stopped run never leaves an empty cases file."""
    t = p.with_suffix(".tmp")
    t.write_text(json.dumps(C, indent=1), encoding="utf-8")
    t.replace(p)


def case(work, rule, sheet=None, box=None, readings=None, note="", key=None, effect=None, pdf=None):
    """Add one case to the job's ruling sheet (kept once per key).
    Drawing case: sheet + box (pt) + readings {letter: [drawn items]} (pdf = the PDF when there is no drawing store).
    Any other case (MTO row, crane pick, truss reading, step, slide): readings {letter: "what that option builds"}.
    effect {letter: "+5.6 t" / "+1 crane day" / ...}: what each option changes."""
    p = Path(work) / "rulings" / "cases.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    C = _cases(p)
    key = key or f"{rule}:{sheet}:{note}"
    if not any(c["key"] == key for c in C):
        C.append({"key": key, "rule": rule, "sheet": sheet, "box": [round(v, 1) for v in box] if box else None,
                  "readings": readings or {}, "note": note, "effect": effect or {}, "pdf": str(pdf) if pdf else None})
        _write(p, C)


def reset_cases(work, rules=None):
    """Clear the job's cases (only those of `rules` when given: a script clears the cases it makes itself)."""
    p = Path(work) / "rulings" / "cases.json"
    if not p.exists():
        return
    if rules is None:
        p.unlink()
        return
    _write(p, [c for c in _cases(p) if c["rule"] not in rules])


def sheet(work, out=None, parts=None):
    """The job's ruling sheet, grouped by kind of drawing situation (ruling_sheet.py)."""
    import ruling_sheet
    work = Path(work)
    cp = work / "rulings" / "cases.json"
    C = _cases(cp)
    out = Path(out or work / "rulings" / "rulings.html")
    n = ruling_sheet.build(work, load()["rules"], C, ROOT, out, parts)
    print(f"ruling sheet: {n['decisions']} decision(s), {n['cases']} open case(s), {n['samples']} sample(s), {n['kb']} kB -> {out}")
    return n


def answer(work, key, letter, note=""):
    """the curator's letter for a rule (R08), a sample (C4.2: its rule) or a category (C4: every open rule in it with that option)."""
    import re as _re
    R = load()
    letter = letter.upper()
    if _re.fullmatch(r"C\d+\.\d+", key):
        rids = [json.loads((Path(work) / "rulings/samples.json").read_text(encoding="utf-8"))[key]]
    elif _re.fullmatch(r"C\d+", key):
        rids = [r for r in json.loads((Path(work) / "rulings/categories.json").read_text(encoding="utf-8")).get(key, [])
                if letter in R["rules"][r]["options"] and not R["rules"][r].get("answer")]
    else:
        rids = [key]
    for rid in rids:
        assert letter in R["rules"][rid]["options"], f"{rid} has no option {letter}"
        R["rules"][rid].update(answer=letter, note=note)
        print(f"{rid} = {letter}: {R['rules'][rid]['options'][letter]}")
    save(R)
    return rids


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sheet", "answer", "list", "add-rule"])
    ap.add_argument("--q")
    ap.add_argument("--opt", action="append", default=[])
    ap.add_argument("--default")
    ap.add_argument("args", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--note", default="")
    ap.add_argument("--parts")
    a = ap.parse_args()
    if a.cmd == "sheet":
        sheet(a.args[0], a.out, a.parts)
    elif a.cmd == "answer":
        if len(a.args) == 3:                                    # answer WORK C4 A (category / sample needs the job)
            answer(a.args[0], a.args[1], a.args[2], a.note)
        else:
            answer(None, a.args[0], a.args[1], a.note)
    elif a.cmd == "add-rule":
        opts = dict(o.split("=", 1) for o in a.opt)
        rule(a.args[0], a.q, opts, a.default)
        print(f"{a.args[0]}: default {a.default}; " + "; ".join(f"{k} {v}" for k, v in opts.items()))
    else:
        for k, v in load()["rules"].items():
            print(f"{k} answer {v.get('answer') or '-'} (default {v['default']}): {v['q'][:90]}")
