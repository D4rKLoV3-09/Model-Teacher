#!/usr/bin/env python3
"""Count check per view (the curator: N members drawn in a view = N members modelled).

    import reconcile
    rep = reconcile.run(views, fate, members, out_json, out_md)

views    register views (plans + elevations / sections / truss placements), each {sheet, title, where?, status, rows}
fate     register row id -> why it is not a member of the model on its own (set by assemble / build: "R02 unmarked", "R08",
         "no section", "column <mark>", "not this building" ...)
members  the final members (after frame.py), each with "ids" = the register rows it was built from

Every drawn member row (a row that is not "not a member") must end in exactly one of:
  modelled   its id is in a final member (alone, or merged with its copy in another view: one member, several views)
  left out   a fate with its reason (listed per view)
A row with neither is LOST: the check fails and names the sheet, the view and the row.
A view not placed counts all its drawn rows as unread (fails too, unless its fate says it is not this building).
"""
import json
from collections import Counter
from pathlib import Path


def _drawn(r):
    return not str(r.get("status", "")).startswith("not a member")


def run(views, fate, members, out_json, out_md=None, skip_views=()):
    built = {}
    for m in members:
        for i in m.get("ids") or []:
            built.setdefault(i, m["id"])
    table, lost_all, tot = [], [], Counter()
    for v in views:
        name = f"p{v['sheet']} {v['title'][:60]}" + (f" @ {v['where']}" if v.get("where") else "")
        skip = next((why for pat, why in skip_views if pat in v["title"]), None)
        if v["status"] != "read":
            table.append({"view": name, "drawn": None, "modelled": 0, "left": {}, "lost": [], "status": skip or f"NOT PLACED: {v['status'][:80]}"})
            if not skip:
                lost_all.append(f"{name}: view not placed")
            continue
        rows = [r for r in v["rows"] if _drawn(r)]
        mod, left, lost = 0, Counter(), []
        for r in rows:
            if r["id"] in built:
                mod += 1
            elif r["id"] in fate:
                left[fate[r["id"]]] += 1
            elif skip:
                left[skip] += 1
            else:
                lost.append(f"{r['id']} {r.get('mark') or '(no mark)'}")
        marked = sum(1 for r in rows if r.get("mark"))
        tot.update(drawn=len(rows), marked=marked, modelled=mod, lost=len(lost), **{"left": sum(left.values())})
        table.append({"view": name, "drawn": len(rows), "marked": marked, "modelled": mod, "left": dict(left), "lost": lost,
                      "status": "OK" if not lost else "LOST"})
        lost_all += [f"{name}: {x}" for x in lost]
    rep = {"verdict": "PASS" if not lost_all else "FAIL", "totals": dict(tot), "lost": lost_all, "views": table}
    Path(out_json).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    if out_md:
        L = ["# Count check per view", "", f"**{rep['verdict']}**: drawn {tot['drawn']} (marked {tot['marked']}), modelled {tot['modelled']}, "
             f"left out with a reason {tot['left']}, lost {len(lost_all)}", "",
             "| View | Drawn | Marked | Modelled | Left out (reason) | Lost |", "|---|---|---|---|---|---|"]
        for t in table:
            lf = "; ".join(f"{k} {n}" for k, n in sorted(t["left"].items(), key=lambda kv: -kv[1]))
            L.append(f"| {t['view']} | {t['drawn'] if t['drawn'] is not None else '-'} | {t.get('marked', '-')} | {t['modelled']} | "
                     f"{lf or ('-' if t['drawn'] is not None else t['status'])} | {len(t['lost']) or ''} |")
        Path(out_md).write_text("\n".join(L) + "\n", encoding="utf-8")
    return rep
