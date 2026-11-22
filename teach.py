#!/usr/bin/env python3
"""Model Teacher: one command line for the whole teaching loop (any PC, Claude Code or a claude.ai chat).

    python teach.py setup                               install the packages, check the readers (run once)
    python teach.py start JOB PDF [PDF ...] --by NAME   job folder, split into buildings, draft jobs/JOB/job.json
    python teach.py build JOB                           3D model + ruling sheet; numbers logged
    python teach.py note JOB --said "..." [--rule ID] [--what "..."] [--fixture NAME] [--kind correction|rule|note]
                                                        record a correction and what was done with it
    python teach.py answer JOB RULE LETTER --said "..." answer a ruling for every job (rules/rulings.json) + log it
    python teach.py status JOB                          what was taught so far, numbers before / after
    python teach.py pack JOB [--with-model]             zip of everything taught -> out/ (send it to the curator)

Everything taught is kept in jobs/JOB/teaching/log.jsonl; `pack` adds every file changed or added since this kit was
downloaded (compared with BASELINE.json), so nothing has to be pushed anywhere.
"""
import argparse
import datetime
import hashlib
import importlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
CORE = {"pymupdf": "pymupdf", "numpy": "numpy", "shapely": "shapely", "matplotlib": "matplotlib", "PIL": "pillow"}
OPTIONAL = {"rapidocr_onnxruntime": "rapidocr-onnxruntime", "potrace": "potracer"}   # only for scanned / outlined-text sheets
WATCH = ("generate-model", "generate-mto", "rules", "TEACHING.md", "CLAUDE.md", ".claude", "teach.py")
SKIP = ("__pycache__", "/tests/out/", ".pyc", ".tmp")


def sh(*cmd, check=True):
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and r.returncode:
        sys.exit(f"FAILED: {' '.join(map(str, cmd))}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r


def has(mod):
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


def pip(pkgs):
    for extra in ([], ["--break-system-packages"], ["--user"]):
        r = subprocess.run([PY, "-m", "pip", "install", "-q", *extra, *pkgs], capture_output=True, text=True)
        if r.returncode == 0:
            return True, ""
        last = (r.stderr or r.stdout).strip().splitlines()[-1:] or [""]
    return False, last[0]


def setup(a):
    miss = [p for m, p in CORE.items() if not has(m)]
    if miss:
        ok, err = pip(miss)
        importlib.invalidate_caches()
        if not ok or any(not has(m) for m in CORE):
            sys.exit(f"SETUP BLOCKED: could not install {', '.join(miss)} ({err}).\n"
                     "claude.ai chat: Settings > Capabilities > Code execution: allow network egress (package managers), then start a NEW chat.\n"
                     "Own PC: python -m pip install -r requirements.txt")
    opt = [p for m, p in OPTIONAL.items() if not has(m)]
    if opt:
        pip(opt)
    importlib.invalidate_caches()
    ocr = all(has(m) for m in OPTIONAL)
    g = sh(PY, "generate-model/tests/golden_check.py", check=False)
    last = (g.stdout.strip().splitlines() or ["(no output)"])[-1]
    if g.returncode:
        sys.exit(f"SETUP FAILED: golden set does not pass on a fresh kit: {last}\n{g.stdout[-1500:]}{g.stderr[-1500:]}")
    print(f"SETUP OK: python {sys.version.split()[0]}, core packages in, OCR for scanned sheets {'in' if ocr else 'NOT installed (vector PDFs only)'}; {last}")


def log(job, **e):
    p = ROOT / "jobs" / job / "teaching" / "log.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    n = sum(1 for _ in p.open(encoding="utf-8")) + 1 if p.exists() else 1
    e = {"n": n, "ts": datetime.datetime.now().isoformat(timespec="seconds"), **e}
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return e


def entries(job):
    p = ROOT / "jobs" / job / "teaching" / "log.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def start(a):
    J = ROOT / "jobs" / a.job
    orig = J / "Drawings" / "Original Drawings"
    orig.mkdir(parents=True, exist_ok=True)
    for f in a.pdf:
        f = Path(f)
        if not f.exists():
            sys.exit(f"no such file: {f}")
        if f.resolve() != (orig / f.name).resolve():
            shutil.copy2(f, orig / f.name)
    cmd = [PY, "generate-model/split_drawings.py", "--job", a.job, "--no-lean"]
    if not all(has(m) for m in OPTIONAL):
        cmd += ["--no-ocr", "--no-vectorize"]
    r = sh(*cmd, check=False)
    org = J / "Drawings" / "Organized Drawings"
    man = org / "manifest.json"
    pdf, why = None, ""
    if r.returncode == 0 and man.exists():
        m = json.loads(man.read_text(encoding="utf-8"))
        bl = m.get("buildings") or []
        pick = next((b for b in bl if b.get("name") == m.get("primary") or b.get("slug") == m.get("primary")), bl[0] if bl else None)
        cands = sorted(org.glob("[0-9][0-9]_*.pdf"))
        if pick:
            pdf = next((c for c in cands if pick.get("slug", "#") in c.name), cands[0] if cands else None)
            why = f"primary building {pick.get('name')} of {len(bl)}: {m.get('primary_why', '')}"
        print("split: " + "; ".join(f"{c.name}" for c in cands))
    else:
        print(f"split did not run cleanly ({(r.stderr or r.stdout).strip().splitlines()[-1:]}): the whole PDF is read as one building")
    if pdf is None:
        org.mkdir(parents=True, exist_ok=True)
        pdf = org / "01_building.pdf"
        src = sorted(orig.glob("*.pdf"))
        if len(src) == 1:
            shutil.copy2(src[0], pdf)
        else:
            import pymupdf
            out = pymupdf.open()
            for s in src:
                out.insert_pdf(pymupdf.open(s))
            out.save(pdf)
        why = "no split: every sheet given"
    cfg_p = J / "job.json"
    if not cfg_p.exists():
        cfg = json.loads((ROOT / "generate-model/templates/job.json").read_text(encoding="utf-8"))
        cfg.pop("_help", None)
        cfg.update({"title": pdf.stem.split("_", 1)[-1].replace("-", " ").title(), "pdf": str(pdf.relative_to(J)).replace("\\", "/")})
        cfg_p.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    log(a.job, kind="start", by=a.by, said=f"drawings {', '.join(Path(f).name for f in a.pdf)}", what=f"building {pdf.name} ({why})")
    print(f"START OK: jobs/{a.job}/job.json drafted for {pdf.name} ({why}). Fill grids / datum / marks from the drawings "
          f"(TEACHING.md step 2), then: python teach.py build {a.job}")


def build(a):
    r = subprocess.run([PY, "generate-model/build_job.py", a.job], cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True)
    print(r.stdout[-6000:])
    if r.returncode:
        print(r.stderr[-3000:], file=sys.stderr)
        log(a.job, kind="build", what="FAILED: " + ((r.stdout + r.stderr).strip().splitlines() or [""])[-1][:300])
        sys.exit(1)
    s = json.loads((ROOT / "jobs" / a.job / "Models/Models Data/summary.json").read_text(encoding="utf-8"))
    prev = next((e["numbers"] for e in reversed(entries(a.job)) if e.get("kind") == "build" and e.get("numbers")), None)
    log(a.job, kind="build", numbers=s)
    if prev:
        print("vs last build: " + ", ".join(f"{k} {prev.get(k)} -> {s[k]}" for k in ("parts", "t", "modelled", "lost") if prev.get(k) != s[k]) or "no change")


def note(a):
    e = log(a.job, kind=a.kind, said=a.said, rule=a.rule, what=a.what, fixture=a.fixture, files=a.files)
    print(f"logged T{e['n']:02d} ({a.kind})")


def answer(a):
    W = ROOT / "jobs" / a.job / "Models/_work"
    r = sh(PY, "generate-model/rulings.py", "answer", str(W), a.rule, a.letter, "--note", a.said, check=False)
    print(r.stdout.strip() or r.stderr.strip())
    if r.returncode:
        sys.exit(1)
    e = log(a.job, kind="answer", said=a.said, rule=a.rule, what=f"answer {a.letter}")
    print(f"logged T{e['n']:02d}; rebuild to see it: python teach.py build {a.job}")


def status(a):
    E = entries(a.job)
    if not E:
        sys.exit(f"nothing logged for {a.job}")
    for e in E:
        num = e.get("numbers")
        txt = (f"{num['parts']} parts {num['t']} t, modelled {num['modelled']}, lost {num['lost']}" if num else
               " | ".join(str(e[k]) for k in ("said", "rule", "what", "fixture") if e.get(k)))
        print(f"T{e['n']:02d} {e['ts'][:16]} {e['kind']:<10} {txt[:160]}")
    b = [e["numbers"] for e in E if e.get("numbers")]
    if len(b) > 1:
        print(f"first -> last build: {b[0]['parts']} -> {b[-1]['parts']} parts, {b[0]['t']} -> {b[-1]['t']} t, lost {b[0]['lost']} -> {b[-1]['lost']}")


def norm_hash(p):
    b = p.read_bytes()
    if b"\0" not in b[:4096]:
        b = b.replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()


def changed():
    base = json.loads((ROOT / "BASELINE.json").read_text(encoding="utf-8"))
    out = []
    for w in WATCH:
        p = ROOT / w
        for f in ([p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file()) if p.exists() else []):
            rel = f.relative_to(ROOT).as_posix()
            if any(s in "/" + rel for s in SKIP):
                continue
            if base["files"].get(rel) != norm_hash(f):
                out.append((rel, "changed" if rel in base["files"] else "added"))
    gone = [(k, "deleted") for k in base["files"] if not (ROOT / k).exists()]
    return base, out + gone


def pack(a):
    J = ROOT / "jobs" / a.job
    E = entries(a.job)
    if not E:
        sys.exit(f"nothing logged for {a.job}: teach first")
    g = sh(PY, "generate-model/tests/golden_check.py", check=False)
    gl = (g.stdout.strip().splitlines() or [""])[-1]
    base, ch = changed()
    by = next((e.get("by") for e in E if e.get("by")), "unknown")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    out = ROOT / "out" / f"teaching_{''.join(c if c.isalnum() else '-' for c in by)}_{a.job}_{stamp}.zip"
    out.parent.mkdir(exist_ok=True)
    B = [e["numbers"] for e in E if e.get("numbers")]
    rep = [f"# Teaching pack: {a.job}", f"by {by}, {stamp}; kit {base.get('kit', '?')}; golden set: {gl}", ""]
    if B:
        rep += [f"Model: first build {B[0]['parts']} parts {B[0]['t']} t (lost {B[0]['lost']}) -> last {B[-1]['parts']} parts {B[-1]['t']} t "
                f"(lost {B[-1]['lost']})", ""]
    rep += ["## What was taught"] + [f"- T{e['n']:02d} {e['kind']}: " + " | ".join(f"{k}: {e[k]}" for k in ("said", "rule", "what", "fixture", "files")
                                                                          if e.get(k)) for e in E if e["kind"] not in ("build",)]
    rep += ["", "## Files changed or added (full copies under files/)"] + [f"- {s}: {r}" for r, s in ch]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("REPORT.md", "\n".join(rep) + "\n")
        z.writestr("manifest.json", json.dumps({"job": a.job, "by": by, "kit": base.get("kit"), "golden": gl, "changes": ch, "builds": B}, indent=1))
        z.write(J / "teaching" / "log.jsonl", "teaching/log.jsonl")
        for f in (J / "job.json", J / "Models/Models Data/summary.json", J / "Models/Models Data/count_check.md", J / "Models/_work/rulings/cases.json"):
            if f.exists():
                z.write(f, "job/" + f.name)
        for r, s in ch:
            if s != "deleted":
                z.write(ROOT / r, "files/" + r)
        if a.with_model:
            for f in (J / "Models").glob("*.html"):
                z.write(f, "job/" + f.name)
    log(a.job, kind="pack", what=out.name)
    print(f"PACK OK: {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.1f} MB): {len([1 for e in E if e['kind'] != 'build'])} log entries, "
          f"{len(ch)} files changed / added; golden {gl}. Send this zip to the curator.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("setup").set_defaults(fn=setup)
    p = sp.add_parser("start"); p.add_argument("job"); p.add_argument("pdf", nargs="+"); p.add_argument("--by", required=True); p.set_defaults(fn=start)
    p = sp.add_parser("build"); p.add_argument("job"); p.set_defaults(fn=build)
    p = sp.add_parser("note"); p.add_argument("job"); p.add_argument("--said", required=True); p.add_argument("--rule"); p.add_argument("--what")
    p.add_argument("--fixture"); p.add_argument("--files", nargs="*"); p.add_argument("--kind", default="correction", choices=["correction", "rule", "note", "decision"])
    p.set_defaults(fn=note)
    p = sp.add_parser("answer"); p.add_argument("job"); p.add_argument("rule"); p.add_argument("letter"); p.add_argument("--said", default="")
    p.set_defaults(fn=answer)
    p = sp.add_parser("status"); p.add_argument("job"); p.set_defaults(fn=status)
    p = sp.add_parser("pack"); p.add_argument("job"); p.add_argument("--with-model", action="store_true"); p.set_defaults(fn=pack)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
