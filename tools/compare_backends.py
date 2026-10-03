"""End-to-end check that the C++ CLI (cpp/build/faqem) reproduces the Python CLI (cli.py) exactly.

    .venv/bin/python tools/compare_backends.py                       # 7 SKUs x Low/Medium/High/Ultra
    .venv/bin/python tools/compare_backends.py model.glb --modes High "--faces 5000" "--faces 5000 --fast"

For every input and mode both CLIs run into separate folders. Then the outputs are compared:
the printed face count and fidelity line, the .obj text (byte for byte), and every GLB's accessor
arrays, material values and decoded texture pixels. Exit code 1 if anything differs.
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tests.test_cpp_parity import _glb_arrays, same  # noqa: E402

PY = os.path.join(ROOT, ".venv", "bin", "python")
CPP = os.path.join(ROOT, "cpp", "build-parity", "faqem")  # the Python-parity build
DEFAULT_INPUTS = sorted(glob.glob(os.path.join(ROOT, "checking copy", "*.glb")))
DEFAULT_MODES = ["--auto Low", "--auto Medium", "--auto High", "--auto Ultra"]


def run(cmd, out_dir):
    t = time.perf_counter()
    p = subprocess.run(cmd + ["-o", out_dir], capture_output=True, text=True,
                       env=dict(os.environ, FAQEM_BACKEND="python"))
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{p.stdout}\n{p.stderr}")
    return p.stdout, time.perf_counter() - t


def summary(stdout):
    faces = re.search(r"\n([\d,]+) -> ([\d,]+) faces", stdout)
    fid = re.search(r"fidelity: (.*)", stdout)
    best = re.search(r"best for this face count: (.*)", stdout)
    return (faces.group(0).strip() if faces else None, fid.group(1) if fid else None,
            best.group(1) if best else None)


def compare_dirs(a, b):
    problems = []
    names = sorted(os.listdir(a))
    if names != sorted(os.listdir(b)):
        problems.append(f"file sets differ: {names} vs {sorted(os.listdir(b))}")
    for n in names:
        pa, pb = os.path.join(a, n), os.path.join(b, n)
        if not os.path.exists(pb):
            continue
        if n.endswith(".obj"):
            if open(pa, "rb").read() != open(pb, "rb").read():
                problems.append(f"{n}: text differs")
        elif n.endswith(".glb"):
            ga, gb = _glb_arrays(pa), _glb_arrays(pb)
            if len(ga) != len(gb):
                problems.append(f"{n}: {len(ga)} vs {len(gb)} primitives")
                continue
            for x, y in zip(ga, gb):
                for k in set(x) | set(y):
                    if k not in x or k not in y:
                        problems.append(f"{n}: {k} only on one side")
                    elif k in ("material", "mode"):
                        if x[k] != y[k]:
                            problems.append(f"{n}: {k} {x[k]} vs {y[k]}")
                    elif (x[k] is None) != (y[k] is None) or (x[k] is not None and not same(x[k], y[k])):
                        problems.append(f"{n}: {k} differs")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", default=DEFAULT_INPUTS)
    ap.add_argument("--modes", nargs="+", default=DEFAULT_MODES, help='CLI argument strings, e.g. "--faces 5000"')
    ap.add_argument("--keep", help="keep the outputs in this folder")
    ap.add_argument("--cpp", default=CPP, help="C++ CLI to compare (default: the Python-parity build)")
    ap.add_argument("--approx", action="store_true",
                    help="portable build: report face counts / worst deviation side by side instead of "
                         "requiring identical bits; checks Automatic results stay within their limit")
    a = ap.parse_args()
    if a.approx:
        return approx(a)
    if not os.path.exists(CPP):
        sys.exit(f"build the C++ CLI first: {CPP} not found (see cpp/README.md)")
    work = a.keep or tempfile.mkdtemp(prefix="faqem_compare_")
    failures = 0
    print(f"{'input':42s} {'mode':22s} {'faces (py = cpp)':22s} {'python':>8s} {'c++':>8s}  result")
    for inp in a.inputs:
        for mode in a.modes:
            tag = re.sub(r"[^A-Za-z0-9]+", "_", f"{os.path.basename(inp)}_{mode}")
            dp, dc = os.path.join(work, tag, "py"), os.path.join(work, tag, "cpp")
            args = [inp] + mode.split()
            out_p, tp = run([PY, os.path.join(ROOT, "cli.py")] + args, dp)
            out_c, tc = run([CPP] + args, dc)
            sp, sc = summary(out_p), summary(out_c)
            problems = [] if sp == sc else [f"report differs:\n    py  {sp}\n    cpp {sc}"]
            problems += compare_dirs(dp, dc)
            faces = sp[0].split("->")[1].replace("faces", "").strip() if sp[0] else "?"
            status = "identical" if not problems else "DIFFERENT"
            print(f"{os.path.basename(inp)[:42]:42s} {mode:22s} {faces:22s} {tp:7.1f}s {tc:7.1f}s  {status}", flush=True)
            for p in problems:
                print("    " + p)
            failures += bool(problems)
    print(f"\n{failures} of {len(a.inputs) * len(a.modes)} runs differ" if failures else "\nall runs identical")
    sys.exit(1 if failures else 0)


LIMITS = {"Low": 0.5, "Medium": 0.2, "High": 0.1, "Ultra": 0.05}


def approx(a):
    """Portable build vs Python: same face count / worst deviation? (no bit identity expected)"""
    work = a.keep or tempfile.mkdtemp(prefix="faqem_approx_")
    rows, out_of_limit, same_faces, n = [], 0, 0, 0
    print(f"{'input':42s} {'mode':16s} {'faces py':>9s} {'faces c++':>9s} {'worst py %':>11s} {'worst c++ %':>11s}  note")
    for inp in a.inputs:
        for mode in a.modes:
            tag = re.sub(r"[^A-Za-z0-9]+", "_", f"{os.path.basename(inp)}_{mode}")
            args = [inp] + mode.split()
            out_p, _ = run([PY, os.path.join(ROOT, "cli.py")] + args, os.path.join(work, tag, "py"))
            out_c, _ = run([a.cpp] + args, os.path.join(work, tag, "cpp"))
            vals = []
            for out in (out_p, out_c):
                f = re.search(r"-> ([\d,]+) faces", out).group(1).replace(",", "")
                h = float(re.search(r'"hausdorff": ([^,}]+)', out).group(1)) * 100
                vals.append((int(f), h))
            (fp, hp), (fc, hc) = vals
            lvl = mode.split()[-1] if mode.startswith("--auto") else None
            note = []
            if fp == fc:
                same_faces += 1
            else:
                note.append(f"{fc - fp:+d} faces")
            if lvl and hc > LIMITS[lvl] + 1e-9:
                note.append("C++ ABOVE LIMIT")
                out_of_limit += 1
            n += 1
            print(f"{os.path.basename(inp)[:42]:42s} {mode:16s} {fp:9d} {fc:9d} {hp:11.4f} {hc:11.4f}  "
                  f"{', '.join(note) or 'same'}", flush=True)
    print(f"\n{same_faces} of {n} runs give the same face count as Python; "
          f"{out_of_limit} Automatic results above their limit")
    sys.exit(1 if out_of_limit else 0)


if __name__ == "__main__":
    main()
