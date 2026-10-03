"""Cross-platform identity check for the portable C++ build.

    python3 tools/portable_golden.py --write            # on the reference machine: record digests
    python3 tools/portable_golden.py                    # anywhere else: must reproduce them exactly
    python3 tools/portable_golden.py --faqem path/to/faqem(.exe) --quick

Runs `faqem --digest` on a fixed corpus (the 7 SKUs x 4 detail levels, face-count and --fast runs,
the Paper Table 1 preset, OBJ inputs) and compares the SHA-256 of every result array and output
file with cpp/tests/golden_portable.json. Standard library only (no numpy needed). Exit code 1 on
any difference.
"""
import argparse
import glob
import json
import os
import platform
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(ROOT, "cpp", "tests", "golden_portable.json")


def corpus(quick):
    skus = sorted(glob.glob(os.path.join(ROOT, "checking copy", "*.glb")))
    small = os.path.join(ROOT, "tripo3-wooden+dresser+3d+model.glb.glb")
    boots = os.path.join(ROOT, "checking copy", "buwch_Boots (1).glb")
    cases = []
    levels = ["Low", "High"] if quick else ["Low", "Medium", "High", "Ultra"]
    for s in ([skus[0], boots] if quick else skus):
        for lvl in levels:
            cases.append((s, ["--auto", lvl]))
    cases += [(boots, ["--faces", "3000"]), (boots, ["--faces", "3000", "--fast"]),
              (small, ["--auto", "Medium"]), (small, ["--faces", "800", "--w-plane-area", "1", "--virtual-edges", "true"])]
    for obj in sorted(glob.glob(os.path.join(ROOT, "samples", "*", "*.obj")))[: (1 if quick else 4)]:
        cases.append((obj, ["--faces", "500", "--fast"]))
    return cases


def key(path, args):
    return os.path.relpath(path, ROOT).replace("\\", "/") + " " + " ".join(args)


THREADS = []


def run(faqem, path, args, out):
    p = subprocess.run([faqem, path, *args, *THREADS, "--digest", "-o", out], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"faqem failed on {key(path, args)}:\n{p.stdout}\n{p.stderr}")
    d = {}
    for line in p.stdout.splitlines():
        if line.startswith("digest "):
            parts = line.split()
            d[" ".join(parts[1:-1])] = parts[-1]
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    exe = "faqem.exe" if os.name == "nt" else "faqem"
    ap.add_argument("--faqem", default=os.path.join(ROOT, "cpp", "build", exe))
    ap.add_argument("--write", action="store_true", help="record the digests as the new golden file")
    ap.add_argument("--quick", action="store_true", help="a smaller corpus (about 10 runs)")
    ap.add_argument("--threads", type=int, help="pass --threads N to faqem (results must not depend on it)")
    a = ap.parse_args()
    if a.threads is not None:
        THREADS[:] = ["--threads", str(a.threads)]
    cases = corpus(a.quick)
    golden = {}
    if not a.write:
        with open(GOLDEN) as f:
            golden = json.load(f)["cases"]
    results, failures = {}, 0
    with tempfile.TemporaryDirectory() as tmp:
        for i, (path, args) in enumerate(cases):
            k = key(path, args)
            d = run(a.faqem, path, args, os.path.join(tmp, str(i)))
            if d.get("mode") != "portable":
                sys.exit(f"{a.faqem} is not a portable build (mode {d.get('mode')})")
            results[k] = d
            if a.write:
                print(f"recorded  {k}", flush=True)
                continue
            g = golden.get(k)
            if g is None:
                print(f"MISSING   {k} (not in the golden file)")
                failures += 1
                continue
            diff = sorted(n for n in set(g) | set(d) if g.get(n) != d.get(n))
            print(f"{'identical' if not diff else 'DIFFERENT'} {k}" + ("" if not diff else f"  -> {diff}"), flush=True)
            failures += bool(diff)
    if a.write:
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        with open(GOLDEN, "w") as f:
            json.dump({"recorded_on": f"{platform.system()} {platform.machine()}", "cases": results}, f, indent=1,
                      sort_keys=True)
        print(f"\nwrote {len(results)} cases to {GOLDEN}")
        return
    print(f"\n{'all ' + str(len(cases)) + ' runs identical' if not failures else str(failures) + ' runs differ'} "
          f"({platform.system()} {platform.machine()})")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
