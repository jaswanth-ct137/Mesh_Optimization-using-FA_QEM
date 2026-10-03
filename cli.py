"""Command-line FA-QEM simplification.

    python cli.py input.glb --ratio 0.1 -o out_dir
    python cli.py scan.stl --faces 5000 --no-normal-map
    python cli.py doll.glb --auto Medium           # face count chosen for a max deviation
"""
import argparse
import json
import os
import time

from faqem.core import DEFAULTS
from faqem.engine import LoadedMesh, run


def main():
    ap = argparse.ArgumentParser(description="FA-QEM feature-aware mesh simplification")
    ap.add_argument("input")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--ratio", type=float, default=0.1, help="fraction of faces to keep (default 0.1)")
    g.add_argument("--faces", type=int, help="target face count")
    g.add_argument("--auto", choices=["Low", "Medium", "High", "Ultra"],
                   help="automatic face count for a detail level (guaranteed max deviation)")
    g.add_argument("--auto-deviation", type=float, help="automatic mode with an exact max deviation (%% of size)")
    ap.add_argument("--fast", action="store_true",
                    help="--ratio / --faces: one plain FA-QEM pass instead of the best result for the count")
    ap.add_argument("--engine", choices=["cpp-mt", "cpp", "cpp-mac", "python"], default=None,
                    help="cpp-mt: C++ on all cores (default when built), cpp: C++ portable (one core for "
                         "the collapse checks), cpp-mac: C++ bit-identical to Python (macOS), python")
    ap.add_argument("-o", "--out", default="output")
    ap.add_argument("--no-color", action="store_true", help="skip colour / texture transfer")
    ap.add_argument("--no-normal-map", action="store_true", help="skip normal-map baking")
    ap.add_argument("--atlas", type=int, default=None, help="texture atlas size (default: automatic)")
    ap.add_argument("--no-metrics", action="store_true")
    INTERNAL = {"auto", "max_error", "local_tol"}  # set by --auto / --auto-deviation
    for k, v in DEFAULTS.items():
        if k in INTERNAL:
            continue
        if isinstance(v, bool):
            ap.add_argument(f"--{k.replace('_', '-')}", type=lambda s: s.lower() in ("1", "true", "yes"), default=None,
                            help=f"(default {v})")
        else:
            ap.add_argument(f"--{k.replace('_', '-')}", type=float, default=None, help=f"(default {v})")
    a = ap.parse_args()

    t = time.perf_counter()
    mesh = LoadedMesh(a.input, engine=a.engine)
    print(f"loaded {a.input}: {mesh.stats} ({time.perf_counter() - t:.1f}s)")
    from faqem.engine import describe
    print(f"engine: {describe(mesh)}")
    target = a.faces or int(round(mesh.stats["faces"] * a.ratio))
    from faqem.pipeline import DETAIL_LEVELS
    auto_tol = a.auto_deviation / 100.0 if a.auto_deviation else (DETAIL_LEVELS[a.auto] if a.auto else None)
    opts = {k: getattr(a, k) for k in DEFAULTS if k not in INTERNAL and getattr(a, k, None) is not None}
    res = run(mesh, None if auto_tol else target, a.out, options=opts, auto_tolerance=auto_tol,
              fast=a.fast, bake_color=not a.no_color, bake_normal=not a.no_normal_map,
              atlas_size=a.atlas, compute_metrics=not a.no_metrics,
              progress=lambda f, msg: print(f"  [{f * 100:5.1f}%] {msg}", flush=True))
    st = res["stats"]
    print(f"\n{st['input_faces']:,} -> {st['output_faces']:,} faces in {st['time_total']:.2f}s "
          f"(bake {res['time_bake']:.2f}s)")
    if st.get("best_for_count"):
        print(f"best for this face count: worst deviation {res['metrics']['hausdorff'] * 100:.3f}% "
              f"(plain FA-QEM {st['plain_hausdorff'] * 100:.3f}%), {len(st['count_passes'])} passes")
    if "metrics" in res:
        print("fidelity:", json.dumps({k: round(v, 7) for k, v in res["metrics"].items()}))
    for k in ("textured_glb", "geometry_glb", "geometry_obj"):
        if res.get(k):
            print(f"  {k:13s} {os.path.abspath(res[k])}")


if __name__ == "__main__":
    main()
