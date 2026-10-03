"""Quick check that FA-QEM works in this environment.

    python smoke_test.py                  # generated sphere + box
    python smoke_test.py path/to/mesh.glb # also a real mesh (any format trimesh reads)

The first run compiles the Numba code (about 30-60 s); later runs are fast.
Exits with code 1 if anything is wrong.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from faqem_adapter import simplify_mesh_faqem  # noqa: E402

LIMITS = {"Low": 0.5, "Medium": 0.2, "High": 0.1, "Ultra": 0.05}


def sphere(n=120, r=1.0):
    u, v = np.meshgrid(np.linspace(0, np.pi, n), np.linspace(0, 2 * np.pi, 2 * n, endpoint=False), indexing="ij")
    P = np.stack([r * np.sin(u) * np.cos(v), r * np.cos(u), r * np.sin(u) * np.sin(v)], -1).reshape(-1, 3)
    F = []
    m = 2 * n
    for i in range(n - 1):
        for j in range(m):
            a, b, c, d = i * m + j, i * m + (j + 1) % m, (i + 1) * m + j, (i + 1) * m + (j + 1) % m
            F += [(a, c, b), (b, c, d)]
    return P, np.array(F)


def box(n=40):
    import trimesh
    t = trimesh.creation.box(extents=(2.0, 1.0, 0.5))
    t = t.subdivide_to_size(max_edge=2.0 / n)
    return np.asarray(t.vertices), np.asarray(t.faces)


def check(name, V, F):
    ok = True
    for lvl, lim in LIMITS.items():
        t = time.perf_counter()
        V2, F2, info = simplify_mesh_faqem(V, F, detail=lvl)
        good = info["within_limit"] and info["deviation_percent"] <= lim + 1e-9 and 0 < len(F2) < len(F)
        good = good and np.allclose(V2.min(0), V.min(0), atol=0.02 * np.ptp(V, 0).max())   # same space
        ok &= bool(good)
        print(f"{name:10s} {lvl:6s}: {len(F):>9,} -> {len(F2):>8,} faces, deviation {info['deviation_percent']:.3f}% "
              f"(limit {lim}%) {'OK' if good else 'FAIL'}  {time.perf_counter() - t:.1f}s", flush=True)
    target = max(100, len(F) // 20)
    V2, F2, info = simplify_mesh_faqem(V, F, target_faces=target)
    good = abs(len(F2) - target) <= max(4, 0.02 * target)
    ok &= bool(good)
    print(f"{name:10s} faces={target}: {len(F2):,} faces, deviation {info['deviation_percent']:.3f}% {'OK' if good else 'FAIL'}")
    return ok


def main():
    ok = check("sphere", *sphere()) & check("box", *box())
    if len(sys.argv) > 1:
        import trimesh
        m = trimesh.load(sys.argv[1], force="mesh", process=False)
        ok &= check(os.path.basename(sys.argv[1])[:10], np.asarray(m.vertices), np.asarray(m.faces))
    print("SMOKE TEST", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
