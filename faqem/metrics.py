"""Geometric fidelity metrics of the paper's evaluation (Sec. 4.1): symmetric Hausdorff
distance and mean squared Chamfer distance, measured point-to-surface in the unit-diagonal
normalised space (i.e. relative to the bounding-box diagonal)."""
import numpy as np
from numba import njit, prange

from .bvh import BVH, bvh_closest


def sample_surface(P, F, n, seed=0, include_vertices=True):
    a, b, c = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2
    rng = np.random.default_rng(seed)
    total = area.sum()
    if not total > 0:
        return P.copy()
    f = rng.choice(len(F), size=n, p=area / total)
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    pts = a[f] + u[:, None] * (b[f] - a[f]) + v[:, None] * (c[f] - a[f])
    if include_vertices:
        step = max(1, len(P) // n)
        pts = np.concatenate([pts, P[::step]])
    return pts


@njit(cache=True, parallel=True)
def _dists(pts, P, F, idx, box, left, start, count):
    n = pts.shape[0]
    out_d2 = np.empty(n)
    reps = np.zeros((1, 3), np.int64)
    fn = np.zeros((1, 3))
    for i in prange(n):
        stack = np.empty(256, np.int64)
        sd = np.empty(256)
        o = np.empty(6)
        t = np.empty(6)
        _, d2 = bvh_closest(P, F, idx, box, left, start, count, pts[i, 0], pts[i, 1], pts[i, 2], -1,
                            0, reps, -1, -1, -1, fn, 0.0, 0.0, 0.0, stack, sd, o, t)
        out_d2[i] = d2
    return out_d2


def point_to_mesh_d2(pts, bvh):
    return _dists(np.ascontiguousarray(pts), *bvh.arrays)


def compare_meshes(PA, FA, PB, FB, samples=50000, bvh_a=None):
    """Hausdorff / Chamfer between two meshes in the same (normalised) space."""
    A = bvh_a or BVH(PA, FA)
    B = BVH(PB, FB)
    sa = sample_surface(PA, FA, samples, 7)
    sb = sample_surface(PB, FB, samples, 11)
    dab = point_to_mesh_d2(sa, B)
    dba = point_to_mesh_d2(sb, A)
    return dict(
        hausdorff=float(np.sqrt(max(dab.max(), dba.max()))),
        chamfer=float(dab.mean() + dba.mean()),
        mean_distance=float((np.sqrt(dab).mean() + np.sqrt(dba).mean()) / 2),
    )
