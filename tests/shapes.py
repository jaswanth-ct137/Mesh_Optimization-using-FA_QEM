"""Procedural test meshes covering the cases the paper targets."""
import numpy as np


def param_surface(nu, nv, fn, wrap_u=False, wrap_v=False):
    cu = nu if wrap_u else nu + 1
    cv = nv if wrap_v else nv + 1
    u, v = np.meshgrid(np.arange(cu) / nu, np.arange(cv) / nv)
    P = np.stack(fn(u.ravel(), v.ravel()), axis=1)
    i, j = np.meshgrid(np.arange(nu), np.arange(nv))
    i, j = i.ravel(), j.ravel()
    i1 = (i + 1) % nu if wrap_u else i + 1
    j1 = (j + 1) % nv if wrap_v else j + 1
    a, b, c, d = j * cu + i, j * cu + i1, j1 * cu + i1, j1 * cu + i
    F = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    return P, F


def merge(parts):
    Ps, Fs, off = [], [], 0
    for P, F in parts:
        Ps.append(P); Fs.append(F + off); off += len(P)
    return np.concatenate(Ps), np.concatenate(Fs)


def sphere(n=200, r=1.0, center=(0, 0, 0), bump=None):
    def fn(u, v):
        th, ph = u * 2 * np.pi, v * np.pi
        rr = r + (bump(th, ph) if bump else 0)
        return (center[0] + rr * np.sin(ph) * np.cos(th), center[1] + rr * np.cos(ph),
                center[2] + rr * np.sin(ph) * np.sin(th))
    return param_surface(n, n // 2, fn, wrap_u=True)


def cube(n=60):
    parts = []
    for axis in range(3):
        for s in (-1, 1):
            def fn(u, v, axis=axis, s=s):
                a, b = 2 * u - 1, (2 * v - 1) * s
                c = np.full_like(a, s)
                coords = [None, None, None]
                coords[axis] = c
                coords[(axis + 1) % 3] = a
                coords[(axis + 2) % 3] = b
                return tuple(coords)
            parts.append(param_surface(n, n, fn))
    return merge(parts)


def bowl(n=200):
    def fn(u, v):
        th, ph = u * 2 * np.pi, v * np.pi * 0.55
        r = 1 + 0.08 * np.sin(6 * th) * v
        return r * np.sin(ph) * np.cos(th), -r * np.cos(ph), r * np.sin(ph) * np.sin(th)
    return param_surface(n, n // 4, fn, wrap_u=True)


def star_disk(n=300):
    def fn(u, v):
        th = u * 2 * np.pi
        R = (0.6 + 0.4 * np.abs(np.cos(2.5 * th))) * np.maximum(v, 1e-3)
        return R * np.cos(th), np.zeros_like(R), R * np.sin(th)
    return param_surface(n, n // 6, fn, wrap_u=True)


def detail_torus(n=400):
    def fn(u, v):
        a, b = u * 2 * np.pi, v * 2 * np.pi
        r = 0.35 + 0.02 * (np.abs(np.sin(12 * a)) > 0.5) * np.sign(np.sin(12 * a)) + 0.01 * np.sin(40 * b)
        return (1 + r * np.cos(b)) * np.cos(a), r * np.sin(b), (1 + r * np.cos(b)) * np.sin(a)
    return param_surface(n, n // 2, fn, wrap_u=True, wrap_v=True)


def noisy_sphere(n=240, amp=0.004, seed=0):
    """Scan-like noise on a closed sphere (noise applied after welding the poles)."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from faqem.prep import prepare_mesh
    m = prepare_mesh(*sphere(n))
    return m.positions + np.random.default_rng(seed).uniform(-amp, amp, m.positions.shape), m.faces


def components():
    c = cube(20)
    c = (c[0] * 0.2 + np.array([-0.9, 0, 0]), c[1])
    return merge([sphere(120, 0.5), sphere(80, 0.3, (0.805, 0, 0)), sphere(60, 0.2, (-0.2, 0.705, 0)), c])


def non_manifold_fins(n=60):
    parts = []
    for k in range(3):
        ang = k * 2 * np.pi / 3
        def fn(u, v, ang=ang):
            r = v * (1 + 0.3 * np.sin(np.pi * u))
            return r * np.cos(ang), 2 * u - 1, r * np.sin(ang)
        parts.append(param_surface(n, n, fn))
    return merge(parts)


ALL = dict(sphere=lambda: sphere(300), cube=cube, bowl=bowl, star_disk=star_disk,
           detail_torus=detail_torus, noisy_sphere=noisy_sphere, components=components,
           non_manifold_fins=non_manifold_fins)


# ---- hard-surface shapes ----------------------------------------------------------------
def noisy_cube(n=40, amp=1e-4, seed=0):
    """Cube whose vertices are jittered along the face normals (after welding)."""
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from faqem.prep import prepare_mesh
    m = prepare_mesh(*cube(n))
    P = m.positions.copy()
    rng = np.random.default_rng(seed)
    ax = np.argmax(np.abs(P), axis=1)               # dominant axis ~ face normal
    on_edge = (np.abs(P) > np.abs(P).max() * 0.999).sum(1) > 1
    noise = rng.uniform(-amp, amp, len(P)) * ~on_edge
    P[np.arange(len(P)), ax] += noise * np.sign(P[np.arange(len(P)), ax])
    return P, m.faces


def voxel_surface(mask, scale=1.0):
    """Closed quad surface of a boolean voxel grid (each exposed voxel face = 2 triangles)."""
    mask = np.pad(mask.astype(bool), 1)
    verts, faces, index = [], [], {}
    def vid(p):
        if p not in index:
            index[p] = len(verts); verts.append(p)
        return index[p]
    X, Y, Z = mask.shape
    dirs = [((1, 0, 0), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]), ((-1, 0, 0), [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),
            ((0, 1, 0), [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]), ((0, -1, 0), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),
            ((0, 0, 1), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]), ((0, 0, -1), [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)])]
    for x, y, z in zip(*np.nonzero(mask)):
        for (dx, dy, dz), quad in dirs:
            if mask[x + dx, y + dy, z + dz]:
                continue
            q = [vid((x + a, y + b, z + c)) for a, b, c in quad]
            faces += [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
    return np.array(verts, float) * scale, np.array(faces, np.int64)


def l_block(n=12):
    """L-shaped prism: minimal = 4 + 4 (L caps) + 6 x 2 (walls) = 20 triangles."""
    mask = np.zeros((2 * n, 2 * n, n), bool)
    mask[:n, :, :] = True
    mask[:, :n, :] = True
    return voxel_surface(mask, 1.0 / n)


def box_with_knob(n=30):
    """Subdivided box plus a round knob (separate, touching component)."""
    P, F = cube(n)
    kP, kF = sphere(60, 0.25, (0, 0, 1.2))
    return merge([(P, F), (kP, kF)])


ALL.update(noisy_cube=noisy_cube, l_block=l_block, box_with_knob=box_with_knob)


def rounded_box(n=40, r=0.01):
    """Subdivided cube with rounded edges/corners of radius r (cube half-size 1)."""
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from faqem.prep import prepare_mesh
    m = prepare_mesh(*cube(n))
    P = m.positions / np.abs(m.positions).max()          # back to [-1, 1]^3
    c = np.clip(P, -(1 - r), 1 - r)
    d = P - c
    l = np.linalg.norm(d, axis=1, keepdims=True)
    P = c + r * d / np.maximum(l, 1e-12)
    return P, m.faces


ALL.update(rounded_box=rounded_box)
