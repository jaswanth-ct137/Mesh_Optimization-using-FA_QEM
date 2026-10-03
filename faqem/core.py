"""FA-QEM geometric simplification (Bhosikar et al., arXiv:2605.14029, Algorithm 1).

    Q_gf^k   = Q_base^k + Q_boundary^k + Q_normal^k                         (Eq. 4)
    Q_base   = sum_p K_p / (A_p)^w_plane_area   (inverse-area weighting)     (Eq. 5)
    Q_bound  = w_boundary * kappa * (p1 p1^T + p2 p2^T)                      (Eq. 6-9)
    Q_normal = w_normal * p p^T     (tangent plane of the original normal)   (Eq. 10)
    cost     = v'^T (Q_i + Q_j) v'  +  w_area * v'^T Q_A v'                  (Eq. 2, 3, 11)

Robustness (Sec. 3.3, supplementary 6.2): generalised non-manifold adjacency, normal
flip veto, virtual edges between nearby components, tiny-edge guard, singular-system
fallback to {v_i, v_j, (v_i+v_j)/2}.

The mesh is processed in a unit-diagonal normalised space (prep.py). The two scale
dependent terms are made dimensionless so that Table 1 weights mean the same thing on
every model: face areas are taken relative to the mean face area, and the boundary
area quadric is divided by the squared mean edge length.

All hot loops are Numba-compiled (the paper also JIT-compiles its numerical hotspots).
"""
import time

import numpy as np
from numba import njit

from .bvh import BVH, bvh_closest, closest_on_tri
from .prep import unique_edges
from .virtual_edges import find_virtual_edges

# Exact hyper-parameters of the paper (Table 1 / supplementary Table 5).
PAPER_TABLE1 = dict(w_area=100.0, w_boundary=500.0, w_normal=0.01, w_plane_area=1.0, virtual_edges=True)

DEFAULTS = dict(
    # Table 1 weights ...
    w_area=100.0,
    w_boundary=500.0,
    w_normal=0.01,
    # ... except the two terms that measurably hurt on real AI-generated / scanned meshes in
    # our ablation (see README): inverse-area weighting (Eq. 5) and virtual edges.
    w_plane_area=0.0,
    virtual_edges=False,
    virtual_tau=0.01,        # proximity threshold, fraction of the bbox diagonal (Sec. 3.3)
    min_edge_rel=1e-8,       # shorter edges get infinite cost (supplementary 6.2)
    flip_threshold=0.0,      # reject collapse if cos(n_old, n_new) < threshold (paper: dot < 0)
    kappa_dimensionless=True, # Eq. 6 kappa x half-chord (scale free); False = raw kappa (bbox units)
    boundary_window=1,       # chain distance of v2, v3 in Eq. 6 (1 = immediate neighbours)
    # implementation safeguards
    preserve_topology=True,  # link condition / edge-valence guard on real edges
    topology_penalty=10.0,   # a topology-changing collapse is re-queued at cost * penalty
                             # (inf = never change topology, 1 = ignore topology)
    max_move_factor=2.0,     # ignore optimal positions farther than this * |e| from the midpoint
    plane_area_clamp=100.0,  # clamp of the Eq. 5 weights (relative to the mean face area)
    max_rebuilds=8,
    # ---- automatic mode (pipeline.simplify_auto) -------------------------------------------
    max_error=0.0,           # reject collapses with mean deviation above this (x diag); 0 = off
    auto=False,              # no face target: simplify until the error limits stop it
    local_tol=0.0,           # every original point stays within this of the result (x diag);
                             # checked at each collapse (local worst-case guarantee); 0 = off
)

PRESETS = {
    "Recommended": {},
    "Paper Table 1 (exact)": PAPER_TABLE1,
    "Plain QEM (ablation baseline)": dict(w_area=0.0, w_boundary=0.0, w_normal=0.0, w_plane_area=0.0,
                                          virtual_edges=False),
}

QN = 10  # symmetric 4x4 stored as [q00 q01 q02 q03 q11 q12 q13 q22 q23 q33]

# counters layout
C_FACES, C_COLL, C_FLIP, C_LINK, C_DUP, C_STALE, C_REBUILD, C_VIRT, C_SINCE, C_DONE, C_ERR = range(11)
N_COUNTERS = 16


# =====================================================================================
# quadric helpers
# =====================================================================================
def _plane_quadrics(n, d, w):
    """Vectorised w * p p^T for planes p = [n, d]; returns (k, 10)."""
    a, b, c = n[:, 0], n[:, 1], n[:, 2]
    return w[:, None] * np.stack(
        [a * a, a * b, a * c, a * d, b * b, b * c, b * d, c * c, c * d, d * d], axis=1)


def _scatter_add(Q, idx, vals):
    for k in range(QN):
        Q[:, k] += np.bincount(idx, weights=vals[:, k], minlength=len(Q))


def build_quadrics(P, F, edges, edge_counts, o):
    """Composite geometric-feature quadric Q_gf per vertex (Eq. 4) + scale info.
    Also returns Qgeo / Ageo (plain area-weighted face planes) for the error cap."""
    nV = len(P)
    Q = np.zeros((nV, QN))
    a, b, c = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    cr = np.cross(b - a, c - a)
    ln = np.linalg.norm(cr, axis=1)
    area = ln / 2
    ok = ln > 1e-300
    nrm = np.zeros_like(cr)
    nrm[ok] = cr[ok] / ln[ok, None]
    mean_area = float(area[ok].mean()) if ok.any() else 1.0

    # --- Q_base (Eq. 5): inverse-area weighted face-plane quadrics ------------------
    if o["w_plane_area"] != 0:
        rel = np.maximum(area / mean_area, 1e-6)
        cl = o["plane_area_clamp"]
        w = np.clip(rel ** (-o["w_plane_area"]), 1.0 / cl, cl)
    else:
        w = np.ones(len(F))
    w = np.where(ok, w, 0.0)
    d = -np.einsum("ij,ij->i", nrm, a)
    # plain area-weighted quadric of the original surface (error cap only)
    Kg = _plane_quadrics(nrm, d, np.where(ok, area, 0.0))
    Qgeo = np.zeros((nV, QN))
    Ageo = np.zeros(nV)
    for k in range(3):
        _scatter_add(Qgeo, F[:, k], Kg)
        Ageo += np.bincount(F[:, k], weights=area, minlength=nV)
    Kp = _plane_quadrics(nrm, d, w)
    for k in range(3):
        _scatter_add(Q, F[:, k], Kp)

    # --- Q_normal (Eq. 10): tangent plane of the angle-weighted vertex normal -------
    if o["w_normal"] > 0:
        vn = np.zeros((nV, 3))
        for k in range(3):
            p0, p1, p2 = P[F[:, k]], P[F[:, (k + 1) % 3]], P[F[:, (k + 2) % 3]]
            u, v = p1 - p0, p2 - p0
            ang = np.arctan2(np.linalg.norm(np.cross(u, v), axis=1), np.einsum("ij,ij->i", u, v))
            for j in range(3):
                vn[:, j] += np.bincount(F[:, k], weights=ang * nrm[:, j], minlength=nV)
        l = np.linalg.norm(vn, axis=1)
        good = l > 1e-12
        un = np.zeros_like(vn)
        un[good] = vn[good] / l[good, None]
        dn = -np.einsum("ij,ij->i", un, P)
        Q += _plane_quadrics(un, dn, np.where(good, o["w_normal"], 0.0))

    # --- Q_boundary (Eq. 6-9): curvature-scaled dual-plane splint --------------------
    elen = np.linalg.norm(P[edges[:, 0]] - P[edges[:, 1]], axis=1)
    mean_edge = float(elen.mean()) if len(elen) else 1.0
    bedges = edges[edge_counts == 1]
    n_boundary_vertices = 0
    if len(bedges):
        bv = bedges.reshape(-1)
        n_boundary_vertices = int(len(np.unique(bv)))
    if o["w_boundary"] > 0 and len(bedges):
        # boundary neighbours per vertex; only simple chains (exactly 2 neighbours)
        src = np.concatenate([bedges[:, 0], bedges[:, 1]])
        dst = np.concatenate([bedges[:, 1], bedges[:, 0]])
        order = np.argsort(src, kind="stable")
        src, dst = src[order], dst[order]
        verts, start, count = np.unique(src, return_index=True, return_counts=True)
        sel = count == 2
        v1 = verts[sel]
        n1_, n2_ = dst[start[sel]], dst[start[sel] + 1]
        # neighbours along the boundary chain, `boundary_window` steps away on each side
        # (window 1 == the immediate neighbours of Eq. 6; wider windows average out zig-zag
        # noise of reconstructed / generated rims)
        nb1 = np.full(nV, -1, np.int64); nb2 = np.full(nV, -1, np.int64)
        nb1[v1] = n1_; nb2[v1] = n2_
        def walk(prev, cur, steps):
            for _ in range(steps - 1):
                ok = (nb1[cur] >= 0)
                nxt = np.where(nb1[cur] == prev, nb2[cur], nb1[cur])
                nxt = np.where(ok & (nxt != v1), nxt, cur)
                prev, cur = np.where(nxt != cur, cur, prev), nxt
            return cur
        win = int(o["boundary_window"])
        v2 = walk(v1, n1_, win)
        v3 = walk(v1, n2_, win)
        x1, x2, x3 = P[v1], P[v2], P[v3]
        d1 = x3 - x2                      # delta'
        d2 = x3 - 2 * x1 + x2             # delta''
        dl = np.linalg.norm(d1, axis=1)
        kappa = np.linalg.norm(np.cross(d1, d2), axis=1) / np.maximum(dl, 1e-12) ** 3
        if o["kappa_dimensionless"]:
            kappa = kappa * dl / 2        # curvature x half-chord ~ turning angle: scale free
        else:
            kappa = np.minimum(kappa, 2.0 / mean_edge)
        kappa = np.where(dl > 1e-12, kappa, 0.0)
        wb = o["w_boundary"] * kappa
        n1 = np.cross(x1 - x2, x3 - x1)   # Eq. 7
        l1 = np.linalg.norm(n1, axis=1)
        g1 = l1 > 1e-14
        n1u = np.zeros_like(n1)
        n1u[g1] = n1[g1] / l1[g1, None]
        dd = x1 - x2                      # Eq. 8
        l2 = np.linalg.norm(dd, axis=1)
        g2 = l2 > 1e-14
        du = np.zeros_like(dd)
        du[g2] = dd[g2] / l2[g2, None]
        Qb = _plane_quadrics(n1u, -np.einsum("ij,ij->i", n1u, x1), np.where(g1, wb, 0.0))
        Qb += _plane_quadrics(du, -np.einsum("ij,ij->i", du, x1), np.where(g2, wb, 0.0))
        _scatter_add(Q, v1, Qb)       # Eq. 9

    return Q, mean_area, mean_edge, n_boundary_vertices, Qgeo, Ageo


# =====================================================================================
# Numba kernels
# =====================================================================================
@njit(cache=True, inline="always")
def _eval_q(q, x, y, z):
    return (q[0] * x * x + 2 * q[1] * x * y + 2 * q[2] * x * z + 2 * q[3] * x
            + q[4] * y * y + 2 * q[5] * y * z + 2 * q[6] * y
            + q[7] * z * z + 2 * q[8] * z + q[9])


@njit(cache=True)
def _jacobi3(A, V):
    for i in range(9):
        V[i] = 0.0
    V[0] = 1.0; V[4] = 1.0; V[8] = 1.0
    for _sweep in range(12):
        off = A[1] * A[1] + A[2] * A[2] + A[5] * A[5]
        dg = A[0] * A[0] + A[4] * A[4] + A[8] * A[8]
        if off == 0.0 or off <= 1e-24 * dg:
            return
        for p in range(2):
            for q in range(p + 1, 3):
                apq = A[p * 3 + q]
                if apq == 0.0:
                    continue
                theta = (A[q * 3 + q] - A[p * 3 + p]) / (2 * apq)
                sg = 1.0 if theta >= 0 else -1.0
                t = sg / (abs(theta) + np.sqrt(theta * theta + 1))
                c = 1 / np.sqrt(t * t + 1)
                s = t * c
                for k in range(3):
                    akp = A[k * 3 + p]; akq = A[k * 3 + q]
                    A[k * 3 + p] = c * akp - s * akq
                    A[k * 3 + q] = s * akp + c * akq
                for k in range(3):
                    apk = A[p * 3 + k]; aqk = A[q * 3 + k]
                    A[p * 3 + k] = c * apk - s * aqk
                    A[q * 3 + k] = s * apk + c * aqk
                for k in range(3):
                    vkp = V[k * 3 + p]; vkq = V[k * 3 + q]
                    V[k * 3 + p] = c * vkp - s * vkq
                    V[k * 3 + q] = s * vkp + c * vkq


@njit(cache=True)
def _solve_q(q, mx, my, mz, out, rel_eps, Am, Vm):
    """argmin v^T Q v around the midpoint m; rank-truncated pseudo-inverse for singular Q."""
    a00 = q[0]; a01 = q[1]; a02 = q[2]; a11 = q[4]; a12 = q[5]; a22 = q[7]
    tr = a00 + a11 + a22
    if not (tr > 1e-300):
        return False
    rx = a00 * mx + a01 * my + a02 * mz + q[3]
    ry = a01 * mx + a11 * my + a12 * mz + q[6]
    rz = a02 * mx + a12 * my + a22 * mz + q[8]
    c00 = a11 * a22 - a12 * a12
    c01 = a02 * a12 - a01 * a22
    c02 = a01 * a12 - a02 * a11
    det = a00 * c00 + a01 * c01 + a02 * c02
    if det > rel_eps * 4 * tr * tr * tr:  # well conditioned: Cramer
        c11 = a00 * a22 - a02 * a02
        c12 = a01 * a02 - a00 * a12
        c22 = a00 * a11 - a01 * a01
        inv = 1.0 / det
        out[0] = mx - (c00 * rx + c01 * ry + c02 * rz) * inv
        out[1] = my - (c01 * rx + c11 * ry + c12 * rz) * inv
        out[2] = mz - (c02 * rx + c12 * ry + c22 * rz) * inv
        return True
    Am[0] = a00; Am[1] = a01; Am[2] = a02
    Am[3] = a01; Am[4] = a11; Am[5] = a12
    Am[6] = a02; Am[7] = a12; Am[8] = a22
    _jacobi3(Am, Vm)
    lmax = max(Am[0], Am[4], Am[8])
    if not (lmax > 0):
        return False
    thr = rel_eps * lmax
    dx = 0.0; dy = 0.0; dz = 0.0
    for k in range(3):
        lk = Am[k * 4]
        if lk <= thr:
            continue
        ux = Vm[k]; uy = Vm[3 + k]; uz = Vm[6 + k]
        s = (ux * rx + uy * ry + uz * rz) / lk
        dx -= s * ux; dy -= s * uy; dz -= s * uz
    out[0] = mx + dx; out[1] = my + dy; out[2] = mz + dz
    return True


# ---- growable per-vertex lists in one pool ------------------------------------------
@njit(cache=True)
def _ml_push(pool, st, ln, cp, used, v, val):
    if ln[v] == cp[v]:
        newcap = max(4, cp[v] * 2)
        if used[0] + newcap > pool.shape[0]:
            np2 = np.empty(max(pool.shape[0] * 2, used[0] + newcap), np.int64)
            np2[:used[0]] = pool[:used[0]]
            pool = np2
        s = used[0]
        for i in range(ln[v]):
            pool[s + i] = pool[st[v] + i]
        st[v] = s
        cp[v] = newcap
        used[0] += newcap
    pool[st[v] + ln[v]] = val
    ln[v] += 1
    return pool


@njit(cache=True)
def _ml_remove(pool, st, ln, v, val):
    s = st[v]
    for i in range(ln[v]):
        if pool[s + i] == val:
            pool[s + i] = pool[s + ln[v] - 1]
            ln[v] -= 1
            return True
    return False


@njit(cache=True)
def _ml_has(pool, st, ln, v, val):
    s = st[v]
    for i in range(ln[v]):
        if pool[s + i] == val:
            return True
    return False


# ---- heap: rows [cost, a, b, stamp_a, stamp_b, x, y, z] ------------------------------
@njit(cache=True)
def _hpush(H, hs, cost, a, b, sa, sb, x, y, z):
    n = hs[0]
    if n == H.shape[0]:
        H2 = np.empty((H.shape[0] * 2, 8))
        H2[:n] = H[:n]
        H = H2
    i = n
    hs[0] = n + 1
    while i > 0:
        p = (i - 1) >> 1
        if H[p, 0] <= cost:
            break
        for k in range(8):
            H[i, k] = H[p, k]
        i = p
    H[i, 0] = cost; H[i, 1] = a; H[i, 2] = b; H[i, 3] = sa; H[i, 4] = sb
    H[i, 5] = x; H[i, 6] = y; H[i, 7] = z
    return H


@njit(cache=True)
def _hpop(H, hs, out, tmp):
    for k in range(8):
        out[k] = H[0, k]
    n = hs[0] - 1
    hs[0] = n
    if n == 0:
        return
    for k in range(8):
        tmp[k] = H[n, k]
    cost = tmp[0]
    i = 0
    while True:
        l = 2 * i + 1
        if l >= n:
            break
        r = l + 1
        if r < n and H[r, 0] < H[l, 0]:
            l = r
        if H[l, 0] >= cost:
            break
        for k in range(8):
            H[i, k] = H[l, k]
        i = l
    for k in range(8):
        H[i, k] = tmp[k]


# ---- topology helpers ---------------------------------------------------------------
@njit(cache=True)
def _ring(v, F, pool, st, ln, mark, cnt, tag, out):
    """One-ring of v through its faces; multiplicity in cnt (1 => boundary edge)."""
    tag[0] += 1
    t = tag[0]
    n = 0
    s = st[v]
    for i in range(ln[v]):
        f = pool[s + i]
        for k in range(3):
            u = F[f, k]
            if u == v:
                continue
            if mark[u] != t:
                mark[u] = t
                cnt[u] = 0
                out[n] = u
                n += 1
            cnt[u] += 1
    return n


@njit(cache=True, inline="always")
def _face_has(F, f, v):
    return F[f, 0] == v or F[f, 1] == v or F[f, 2] == v


@njit(cache=True)
def _add_area_edge(QA, P, r, s, h):
    """Eq. 11 (Lindstrom-Turk): 1/2 |E v + t|^2 with E=[e]x, e=v_s-v_r, t=v_r x v_s."""
    rx = P[r, 0]; ry = P[r, 1]; rz = P[r, 2]
    sx = P[s, 0]; sy = P[s, 1]; sz = P[s, 2]
    ex = sx - rx; ey = sy - ry; ez = sz - rz
    tx = ry * sz - rz * sy; ty = rz * sx - rx * sz; tz = rx * sy - ry * sx
    ee = ex * ex + ey * ey + ez * ez
    QA[0] += h * (ee - ex * ex); QA[1] += h * (-ex * ey); QA[2] += h * (-ex * ez)
    QA[4] += h * (ee - ey * ey); QA[5] += h * (-ey * ez)
    QA[7] += h * (ee - ez * ez)
    QA[3] += h * (ty * ez - tz * ey)      # E^T t = t x e
    QA[6] += h * (tz * ex - tx * ez)
    QA[8] += h * (tx * ey - ty * ex)
    QA[9] += h * (tx * tx + ty * ty + tz * tz)


@njit(cache=True, inline="always")
def _seg_d2(px, py, pz, P, i, j, x2, y2, z2, use_xyz):
    """Squared distance from p to segment (P[i], P[j]) or (P[i], (x2,y2,z2)) when use_xyz."""
    ax = P[i, 0]; ay = P[i, 1]; az = P[i, 2]
    if use_xyz:
        bx = x2; by = y2; bz = z2
    else:
        bx = P[j, 0]; by = P[j, 1]; bz = P[j, 2]
    ex = bx - ax; ey = by - ay; ez = bz - az
    ll = ex * ex + ey * ey + ez * ez
    qx = px - ax; qy = py - ay; qz = pz - az
    t = (qx * ex + qy * ey + qz * ez) / ll if ll > 0 else 0.0
    t = min(1.0, max(0.0, t))
    dx = qx - t * ex; dy = qy - t * ey; dz = qz - t * ez
    return dx * dx + dy * dy + dz * dz


@njit(cache=True)
def _boundary_dev2(a, b, x, y, z, P, F, pool, st, ln, mark, cnt, tag, rbA, rbB):
    """Two-sided squared distance between the open-boundary polyline around edge (a, b) before
    and after collapsing it to (x, y, z); -1 if no boundary is involved, inf if the collapse
    would delete the last faces around (a piece disappears)."""
    na = _ring(a, F, pool, st, ln, mark, cnt, tag, rbA)
    ab_bnd = False
    nA = 0
    for i in range(na):
        u = rbA[i]
        if cnt[u] == 1:
            if u == b:
                ab_bnd = True
            else:
                rbA[nA] = u  # compact boundary neighbours of a (rbA is ours for now)
                nA += 1
    # survivors: faces of a or b that do not contain both
    surv = 0
    for i in range(ln[a]):
        if not _face_has(F, pool[st[a] + i], b):
            surv += 1
    for i in range(ln[b]):
        if not _face_has(F, pool[st[b] + i], a):
            surv += 1
    if surv == 0:
        return np.inf
    nb = _ring(b, F, pool, st, ln, mark, cnt, tag, rbB)
    nB = 0
    for i in range(nb):
        u = rbB[i]
        if cnt[u] == 1 and u != a:
            rbB[nB] = u
            nB += 1
    if nA == 0 and nB == 0 and not ab_bnd:
        return -1.0
    worst = 0.0
    # old boundary points (a, b) -> new polyline (v' - n)
    for pi in range(2):
        p = a if pi == 0 else b
        on_b = ab_bnd or (nA > 0 if pi == 0 else nB > 0)
        if not on_b:
            continue
        dmin = np.inf
        for i in range(nA + nB):
            n = rbA[i] if i < nA else rbB[i - nA]
            dmin = min(dmin, _seg_d2(P[p, 0], P[p, 1], P[p, 2], P, n, n, x, y, z, True))
        worst = max(worst, dmin)
    # new point v' -> old polyline
    dmin = np.inf
    if ab_bnd:
        dmin = _seg_d2(x, y, z, P, a, b, 0.0, 0.0, 0.0, False)
    for i in range(nA):
        dmin = min(dmin, _seg_d2(x, y, z, P, a, rbA[i], 0.0, 0.0, 0.0, False))
    for i in range(nB):
        dmin = min(dmin, _seg_d2(x, y, z, P, b, rbB[i], 0.0, 0.0, 0.0, False))
    return max(worst, dmin)


@njit(cache=True)
def _eval_edge(a, b, P, Q, F, pool, st, ln, mark, cnt, tag, rbA, rbB, prm, S, best):
    """COMPUTECOST (Algorithm 1, l.17-23). Writes v' into best, returns cost_total."""
    w_area = prm[0]; max_move = prm[2]; min_edge2 = prm[3]; area_norm = prm[4]; rel_eps = prm[6]
    ax = P[a, 0]; ay = P[a, 1]; az = P[a, 2]
    bx = P[b, 0]; by = P[b, 1]; bz = P[b, 2]
    len2 = (ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2
    if len2 < min_edge2:
        return np.inf
    Qs = S[0:10]; QA = S[10:20]; sol = S[20:23]; Am = S[23:32]; Vm = S[32:41]
    for k in range(QN):
        Qs[k] = Q[a, k] + Q[b, k]
    has_area = False
    if w_area > 0:
        for k in range(QN):
            QA[k] = 0.0
        h = 0.5 * area_norm
        n = _ring(a, F, pool, st, ln, mark, cnt, tag, rbA)
        for i in range(n):
            if cnt[rbA[i]] == 1:
                _add_area_edge(QA, P, a, rbA[i], h)
                has_area = True
        n = _ring(b, F, pool, st, ln, mark, cnt, tag, rbB)
        for i in range(n):
            u = rbB[i]
            if cnt[u] == 1 and u != a:
                _add_area_edge(QA, P, b, u, h)
                has_area = True
    mx = (ax + bx) / 2; my = (ay + by) / 2; mz = (az + bz) / 2
    best_cost = np.inf
    # FINDOPTIMALPOSITION with fallback candidates {v_i, v_j, midpoint}.
    # All candidates are kept in S[41:45] / S[45:57] for the flip-retry in run_collapses.
    for cand in range(4):
        S[41 + cand] = np.inf
        if cand == 0:
            if not _solve_q(Qs, mx, my, mz, sol, rel_eps, Am, Vm):
                continue
            x = sol[0]; y = sol[1]; z = sol[2]
            if (x - mx) ** 2 + (y - my) ** 2 + (z - mz) ** 2 > max_move * max_move * len2:
                continue
        elif cand == 1:
            x = ax; y = ay; z = az
        elif cand == 2:
            x = bx; y = by; z = bz
        else:
            x = mx; y = my; z = mz
        c = _eval_q(Qs, x, y, z)
        if has_area:
            c += w_area * _eval_q(QA, x, y, z)
        S[41 + cand] = max(0.0, c)
        S[45 + 3 * cand] = x; S[46 + 3 * cand] = y; S[47 + 3 * cand] = z
        if c < best_cost:
            best_cost = c
            best[0] = x; best[1] = y; best[2] = z
    return max(0.0, best_cost)


@njit(cache=True)
def _flip_ok(P, F, pool, st, ln, v, other, x, y, z, thr, degthr=1e-40):
    s = st[v]
    for i in range(ln[v]):
        f = pool[s + i]
        if _face_has(F, f, other):
            continue
        i0 = F[f, 0]; i1 = F[f, 1]; i2 = F[f, 2]
        ux = P[i1, 0] - P[i0, 0]; uy = P[i1, 1] - P[i0, 1]; uz = P[i1, 2] - P[i0, 2]
        wx = P[i2, 0] - P[i0, 0]; wy = P[i2, 1] - P[i0, 1]; wz = P[i2, 2] - P[i0, 2]
        ox = uy * wz - uz * wy; oy = uz * wx - ux * wz; oz = ux * wy - uy * wx
        q0x = x if i0 == v else P[i0, 0]; q0y = y if i0 == v else P[i0, 1]; q0z = z if i0 == v else P[i0, 2]
        q1x = x if i1 == v else P[i1, 0]; q1y = y if i1 == v else P[i1, 1]; q1z = z if i1 == v else P[i1, 2]
        q2x = x if i2 == v else P[i2, 0]; q2y = y if i2 == v else P[i2, 1]; q2z = z if i2 == v else P[i2, 2]
        ux = q1x - q0x; uy = q1y - q0y; uz = q1z - q0z
        wx = q2x - q0x; wy = q2y - q0y; wz = q2z - q0z
        nx = uy * wz - uz * wy; ny = uz * wx - ux * wz; nz = ux * wy - uy * wx
        oo = ox * ox + oy * oy + oz * oz
        nn = nx * nx + ny * ny + nz * nz
        if oo < degthr:
            continue  # already degenerate
        if nn < 1e-12 * oo:
            return False  # would become degenerate
        if ox * nx + oy * ny + oz * nz < thr * np.sqrt(oo * nn):
            return False  # normal flip (Sec. 3.3)
    return True


@njit(cache=True)
def _lh_check(a, b, x, y, z, P, F, pool, st, ln, K):
    """Local worst-case guarantee. Every tracked original point (K[2]) belongs to one current
    face (lists K[4]/K[5]) and lies within its tolerance (K[3], squared) of that face. When a
    and b merge at (x, y, z), the points of all faces around a and b must stay within their
    tolerance of the new fan; each one's nearest fan face is kept in K[6] for the re-linking.
    The new vertex must also lie within tolerance of the original surface (the other direction)."""
    ptP = K[2]; tol2 = K[3]; fhead = K[4]; pnext = K[5]; pbest = K[6]; vtol2 = K[7]
    fan = K[9]; T3 = K[10]; F3 = K[11]; tmp = K[12]; stack = K[13]; sd = K[14]; out = K[15]
    nf = 0
    for i in range(ln[a]):
        f = pool[st[a] + i]
        if not _face_has(F, f, b) and nf < fan.shape[0]:
            fan[nf] = f
            nf += 1
    for i in range(ln[b]):
        f = pool[st[b] + i]
        if not _face_has(F, f, a) and nf < fan.shape[0]:
            fan[nf] = f
            nf += 1
    if nf == 0:
        return True
    lim = min(vtol2[a], vtol2[b])
    fb, d2 = bvh_closest(K[16], K[17], K[18], K[19], K[20], K[21], K[22], x, y, z, -1,
                         0, K[24], -1, -1, -1, K[23], 0.0, 0.0, 0.0, stack, sd, out, tmp, lim * 1.0001)
    if fb < 0 or d2 > lim:
        return False
    # the new faces themselves must stay close to the original: sample each on a barycentric
    # grid whose spacing follows the tolerance (large faces can drift away between their corners)
    sl = np.sqrt(lim)
    for j in range(nf):
        g = fan[j]
        kv = 0
        for k in range(3):
            u = F[g, k]
            if u == a or u == b:
                T3[k, 0] = x; T3[k, 1] = y; T3[k, 2] = z
                kv = k
            else:
                T3[k, 0] = P[u, 0]; T3[k, 1] = P[u, 1]; T3[k, 2] = P[u, 2]
        emax = 0.0
        for k in range(3):
            k2 = (k + 1) % 3
            e = np.sqrt((T3[k, 0] - T3[k2, 0]) ** 2 + (T3[k, 1] - T3[k2, 1]) ** 2 + (T3[k, 2] - T3[k2, 2]) ** 2)
            emax = max(emax, e)
        ng = int(min(8.0, max(2.0, np.ceil(emax / (3.0 * sl)))))
        for gi in range(ng + 1):
            for gj in range(ng + 1 - gi):
                gk = ng - gi - gj
                if gi == ng or gj == ng or gk == ng:
                    continue                          # the corners themselves
                if (kv == 0 and gi == 0) or (kv == 1 and gj == 0) or (kv == 2 and gk == 0):
                    continue                          # on the edge opposite v*: it did not move
                wa, wb = gi / ng, gj / ng
                wc = 1.0 - wa - wb
                qx = wa * T3[0, 0] + wb * T3[1, 0] + wc * T3[2, 0]
                qy = wa * T3[0, 1] + wb * T3[1, 1] + wc * T3[2, 1]
                qz = wa * T3[0, 2] + wb * T3[1, 2] + wc * T3[2, 2]
                fb, d2 = bvh_closest(K[16], K[17], K[18], K[19], K[20], K[21], K[22], qx, qy, qz, -1,
                                     0, K[24], -1, -1, -1, K[23], 0.0, 0.0, 0.0, stack, sd, out, tmp,
                                     lim * 1.0001)
                if fb < 0 or d2 > lim:
                    return False
    for side in range(2):
        v = a if side == 0 else b
        o = b if side == 0 else a
        for i in range(ln[v]):
            f = pool[st[v] + i]
            if side == 1 and _face_has(F, f, o):
                continue                      # shared face, already visited from a
            p = fhead[f]
            while p >= 0:
                px, py, pz = ptP[p, 0], ptP[p, 1], ptP[p, 2]
                t = tol2[p]
                best = np.inf
                bf = -1
                for j in range(nf):
                    g = fan[j]
                    for k in range(3):
                        u = F[g, k]
                        if u == a or u == b:
                            T3[k, 0] = x; T3[k, 1] = y; T3[k, 2] = z
                        else:
                            T3[k, 0] = P[u, 0]; T3[k, 1] = P[u, 1]; T3[k, 2] = P[u, 2]
                    dd = closest_on_tri(T3, F3, 0, px, py, pz, tmp)
                    if dd < best:
                        best = dd
                        bf = g
                        if dd <= 0.01 * t:
                            break
                if best > t:
                    return False
                pbest[p] = bf
                p = pnext[p]
    return True


@njit(cache=True)
def _lh_relink(a, b, F, pool, st, ln, K, cnt_buf):
    """Before the collapse: detach the points of every face around a and b (kept in K[8])."""
    fhead = K[4]; pnext = K[5]; buf = K[8]
    n = 0
    for side in range(2):
        v = a if side == 0 else b
        o = b if side == 0 else a
        for i in range(ln[v]):
            f = pool[st[v] + i]
            if side == 1 and _face_has(F, f, o):
                continue
            p = fhead[f]
            while p >= 0:
                buf[n] = p
                n += 1
                p = pnext[p]
            fhead[f] = -1
    cnt_buf[0] = n


@njit(cache=True)
def _lh_attach(K, n):
    """After the collapse: attach the detached points to their nearest fan face."""
    fhead = K[4]; pnext = K[5]; pbest = K[6]; buf = K[8]
    for i in range(n):
        p = buf[i]
        g = pbest[p]
        pnext[p] = fhead[g]
        fhead[g] = p


@njit(cache=True)
def _can_collapse(a, b, x, y, z, P, F, pool, st, ln, vpool, vst, vln,
                  mark, cnt, tag, mark2, tag2, rbA, rbB, prm, C, cA, oppc, allow_topo, K):
    """0 = reject, 1 = real edge OK, 2 = virtual edge OK, 3 = rejected by a normal flip only,
    4 = would change topology (link condition / boundary pinch)."""
    # error cap: area-weighted mean squared distance to the original face planes
    if prm[8] < np.inf:
        Qgeo = K[0]; Ageo = K[1]
        cap = prm[8]
        e = 0.0
        qg = np.empty(QN)
        for k in range(QN):
            qg[k] = Qgeo[a, k] + Qgeo[b, k]
        e = _eval_q(qg, x, y, z) / max(Ageo[a] + Ageo[b], 1e-300)
        if e > cap:
            C[C_ERR] += 1
            return 0
        # open boundaries: the plane-distance cap above cannot see a rim sliding within its
        # own plane (an open patch could shrink away) -> bound the rim movement as well
        bd = _boundary_dev2(a, b, x, y, z, P, F, pool, st, ln, mark, cnt, tag, rbA, rbB)
        if bd > cap:
            C[C_ERR] += 1
            return 0
    tag2[0] += 1
    t2 = tag2[0]
    shared = 0
    sb = st[b]
    for i in range(ln[b]):
        f = pool[sb + i]
        if not _face_has(F, f, a):
            continue
        shared += 1
        for k in range(3):
            u = F[f, k]
            if u != a and u != b:
                if mark2[u] != t2:
                    mark2[u] = t2
                    oppc[u] = 0
                oppc[u] += 1
    virtual = shared == 0
    if virtual:
        if not _ml_has(vpool, vst, vln, a, b):
            return 0  # stale: neither real nor virtual
    elif prm[5] > 0 and not allow_topo:  # link condition
        n = _ring(a, F, pool, st, ln, mark, cnt, tag, rbA)
        ta = tag[0]
        a_bound = False
        for i in range(n):
            if cnt[rbA[i]] == 1:
                a_bound = True
                break
        for i in range(ln[b]):
            f = pool[sb + i]
            for k in range(3):
                u = F[f, k]
                if u == a or u == b:
                    continue
                if mark[u] == ta and mark2[u] != t2:
                    C[C_LINK] += 1
                    return 4
        if shared >= 2 and a_bound:
            n = _ring(b, F, pool, st, ln, mark, cnt, tag, rbB)
            for i in range(n):
                if cnt[rbB[i]] == 1:
                    C[C_LINK] += 1
                    return 4  # would pinch two boundary loops
    # edge-valence guard (generalised manifoldness): after the collapse the edge (a,x)
    # carries cnt_a[x] + cnt_b[x] - 2*opp[x] faces. Never create a new non-manifold edge
    # or grow an existing one - otherwise folded "fins" stack up on meshes with touching
    # layers / virtual junctions (every stacked face would still pass the flip test).
    n = _ring(a, F, pool, st, ln, mark, cnt, tag, rbA)
    for i in range(n):
        cA[rbA[i]] = cnt[rbA[i]]
    nb = _ring(b, F, pool, st, ln, mark, cnt, tag, rbB)
    bad = False
    for i in range(nb):
        u = rbB[i]
        ca = cA[u]
        if u == a or ca == 0:
            continue  # (a,u) is new: it inherits exactly (b,u)'s faces
        cb = cnt[u]
        o_ = oppc[u] if mark2[u] == t2 else 0
        new = ca + cb - 2 * o_
        if new > 2 and new > max(ca, cb):
            bad = True
            break
    for i in range(n):
        cA[rbA[i]] = 0
    if bad and not allow_topo:
        C[C_LINK] += 1
        return 4
    # duplicate faces: (b,x,y) -> (a,x,y) must not exist already
    sa = st[a]
    for i in range(ln[b]):
        f = pool[sb + i]
        if _face_has(F, f, a):
            continue
        x1 = -1; y1 = -1
        for k in range(3):
            u = F[f, k]
            if u != b:
                if x1 < 0:
                    x1 = u
                else:
                    y1 = u
        if allow_topo:
            break  # duplicates are resolved by the collapse itself
        for j in range(ln[a]):
            g = pool[sa + j]
            if _face_has(F, g, x1) and _face_has(F, g, y1):
                C[C_DUP] += 1
                return 4
    if not _flip_ok(P, F, pool, st, ln, a, b, x, y, z, prm[1]) or \
       not _flip_ok(P, F, pool, st, ln, b, a, x, y, z, prm[1]):
        C[C_FLIP] += 1
        return 3
    if prm[9] > 0 and not _lh_check(a, b, x, y, z, P, F, pool, st, ln, K):
        C[C_ERR] += 1
        return 0
    return 2 if virtual else 1


@njit(cache=True)
def _push_edge(H, hs, a, b, P, Q, F, pool, st, ln, mark, cnt, tag, rbA, rbB, prm, S, best, stamp):
    c = _eval_edge(a, b, P, Q, F, pool, st, ln, mark, cnt, tag, rbA, rbB, prm, S, best)
    if c < np.inf:
        H = _hpush(H, hs, c, a, b, stamp[a], stamp[b], best[0], best[1], best[2])
    return H


@njit(cache=True)
def init_queue(edges, vpairs, P, Q, F, pool, st, ln, mark, cnt, tag, rbA, rbB, prm, stamp, H, hs):
    """POPULATEPRIORITYQUEUE."""
    S = np.zeros(64)
    best = np.zeros(3)
    for e in range(edges.shape[0]):
        H = _push_edge(H, hs, edges[e, 0], edges[e, 1], P, Q, F, pool, st, ln, mark, cnt, tag,
                       rbA, rbB, prm, S, best, stamp)
    for e in range(vpairs.shape[0]):
        H = _push_edge(H, hs, vpairs[e, 0], vpairs[e, 1], P, Q, F, pool, st, ln, mark, cnt, tag,
                       rbA, rbB, prm, S, best, stamp)
    return H


@njit(cache=True, inline="always")
def nV_arr_len(P):
    return P.shape[0]


@njit(cache=True)
def run_collapses(P, Q, F, face_alive, v_alive, stamp, parent,
                  pool, st, ln, cp, used, vpool, vst, vln, vcp, vused,
                  H, hs, mark, cnt, tag, mark2, tag2, rbA, rbB, prm, C,
                  target_faces, stop_faces, max_rebuilds, hist, K):
    """Main loop of Algorithm 1 (l.5-12); runs until faces <= stop_faces."""
    S = np.zeros(64)
    best = np.zeros(3)
    cA = np.zeros(nV_arr_len(P), np.int64)
    oppc = np.zeros(nV_arr_len(P), np.int64)
    lh_n = np.zeros(1, np.int64)
    top = np.zeros(8)
    tmp = np.zeros(8)
    nV = P.shape[0]
    stop = max(target_faces, stop_faces)
    while C[C_FACES] > stop:
        if hs[0] == 0:
            if C[C_SINCE] == 0 or C[C_REBUILD] >= max_rebuilds:
                C[C_DONE] = 1
                break
            C[C_REBUILD] += 1
            C[C_SINCE] = 0
            for a in range(nV):
                if not v_alive[a]:
                    continue
                n = _ring(a, F, pool, st, ln, mark, cnt, tag, rbA)
                for i in range(n):
                    u = rbA[i]
                    if u > a:
                        rbB[i] = u  # copy: _eval_edge reuses rbA
                    else:
                        rbB[i] = -1
                nb = rbB[:n].copy()
                for u in nb:
                    if u >= 0:
                        H = _push_edge(H, hs, a, u, P, Q, F, pool, st, ln, mark, cnt, tag,
                                       rbA, rbB, prm, S, best, stamp)
                for i in range(vln[a]):
                    u = vpool[vst[a] + i]
                    if u > a:
                        H = _push_edge(H, hs, a, u, P, Q, F, pool, st, ln, mark, cnt, tag,
                                       rbA, rbB, prm, S, best, stamp)
            if hs[0] == 0:
                C[C_DONE] = 1
                break
        _hpop(H, hs, top, tmp)
        a = int(top[1]); b = int(top[2])
        if not v_alive[a] or not v_alive[b] or stamp[a] != int(top[3]) or stamp[b] != int(top[4]):
            C[C_STALE] += 1
            continue
        x = top[5]; y = top[6]; z = top[7]
        # entries re-queued with a topology penalty carry a half-integer stamp as their flag
        topo_ok = top[3] - np.floor(top[3]) > 0.25
        ok = _can_collapse(a, b, x, y, z, P, F, pool, st, ln, vpool, vst, vln,
                           mark, cnt, tag, mark2, tag2, rbA, rbB, prm, C, cA, oppc, topo_ok, K)
        if ok == 4:
            if prm[7] < np.inf:
                H = _hpush(H, hs, top[0] * prm[7] + 1e-12, a, b, stamp[a] + 0.5, stamp[b], x, y, z)
            continue
        if ok == 3:
            # the optimal v' folds a face: re-queue the edge with the cheapest candidate
            # position (from {v*, v_i, v_j, mid}) that keeps every face orientation.
            _eval_edge(a, b, P, Q, F, pool, st, ln, mark, cnt, tag, rbA, rbB, prm, S, best)
            for _r in range(4):
                bi = -1
                bc = np.inf
                for c in range(4):
                    if S[41 + c] < bc:
                        bc = S[41 + c]
                        bi = c
                if bi < 0:
                    break
                S[41 + bi] = np.inf
                cx = S[45 + 3 * bi]; cy = S[46 + 3 * bi]; cz = S[47 + 3 * bi]
                if cx == x and cy == y and cz == z:
                    continue
                if _flip_ok(P, F, pool, st, ln, a, b, cx, cy, cz, prm[1]) and \
                   _flip_ok(P, F, pool, st, ln, b, a, cx, cy, cz, prm[1]):
                    H = _hpush(H, hs, max(bc, top[0]), a, b, stamp[a] + (0.5 if topo_ok else 0.0),
                               stamp[b], cx, cy, cz)
                    break
            continue
        if ok == 0:
            continue
        if ok == 2:
            C[C_VIRT] += 1
        # ---- COLLAPSEEDGE -----------------------------------------------------------------
        # a collapse to one endpoint keeps that vertex; otherwise keep the larger fan
        at_a = x == P[a, 0] and y == P[a, 1] and z == P[a, 2]
        at_b = x == P[b, 0] and y == P[b, 1] and z == P[b, 2]
        if at_b and not at_a:
            t = a; a = b; b = t
        elif not at_a and ln[b] > ln[a]:
            t = a; a = b; b = t
        # collapse history H (Algorithm 1, l.8): kept, removed, v', cost, kind
        hk = C[C_COLL]
        if hk < hist.shape[0]:
            hist[hk, 0] = a; hist[hk, 1] = b
            hist[hk, 2] = x; hist[hk, 3] = y; hist[hk, 4] = z
            hist[hk, 5] = top[0]; hist[hk, 6] = ok
        if prm[9] > 0:
            _lh_relink(a, b, F, pool, st, ln, K, lh_n)
            vt = K[7]
            vt[a] = min(vt[a], vt[b])
        P[a, 0] = x; P[a, 1] = y; P[a, 2] = z
        Qgeo = K[0]; Ageo = K[1]
        for k in range(QN):
            Q[a, k] += Q[b, k]
            Qgeo[a, k] += Qgeo[b, k]
        Ageo[a] += Ageo[b]
        removed = False
        sbb = st[b]
        for i in range(ln[b]):
            f = pool[sbb + i]
            if _face_has(F, f, a):
                face_alive[f] = 0
                C[C_FACES] -= 1
                removed = True
                for k in range(3):
                    u = F[f, k]
                    if u != a and u != b:
                        _ml_remove(pool, st, ln, u, f)
            else:
                for k in range(3):
                    if F[f, k] == b:
                        F[f, k] = a
                pool = _ml_push(pool, st, ln, cp, used, a, f)
        if removed:
            s = st[a]
            w = 0
            for i in range(ln[a]):
                f = pool[s + i]
                if face_alive[f]:
                    pool[s + w] = f
                    w += 1
            ln[a] = w
        if topo_ok:
            # a topology-changing collapse may fold two faces onto the same vertex triple:
            # drop such zero-volume pairs (one copy if they have the same orientation)
            s = st[a]
            i = 0
            while i < ln[a]:
                f = pool[s + i]
                dup = -1
                for j in range(i + 1, ln[a]):
                    g = pool[s + j]
                    if _face_has(F, g, F[f, 0]) and _face_has(F, g, F[f, 1]) and _face_has(F, g, F[f, 2]):
                        dup = g
                        break
                if dup < 0:
                    i += 1
                    continue
                same = False
                for k in range(3):
                    if F[f, 0] == F[dup, k] and F[f, 1] == F[dup, (k + 1) % 3]:
                        same = True
                kill_n = 1 if same else 2
                for kk in range(kill_n):
                    g = dup if kk == 0 else f
                    face_alive[g] = 0
                    C[C_FACES] -= 1
                    for k in range(3):
                        _ml_remove(pool, st, ln, F[g, k], g)
                i = 0  # list changed: rescan
        ln[b] = 0
        v_alive[b] = 0
        parent[b] = a
        if prm[9] > 0:
            _lh_attach(K, lh_n[0])
        stamp[a] += 1
        stamp[b] += 1
        # transfer virtual neighbours of b to a
        for i in range(vln[b]):
            u = vpool[vst[b] + i]
            _ml_remove(vpool, vst, vln, u, b)
            if u == a:
                continue
            if not _ml_has(vpool, vst, vln, u, a):
                vpool = _ml_push(vpool, vst, vln, vcp, vused, u, a)
                vpool = _ml_push(vpool, vst, vln, vcp, vused, a, u)
        vln[b] = 0
        # ---- UPDATENEIGHBORCOSTS -------------------------------------------------------
        n = _ring(a, F, pool, st, ln, mark, cnt, tag, rbA)
        nb = rbA[:n].copy()
        for u in nb:
            H = _push_edge(H, hs, a, u, P, Q, F, pool, st, ln, mark, cnt, tag,
                           rbA, rbB, prm, S, best, stamp)
        for i in range(vln[a]):
            u = vpool[vst[a] + i]
            H = _push_edge(H, hs, a, u, P, Q, F, pool, st, ln, mark, cnt, tag,
                           rbA, rbB, prm, S, best, stamp)
        C[C_COLL] += 1
        C[C_SINCE] += 1
    return pool, vpool, H


# =====================================================================================
# Python driver
# =====================================================================================
def _build_lists(nV, lists_src, lists_dst, slack=2):
    """Pool-backed per-vertex lists from (src -> dst) pairs."""
    order = np.argsort(lists_src, kind="stable")
    src, dst = lists_src[order], lists_dst[order]
    counts = np.bincount(src, minlength=nV).astype(np.int64)
    cap = np.maximum(counts * slack, 4).astype(np.int64)
    st = np.zeros(nV, np.int64)
    st[1:] = np.cumsum(cap)[:-1]
    used = np.array([int(cap.sum())], np.int64)
    pool = np.empty(int(used[0] * 1.25) + 16, np.int64)
    first = np.zeros(nV, np.int64)
    first[1:] = np.cumsum(counts)[:-1]
    pos = st[src] + (np.arange(len(src)) - first[src])
    pool[pos] = dst
    return pool, st, counts.copy(), cap, used


@njit(cache=True)
def _nearest_incident_face(pts, v0, P, F, pool, st, ln, out):
    """For each point, the face incident to vertex v0[i] that lies closest to it."""
    T3 = np.empty((3, 3))
    F3 = np.zeros((1, 3), np.int64)
    F3[0, 1] = 1
    F3[0, 2] = 2
    tmp = np.empty(6)
    for i in range(pts.shape[0]):
        v = v0[i]
        best = np.inf
        bf = -1
        for j in range(ln[v]):
            f = pool[st[v] + j]
            for k in range(3):
                for c in range(3):
                    T3[k, c] = P[F[f, k], c]
            d = closest_on_tri(T3, F3, 0, pts[i, 0], pts[i, 1], pts[i, 2], tmp)
            if d < best:
                best = d
                bf = f
        out[i] = bf


def _lh_data(on, P_orig, F_orig, P, F, pool, st, ln, tol, reference):
    """Tracked points for the local guarantee (see _lh_check): the original vertices, plus the
    centroids of the original faces on inputs up to 400k faces (their interiors count too)."""
    nV, nF = len(P), len(F)
    if not on:
        dummy = (np.zeros((1, 3)), np.zeros(1), np.full(1, -1, np.int64), np.full(1, -1, np.int64),
                 np.zeros(1, np.int64), np.full(1, np.inf), np.zeros(1, np.int64), np.zeros(1, np.int64),
                 np.zeros((3, 3)), np.array([[0, 1, 2]], np.int64), np.zeros(6), np.zeros(256, np.int64),
                 np.zeros(256), np.zeros(6))
        ref = (np.zeros((1, 3)), np.zeros((1, 3), np.int64), np.zeros(1, np.int64), np.zeros((1, 6)),
               np.full(1, -1, np.int64), np.zeros(1, np.int64), np.zeros(1, np.int64))
        return dummy + ref + (np.zeros((1, 3)), np.zeros((1, 3), np.int64))
    keep = ln > 0
    vids = np.nonzero(keep)[0]
    pts = [P_orig[vids]]
    pface = [pool[st[vids]]]
    if len(F_orig) <= 400_000:
        cf = np.nonzero(keep[F_orig].all(1))[0]
        if len(cf):
            c = P_orig[F_orig[cf]].mean(1)
            near = np.empty(len(cf), np.int64)
            _nearest_incident_face(c, F_orig[cf, 0].astype(np.int64), P, F, pool, st, ln, near)
            ok = near >= 0
            pts.append(c[ok])
            pface.append(near[ok])
    ptP = np.ascontiguousarray(np.concatenate(pts))
    face = np.concatenate(pface).astype(np.int64)
    nP = len(ptP)
    tol2 = np.full(nP, tol * tol)
    fhead = np.full(nF, -1, np.int64)
    pnext = np.full(nP, -1, np.int64)
    order = np.argsort(face, kind="stable")
    fs = face[order]
    first = np.r_[True, fs[1:] != fs[:-1]]
    # chain the points of each face: p0 -> p1 -> ... in sorted order
    pnext[order[:-1]] = np.where(fs[1:] == fs[:-1], order[1:], -1)
    fhead[fs[first]] = order[first]
    vtol2 = np.full(nV, tol * tol)
    ref = reference if reference is not None else BVH(P_orig, F_orig)
    return (ptP, tol2, fhead, pnext, np.zeros(nP, np.int64), vtol2,
            np.zeros(nP, np.int64), np.zeros(nF + 16, np.int64),
            np.zeros((3, 3)), np.array([[0, 1, 2]], np.int64), np.zeros(6), np.zeros(256, np.int64),
            np.zeros(256), np.zeros(6)) + tuple(ref.arrays) + (np.zeros((1, 3)), np.zeros((1, 3), np.int64))


def simplify(P, F, target_faces, options=None, progress=None, reference=None):
    """
    FA-QEM simplification of a prepared (welded, unit-diagonal) mesh.

    Returns dict(positions, faces, vertex_map, stats, history); vertex_map maps every input
    vertex to the output vertex it was collapsed into (the collapse lineage used for successive
    mapping / appearance transfer).
    Automatic mode (options auto=True, max_error and local_tol set) has no face target: it runs
    until the error limits stop it. reference: BVH of (P, F), reused by the local guarantee.
    """
    o = dict(DEFAULTS)
    if options:
        o.update({k: v for k, v in options.items() if v is not None})
    if o["auto"]:
        if not o["max_error"]:
            raise ValueError("auto mode needs max_error (the allowed deviation)")
        target_faces = 1
        # no face target to push for -> never change topology (small parts must not vanish)
        o["topology_penalty"] = np.inf
    t0 = time.perf_counter()
    P_in, F_in = P, F
    P = np.array(P, dtype=np.float64, order="C")
    F = np.array(F, dtype=np.int64, order="C")
    nV, nF = len(P), len(F)
    target_faces = int(max(1, target_faces))

    edges, counts = unique_edges(F)
    Q, mean_area, mean_edge, n_bverts, Qgeo, Ageo = build_quadrics(P, F, edges, counts, o)
    t_quadric = time.perf_counter()

    fid = np.repeat(np.arange(nF, dtype=np.int64), 3)
    pool, st, ln, cp, used = _build_lists(nV, F.reshape(-1), fid)

    vpairs = np.zeros((0, 2), np.int64)
    if o["virtual_edges"]:
        vpairs = find_virtual_edges(P, F, o["virtual_tau"])
    if len(vpairs):
        vpool, vst, vln, vcp, vused = _build_lists(
            nV, np.concatenate([vpairs[:, 0], vpairs[:, 1]]), np.concatenate([vpairs[:, 1], vpairs[:, 0]]))
    else:
        vpool, vst, vln, vcp, vused = _build_lists(nV, np.zeros(0, np.int64), np.zeros(0, np.int64))
    t_virtual = time.perf_counter()

    lh_on = bool(o["local_tol"])
    K = (Qgeo, Ageo) + _lh_data(lh_on, np.asarray(P_in, np.float64), np.asarray(F_in, np.int64), P, F,
                                pool, st, ln, float(o["local_tol"] or 0.0), reference)
    prm = np.array([
        o["w_area"], o["flip_threshold"], o["max_move_factor"], o["min_edge_rel"] ** 2,
        1.0 / (mean_edge * mean_edge), 1.0 if o["preserve_topology"] else 0.0, 1e-3,
        float(o["topology_penalty"]),
        float(o["max_error"]) ** 2 if o["max_error"] else np.inf,   # [8] error cap
        1.0 if lh_on else 0.0,                                       # [9] local guarantee
    ])
    mark = np.full(nV, -1, np.int64); cnt = np.zeros(nV, np.int64); tag = np.zeros(1, np.int64)
    mark2 = np.full(nV, -1, np.int64); tag2 = np.zeros(1, np.int64)
    rbA = np.empty(nV + 1, np.int64); rbB = np.empty(nV + 1, np.int64)
    stamp = np.zeros(nV, np.int64)
    H = np.empty((int(len(edges) * 1.5) + 64, 8))
    hs = np.zeros(1, np.int64)
    H = init_queue(edges, vpairs, P, Q, F, pool, st, ln, mark, cnt, tag, rbA, rbB, prm, stamp, H, hs)
    t_queue = time.perf_counter()

    face_alive = np.ones(nF, np.uint8)
    v_alive = np.ones(nV, np.uint8)
    parent = np.arange(nV, dtype=np.int64)
    C = np.zeros(N_COUNTERS, np.int64)
    C[C_FACES] = nF
    hist = np.zeros((nV, 7))
    steps = np.unique(np.linspace(nF, target_faces, 40).astype(np.int64))[::-1]
    rebuilds = 1 if o["auto"] else int(o["max_rebuilds"])
    for stop in steps:
        pool, vpool, H = run_collapses(
            P, Q, F, face_alive, v_alive, stamp, parent, pool, st, ln, cp, used,
            vpool, vst, vln, vcp, vused, H, hs, mark, cnt, tag, mark2, tag2, rbA, rbB, prm, C,
            target_faces, int(stop), rebuilds, hist, K)
        if progress:
            progress((nF - C[C_FACES]) / max(1, nF - target_faces), int(C[C_FACES]))
        if C[C_DONE] or C[C_FACES] <= target_faces:
            break
    t_loop = time.perf_counter()

    # ---- compact output -------------------------------------------------------------
    alive_f = face_alive.astype(bool)
    Fo = F[alive_f]
    used_v = np.zeros(nV, bool)
    used_v[Fo.reshape(-1)] = True
    new_index = np.full(nV, -1, np.int64)
    new_index[used_v] = np.arange(int(used_v.sum()))
    out_F = new_index[Fo]
    out_P = P[used_v]
    # collapse lineage (successive mapping): original vertex -> final vertex
    root = parent.copy()
    while True:
        nxt = root[root]
        if np.array_equal(nxt, root):
            break
        root = nxt
    vertex_map = new_index[root]

    stats = dict(
        input_faces=nF, input_vertices=nV,
        output_faces=int(len(out_F)), output_vertices=int(len(out_P)),
        collapses=int(C[C_COLL]), rejected_flip=int(C[C_FLIP]), rejected_link=int(C[C_LINK]),
        rejected_duplicate=int(C[C_DUP]), rejected_error=int(C[C_ERR]),
        stale=int(C[C_STALE]), rebuilds=int(C[C_REBUILD]),
        virtual_edges=int(len(vpairs)), virtual_collapses=int(C[C_VIRT]),
        boundary_vertices=n_bverts, reached_target=bool(len(out_F) <= target_faces),
        time_quadrics=t_quadric - t0, time_virtual=t_virtual - t_quadric,
        time_queue=t_queue - t_virtual, time_collapse=t_loop - t_queue,
        time_total=time.perf_counter() - t0,
    )
    return dict(positions=out_P, faces=out_F, vertex_map=vertex_map, stats=stats,
                history=hist[:int(C[C_COLL])])
