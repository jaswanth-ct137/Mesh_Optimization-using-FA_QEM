"""Triangle BVH with closest-point queries (Numba), optionally filtered by collapse lineage."""
import numpy as np
from numba import njit


@njit(cache=True)
def _nth(idx, cen, axis, lo, hi, k):
    while hi > lo:
        pivot = cen[idx[(lo + hi) >> 1], axis]
        i = lo
        j = hi
        while i <= j:
            while cen[idx[i], axis] < pivot:
                i += 1
            while cen[idx[j], axis] > pivot:
                j -= 1
            if i <= j:
                t = idx[i]; idx[i] = idx[j]; idx[j] = t
                i += 1; j -= 1
        if k <= j:
            hi = j
        elif k >= i:
            lo = i
        else:
            return


@njit(cache=True)
def build_bvh(P, F, leaf_size=6):
    nF = F.shape[0]
    idx = np.arange(nF)
    cen = np.empty((nF, 3))
    for f in range(nF):
        for k in range(3):
            cen[f, k] = (P[F[f, 0], k] + P[F[f, 1], k] + P[F[f, 2], k]) / 3.0
    max_nodes = 2 * nF + 8
    box = np.empty((max_nodes, 6))
    left = np.full(max_nodes, -1, np.int64)
    start = np.zeros(max_nodes, np.int64)
    count = np.zeros(max_nodes, np.int64)
    stack = np.empty((128, 3), np.int64)
    sp = 0
    stack[0, 0] = 0; stack[0, 1] = 0; stack[0, 2] = nF
    sp = 1
    n_nodes = 1
    while sp > 0:
        sp -= 1
        node = stack[sp, 0]; s = stack[sp, 1]; e = stack[sp, 2]
        lo0 = np.inf; lo1 = np.inf; lo2 = np.inf; hi0 = -np.inf; hi1 = -np.inf; hi2 = -np.inf
        c0l = np.inf; c1l = np.inf; c2l = np.inf; c0h = -np.inf; c1h = -np.inf; c2h = -np.inf
        for i in range(s, e):
            f = idx[i]
            for k in range(3):
                v = F[f, k]
                x = P[v, 0]; y = P[v, 1]; z = P[v, 2]
                lo0 = min(lo0, x); hi0 = max(hi0, x)
                lo1 = min(lo1, y); hi1 = max(hi1, y)
                lo2 = min(lo2, z); hi2 = max(hi2, z)
            c0l = min(c0l, cen[f, 0]); c0h = max(c0h, cen[f, 0])
            c1l = min(c1l, cen[f, 1]); c1h = max(c1h, cen[f, 1])
            c2l = min(c2l, cen[f, 2]); c2h = max(c2h, cen[f, 2])
        box[node, 0] = lo0; box[node, 1] = lo1; box[node, 2] = lo2
        box[node, 3] = hi0; box[node, 4] = hi1; box[node, 5] = hi2
        ex = c0h - c0l; ey = c1h - c1l; ez = c2h - c2l
        if e - s <= leaf_size or max(ex, ey, ez) <= 0:
            left[node] = -1; start[node] = s; count[node] = e - s
            continue
        axis = 0 if (ex >= ey and ex >= ez) else (1 if ey >= ez else 2)
        mid = (s + e) >> 1
        _nth(idx, cen, axis, s, e - 1, mid)
        l = n_nodes
        n_nodes += 2
        left[node] = l
        if sp + 2 >= stack.shape[0]:
            st2 = np.empty((stack.shape[0] * 2, 3), np.int64)
            st2[:sp] = stack[:sp]
            stack = st2
        stack[sp, 0] = l; stack[sp, 1] = s; stack[sp, 2] = mid; sp += 1
        stack[sp, 0] = l + 1; stack[sp, 1] = mid; stack[sp, 2] = e; sp += 1
    return idx, box[:n_nodes].copy(), left[:n_nodes].copy(), start[:n_nodes].copy(), count[:n_nodes].copy()


@njit(cache=True, inline="always")
def _box_d2(box, n, x, y, z):
    d = 0.0
    v = box[n, 0] - x
    if v > 0:
        d += v * v
    else:
        v = x - box[n, 3]
        if v > 0:
            d += v * v
    v = box[n, 1] - y
    if v > 0:
        d += v * v
    else:
        v = y - box[n, 4]
        if v > 0:
            d += v * v
    v = box[n, 2] - z
    if v > 0:
        d += v * v
    else:
        v = z - box[n, 5]
        if v > 0:
            d += v * v
    return d


@njit(cache=True)
def closest_on_tri(P, F, f, px, py, pz, out):
    """Ericson RTCD 5.1.5. out = [b0, b1, b2, x, y, z]; returns squared distance."""
    a = F[f, 0]; b = F[f, 1]; c = F[f, 2]
    ax = P[a, 0]; ay = P[a, 1]; az = P[a, 2]
    abx = P[b, 0] - ax; aby = P[b, 1] - ay; abz = P[b, 2] - az
    acx = P[c, 0] - ax; acy = P[c, 1] - ay; acz = P[c, 2] - az
    apx = px - ax; apy = py - ay; apz = pz - az
    d1 = abx * apx + aby * apy + abz * apz
    d2 = acx * apx + acy * apy + acz * apz
    v = 0.0; w = 0.0
    if d1 <= 0 and d2 <= 0:
        v = 0.0; w = 0.0
    else:
        bpx = px - P[b, 0]; bpy = py - P[b, 1]; bpz = pz - P[b, 2]
        d3 = abx * bpx + aby * bpy + abz * bpz
        d4 = acx * bpx + acy * bpy + acz * bpz
        if d3 >= 0 and d4 <= d3:
            v = 1.0; w = 0.0
        else:
            vc = d1 * d4 - d3 * d2
            if vc <= 0 and d1 >= 0 and d3 <= 0:
                den = d1 - d3
                v = d1 / den if den != 0 else 0.0
                w = 0.0
            else:
                cpx = px - P[c, 0]; cpy = py - P[c, 1]; cpz = pz - P[c, 2]
                d5 = abx * cpx + aby * cpy + abz * cpz
                d6 = acx * cpx + acy * cpy + acz * cpz
                if d6 >= 0 and d5 <= d6:
                    v = 0.0; w = 1.0
                else:
                    vb = d5 * d2 - d1 * d6
                    if vb <= 0 and d2 >= 0 and d6 <= 0:
                        den = d2 - d6
                        w = d2 / den if den != 0 else 0.0
                        v = 0.0
                    else:
                        va = d3 * d6 - d5 * d4
                        if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
                            den = (d4 - d3) + (d5 - d6)
                            w = (d4 - d3) / den if den != 0 else 0.0
                            v = 1.0 - w
                        else:
                            den = va + vb + vc
                            if abs(den) > 1e-300:
                                v = vb / den
                                w = vc / den
    x = ax + abx * v + acx * w
    y = ay + aby * v + acy * w
    z = az + abz * v + acz * w
    out[0] = 1.0 - v - w; out[1] = v; out[2] = w
    out[3] = x; out[4] = y; out[5] = z
    return (x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2


@njit(cache=True, inline="always")
def _admissible(f, mode, reps, A, B, Cc, fnrm, nx, ny, nz):
    if mode == 0:
        return True
    ok = False
    for k in range(3):
        r = reps[f, k]
        if r == A or r == B or r == Cc:
            ok = True
            break
    if not ok:
        return False
    if mode == 2:
        return fnrm[f, 0] * nx + fnrm[f, 1] * ny + fnrm[f, 2] * nz > 0.0
    return True


@njit(cache=True)
def bvh_closest(P, F, idx, box, left, start, count, px, py, pz, hint,
                mode, reps, A, B, Cc, fnrm, nx, ny, nz, stack, sd, out, tmp, max_d2=np.inf):
    """
    Closest admissible triangle to p. mode 0: any triangle; 1: lineage filter (a corner of the
    triangle collapsed into one of the output vertices A, B, C); 2: lineage + same-facing normal.
    out = [b0, b1, b2, x, y, z]; returns (face, d2) with face = -1 when none admissible.
    """
    best = max_d2  # search radius: filtered searches that find nothing nearby stop early
    bf = -1
    if hint >= 0 and _admissible(hint, mode, reps, A, B, Cc, fnrm, nx, ny, nz):
        d2 = closest_on_tri(P, F, hint, px, py, pz, tmp)
        if d2 < best:
            best = d2
            bf = hint
            for k in range(6):
                out[k] = tmp[k]
    sp = 0
    stack[0] = 0
    sd[0] = _box_d2(box, 0, px, py, pz)
    sp = 1
    while sp > 0:
        sp -= 1
        node = stack[sp]
        if sd[sp] >= best:
            continue
        l = left[node]
        if l < 0:
            s = start[node]
            for i in range(s, s + count[node]):
                f = idx[i]
                if not _admissible(f, mode, reps, A, B, Cc, fnrm, nx, ny, nz):
                    continue
                d2 = closest_on_tri(P, F, f, px, py, pz, tmp)
                if d2 < best:
                    best = d2
                    bf = f
                    for k in range(6):
                        out[k] = tmp[k]
            continue
        dl = _box_d2(box, l, px, py, pz)
        dr = _box_d2(box, l + 1, px, py, pz)
        if sp + 2 >= stack.shape[0]:
            return bf, best  # stack exhausted (depth > 250): return best so far
        if dl < dr:
            if dr < best:
                stack[sp] = l + 1; sd[sp] = dr; sp += 1
            if dl < best:
                stack[sp] = l; sd[sp] = dl; sp += 1
        else:
            if dl < best:
                stack[sp] = l; sd[sp] = dl; sp += 1
            if dr < best:
                stack[sp] = l + 1; sd[sp] = dr; sp += 1
    return bf, best


class BVH:
    def __init__(self, P, F):
        self.P = np.ascontiguousarray(P, dtype=np.float64)
        self.F = np.ascontiguousarray(F, dtype=np.int64)
        self.idx, self.box, self.left, self.start, self.count = build_bvh(self.P, self.F)

    @property
    def arrays(self):
        return self.P, self.F, self.idx, self.box, self.left, self.start, self.count
