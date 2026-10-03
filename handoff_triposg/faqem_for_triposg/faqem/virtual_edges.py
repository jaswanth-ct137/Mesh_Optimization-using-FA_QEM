"""Virtual edge insertion (paper Sec. 3.3, supplementary Sec. 6.2).

1. identify topologically disconnected components (graph connected components),
2. KD-tree of triangle centroids per component; ball query between components with
   threshold tau * L_diag (mesh is normalised, so L_diag = 1),
3. for each proximate triangle pair, the closest vertex pair becomes a virtual collapse
   candidate. To keep the candidate set linear we keep, per vertex, only its closest
   partner in another component.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def face_components(F, nV):
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]]], axis=0)
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(nV, nV))
    n, labels = connected_components(g, directed=False)
    return n, labels[F[:, 0]]


def find_virtual_edges(P, F, tau=0.01, max_components=4000):
    nV = len(P)
    n_comp, fcomp = face_components(F, nV)
    if n_comp <= 1:
        return np.zeros((0, 2), np.int64)
    centroids = P[F].mean(axis=1)
    order = np.argsort(fcomp, kind="stable")
    bounds = np.searchsorted(fcomp[order], np.arange(n_comp + 1))
    comps = [order[bounds[c]:bounds[c + 1]] for c in range(n_comp)]
    # largest components first; tiny floaters beyond the cap are still merged with big ones
    comp_order = np.argsort([-len(c) for c in comps])[:max_components]
    lo = np.array([centroids[comps[c]].min(0) - tau for c in comp_order])
    hi = np.array([centroids[comps[c]].max(0) + tau for c in comp_order])
    trees = {}

    best_d = np.full(nV, np.inf)
    best_u = np.full(nV, -1, np.int64)
    for ii in range(len(comp_order)):
        # candidate partners: components with overlapping (tau-expanded) boxes
        ov = np.all((lo[ii + 1:] <= hi[ii]) & (hi[ii + 1:] >= lo[ii]), axis=1)
        partners = np.nonzero(ov)[0] + ii + 1
        if len(partners) == 0:
            continue
        ci = comp_order[ii]
        if ci not in trees:
            trees[ci] = cKDTree(centroids[comps[ci]])
        for jj in partners:
            cj = comp_order[jj]
            fj = comps[cj]
            d, k = trees[ci].query(centroids[fj], distance_upper_bound=tau)
            hit = np.isfinite(d)
            if not hit.any():
                continue
            fa = comps[ci][k[hit]]      # triangles of component i
            fb = fj[hit]                # proximate triangles of component j
            # closest vertex pair among the 3x3 corner combinations
            va = F[fa]                  # (h,3)
            vb = F[fb]
            diff = P[va][:, :, None, :] - P[vb][:, None, :, :]
            d2 = (diff ** 2).sum(-1).reshape(len(fa), 9)
            arg = d2.argmin(1)
            a = va[np.arange(len(fa)), arg // 3]
            b = vb[np.arange(len(fa)), arg % 3]
            dd = d2[np.arange(len(fa)), arg]
            for x, y, dist in ((a, b, dd), (b, a, dd)):
                o = np.argsort(-dist)  # so the smallest distance is written last
                x, y, dist = x[o], y[o], dist[o]
                better = dist < best_d[x]
                best_d[x[better]] = dist[better]
                best_u[x[better]] = y[better]

    v = np.nonzero(best_u >= 0)[0]
    if len(v) == 0:
        return np.zeros((0, 2), np.int64)
    pairs = np.stack([np.minimum(v, best_u[v]), np.maximum(v, best_u[v])], axis=1)
    pairs = np.unique(pairs, axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    return pairs.astype(np.int64)
