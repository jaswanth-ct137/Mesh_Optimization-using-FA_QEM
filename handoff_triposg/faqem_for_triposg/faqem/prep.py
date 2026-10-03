"""Mesh pre-processing (paper, supplementary Sec. 6.2).

* weld vertices closer than a small tolerance (paper: absolute 1e-6; here 1e-6 x bbox
  diagonal so the tolerance is independent of the model's units),
* drop degenerate faces (repeated vertex) and duplicated faces,
* normalise the model into a unit-diagonal box centred at the origin, so the fixed
  hyper-parameters of Table 1 behave identically for millimetre CAD parts and metre scans.
"""
from dataclasses import dataclass, field

import numpy as np
from numba import njit


@dataclass
class PreparedMesh:
    positions: np.ndarray        # (n, 3) float64, normalised space
    faces: np.ndarray            # (m, 3) int64
    face_orig: np.ndarray        # (m,) prepared face -> input face (same corner order)
    input_to_vertex: np.ndarray  # (n_in,) input vertex -> prepared vertex (-1 if unused)
    center: np.ndarray           # model-space centre
    diag: float                  # model-space bbox diagonal
    removed: dict = field(default_factory=dict)

    def to_model(self, p):
        return np.asarray(p) * self.diag + self.center


def split_non_manifold(P, F):
    """Make the mesh manifold without moving anything: corners around a vertex stay on the
    same vertex only if their faces are connected through manifold (2-face) edges. Every
    other fan gets its own copy of the vertex, so non-manifold edges / bow-tie vertices
    become coincident open boundaries (which FA-QEM's boundary terms then protect)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    m = len(F)
    # directed corner edges: (face, k) owns edge (F[f,k], F[f,k+1])
    a = F.reshape(-1)
    b = F[:, [1, 2, 0]].reshape(-1)
    key = np.minimum(a, b) * len(P) + np.maximum(a, b)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    start = np.r_[0, np.nonzero(np.diff(ks))[0] + 1]
    count = np.diff(np.r_[start, len(ks)])
    two = start[count == 2]
    e1, e2 = order[two], order[two + 1]          # corner-edge ids (f*3+k) sharing a manifold edge
    f1, f2 = e1 // 3, e2 // 3
    # the two faces of a manifold edge (u, v) share their corner at u and their corner at v
    def corner_of(f, v):
        c = np.where(F[f, 0] == v, 0, np.where(F[f, 1] == v, 1, 2))
        return f * 3 + c
    u, v = a[e1], b[e1]
    src = np.concatenate([corner_of(f1, u), corner_of(f1, v)])
    dst = np.concatenate([corner_of(f2, u), corner_of(f2, v)])
    g = coo_matrix((np.ones(len(src)), (src, dst)), shape=(3 * m, 3 * m))
    n_groups, label = connected_components(g, directed=False)
    # each corner group -> one vertex (all corners of a group share the same original vertex)
    orig_v = F.reshape(-1)
    grp_vertex = np.empty(n_groups, np.int64)
    grp_vertex[label] = orig_v
    newF = label.reshape(-1, 3)
    return P[grp_vertex], newF, grp_vertex


@njit(cache=True)
def _propagate_flips(nodes, pred, pair_key, pair_par, m):
    """flip[f] = flip[pred[f]] XOR parity(f, pred[f]) in BFS order (binary search on keys)."""
    flip = np.zeros(m, np.int8)
    for i in range(1, nodes.shape[0]):
        f = nodes[i]
        p = pred[f]
        lo = min(f, p); hi = max(f, p)
        k = lo * m + hi
        j = np.searchsorted(pair_key, k)
        par = pair_par[j] if j < pair_key.shape[0] and pair_key[j] == k else 0
        flip[f] = flip[p] ^ par
    return flip


def orient_consistently(P, F):
    """Flip faces so that neighbours across every manifold edge agree on their winding.

    Per connected piece, a breadth-first search propagates flip parities; the piece then
    keeps whichever orientation the larger share of its area already had, so a correctly
    oriented mesh is never inverted. Non-orientable pieces keep the first consistent
    assignment found. Returns (new F, number of flipped faces)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import breadth_first_order, connected_components

    m = len(F)
    if m == 0:
        return F, 0
    a = F.reshape(-1)
    b = F[:, [1, 2, 0]].reshape(-1)
    key = np.minimum(a, b) * len(P) + np.maximum(a, b)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    start = np.r_[0, np.nonzero(np.diff(ks))[0] + 1]
    count = np.diff(np.r_[start, len(ks)])
    two = start[count == 2]
    e1, e2 = order[two], order[two + 1]
    f1, f2 = e1 // 3, e2 // 3
    # same traversal direction in both faces => inconsistent pair (parity 1)
    parity = ((a[e1] == a[e2]) & (b[e1] == b[e2])).astype(np.int8)
    if not parity.any():
        return F, 0
    g = coo_matrix((np.ones(len(f1)), (f1, f2)), shape=(m, m)).tocsr()
    n_comp, comp = connected_components(g, directed=False)
    pk = np.minimum(f1, f2) * m + np.maximum(f1, f2)
    o = np.argsort(pk)
    pair_key, pair_par = pk[o], parity[o]
    area = np.linalg.norm(np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]]), axis=1)
    flip = np.zeros(m, np.int8)
    # only pieces that contain an inconsistent pair need work
    bad_comps = np.unique(comp[f1[parity == 1]])
    for c in bad_comps:
        members = np.nonzero(comp == c)[0]
        seed = members[np.argmax(area[members])]
        nodes, pred = breadth_first_order(g, seed, directed=False, return_predecessors=True)
        fl = _propagate_flips(nodes.astype(np.int64), pred.astype(np.int64), pair_key, pair_par, m)
        fl = fl[nodes]
        if area[nodes][fl == 1].sum() > area[nodes][fl == 0].sum():
            fl ^= 1
        flip[nodes] = fl
    idx = np.nonzero(flip)[0]
    F = F.copy()
    F[idx] = F[idx][:, ::-1]
    return F, int(len(idx))


def prepare_mesh(positions, faces, weld_tolerance=1e-6, make_manifold=True):
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    lo, hi = positions.min(0), positions.max(0)
    center = (lo + hi) / 2
    diag = float(np.linalg.norm(hi - lo)) or 1.0
    P = (positions - center) / diag

    # --- weld: quantise onto a grid of size tol --------------------------------
    q = np.round(P / weld_tolerance).astype(np.int64)
    _, first, inverse = np.unique(q, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    welded_pos = P[first]
    F = inverse[faces]

    # --- drop degenerate + duplicate faces ----------------------------------------
    good = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
    n_degenerate = int((~good).sum())
    idx = np.nonzero(good)[0]
    Fs = np.sort(F[idx], axis=1)
    _, keep = np.unique(Fs, axis=0, return_index=True)
    keep = np.sort(keep)
    n_duplicate = len(idx) - len(keep)
    face_orig = idx[keep]
    F = F[face_orig]

    # --- drop unreferenced vertices ---------------------------------------------------
    used = np.zeros(len(welded_pos), dtype=bool)
    used[F.reshape(-1)] = True
    remap = np.full(len(welded_pos), -1, dtype=np.int64)
    remap[used] = np.arange(int(used.sum()))
    F = remap[F]
    out_pos = welded_pos[used]
    input_to_vertex = remap[inverse]

    n_split = 0
    if make_manifold:
        P2, F2, src_vertex = split_non_manifold(out_pos, F)
        n_split = len(P2) - len(out_pos)
        if n_split > 0:
            # input vertex -> first copy of its welded vertex
            first_copy = np.full(len(out_pos), -1, np.int64)
            first_copy[src_vertex[::-1]] = np.arange(len(src_vertex))[::-1]
            input_to_vertex = np.where(input_to_vertex >= 0, first_copy[input_to_vertex], -1)
            out_pos, F = P2, F2
    F, n_flipped = orient_consistently(out_pos, F)

    return PreparedMesh(
        positions=np.ascontiguousarray(out_pos),
        faces=np.ascontiguousarray(F),
        face_orig=face_orig,
        input_to_vertex=input_to_vertex,
        center=center,
        diag=diag,
        removed=dict(welded=len(positions) - len(first), degenerate=n_degenerate, duplicate=n_duplicate,
                     split_non_manifold=int(n_split), reoriented=int(n_flipped)),
    )


def unique_edges(F):
    """Unique undirected edges (e, 2) and their incident face counts."""
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=0)
    e.sort(axis=1)
    edges, counts = np.unique(e, axis=0, return_counts=True)
    return edges, counts


def mesh_stats(P, F):
    """Topology diagnostics: components, boundary and non-manifold edges."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(P)
    edges, counts = unique_edges(F)
    g = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    n_comp, _ = connected_components(g, directed=False)
    return dict(
        vertices=int(n),
        faces=int(len(F)),
        edges=int(len(edges)),
        components=int(n_comp),
        boundary_edges=int((counts == 1).sum()),
        non_manifold_edges=int((counts > 2).sum()),
    )
