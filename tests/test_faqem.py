"""pytest suite:  python -m pytest -q tests"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faqem.core import QN, _add_area_edge, _eval_q, _solve_q, simplify  # noqa: E402
from faqem.metrics import compare_meshes  # noqa: E402
from faqem.prep import mesh_stats, prepare_mesh  # noqa: E402
from tests import shapes  # noqa: E402

PLAIN = dict(w_area=0, w_boundary=0, w_normal=0, w_plane_area=0, virtual_edges=False)


def test_area_quadric_is_twice_squared_triangle_area():
    """Eq. 11: v^T Q_A v == 2 * area(v, v_r, v_s)^2 (h = 1/2)."""
    rng = np.random.default_rng(0)
    P = rng.normal(size=(2, 3))
    QA = np.zeros(QN)
    _add_area_edge(QA, P, 0, 1, 0.5)
    for _ in range(10):
        v = rng.normal(size=3)
        area = np.linalg.norm(np.cross(P[0] - v, P[1] - v)) / 2
        assert np.isclose(_eval_q(QA, *v), 2 * area ** 2)


def test_solver_exact_and_degenerate():
    # three orthogonal planes through (1, 2, 3): exact optimum
    q = np.zeros(QN)
    for n in np.eye(3):
        p = np.append(n, -n @ np.array([1.0, 2.0, 3.0]))
        q += np.outer(p, p)[np.triu_indices(4)]
    out = np.zeros(3)
    assert _solve_q(q, 0.0, 0.0, 0.0, out, 1e-3, np.zeros(9), np.zeros(9))
    assert np.allclose(out, [1, 2, 3])
    # single plane z = 0: in-plane directions are free -> stay at the midpoint
    p = np.array([0, 0, 1.0, 0])
    q = np.outer(p, p)[np.triu_indices(4)]
    assert _solve_q(q, 5.0, 6.0, 7.0, out, 1e-3, np.zeros(9), np.zeros(9))
    assert np.allclose(out, [5, 6, 0])


@pytest.mark.parametrize("name", list(shapes.ALL))
def test_reaches_target_and_stays_close(name):
    m = prepare_mesh(*shapes.ALL[name]())
    target = len(m.faces) // 10
    r = simplify(m.positions, m.faces, target)
    assert r["stats"]["output_faces"] <= target
    assert r["faces"].max() < len(r["positions"])
    c = compare_meshes(m.positions, m.faces, r["positions"], r["faces"], samples=10000)
    assert c["hausdorff"] < 0.03, c
    assert len(r["vertex_map"]) == len(m.positions)


def test_boundary_terms_protect_open_rims():
    """Paper Table 4 trend: boundary terms reduce error on open meshes."""
    m = prepare_mesh(*shapes.bowl())
    t = len(m.faces) // 10
    fa = simplify(m.positions, m.faces, t)
    pl = simplify(m.positions, m.faces, t, PLAIN)
    e_fa = compare_meshes(m.positions, m.faces, fa["positions"], fa["faces"], samples=10000)["hausdorff"]
    e_pl = compare_meshes(m.positions, m.faces, pl["positions"], pl["faces"], samples=10000)["hausdorff"]
    assert e_fa < e_pl


def test_no_new_non_manifold_edges_on_manifold_input():
    m = prepare_mesh(*shapes.detail_torus(200))
    r = simplify(m.positions, m.faces, len(m.faces) // 20, dict(topology_penalty=np.inf))
    st = mesh_stats(r["positions"], r["faces"])
    assert st["non_manifold_edges"] == 0 and st["boundary_edges"] == 0 and st["components"] == 1


def test_non_manifold_input_is_split_into_manifold_sheets():
    m = prepare_mesh(*shapes.non_manifold_fins())
    assert mesh_stats(m.positions, m.faces)["non_manifold_edges"] == 0
    assert m.removed["split_non_manifold"] > 0


# ---- mesh preparation ------------------------------------------------------------------
from faqem.prep import orient_consistently  # noqa: E402


def test_orientation_is_repaired():
    m = prepare_mesh(*shapes.cube(20))
    F = m.faces.copy()
    F[::3] = F[::3, ::-1]
    F2, n = orient_consistently(m.positions, F)
    assert n > 0 and np.array_equal(F2, m.faces)


# ---- automatic mode (face count follows from the allowed deviation) ----------------------
class _Mesh:
    """Minimal stand-in for pipeline.LoadedMesh."""
    def __init__(self, P, F):
        from faqem.bvh import BVH
        self.prep = prepare_mesh(P, F)
        self.bvh = BVH(self.prep.positions, self.prep.faces)


def test_auto_cube_becomes_12_faces():
    from faqem.pipeline import simplify_auto
    res, m = simplify_auto(_Mesh(*shapes.cube(30)), 0.001)
    assert res["stats"]["output_faces"] == 12 and m["hausdorff"] <= 0.001


def test_auto_respects_limit_and_adapts_face_count():
    from faqem.pipeline import simplify_auto
    mesh = _Mesh(*shapes.detail_torus(200))
    counts = []
    for tol in (0.005, 0.001):
        res, m = simplify_auto(mesh, tol)
        assert m["hausdorff"] <= tol and res["stats"]["auto_within"]
        counts.append(res["stats"]["output_faces"])
    assert counts[0] < counts[1]  # looser limit -> fewer faces


def test_auto_never_drops_small_parts():
    from faqem.pipeline import simplify_auto
    from faqem.virtual_edges import face_components
    mesh = _Mesh(*shapes.components())
    res, m = simplify_auto(mesh, 0.005)
    n_in = face_components(mesh.prep.faces, len(mesh.prep.positions))[0]
    n_out = face_components(res["faces"], len(res["positions"]))[0]
    assert n_out == n_in and m["hausdorff"] <= 0.005


# ---- local worst-case guarantee ------------------------------------------------------------
def test_local_guarantee_bounds_every_original_vertex():
    from faqem.bvh import BVH
    from faqem.metrics import point_to_mesh_d2
    for P, F in (shapes.noisy_sphere(), shapes.box_with_knob(), shapes.bowl()):
        pr = prepare_mesh(P, F)
        tol = 0.002
        res = simplify(pr.positions, pr.faces, 1, dict(auto=True, max_error=tol, local_tol=tol))
        d = np.sqrt(point_to_mesh_d2(pr.positions, BVH(res["positions"], res["faces"])))
        assert d.max() <= tol * 1.0001
        assert len(res["faces"]) < len(pr.faces) / 2


def test_auto_simplifies_smooth_parts_despite_one_noisy_spot():
    """A single rough patch must not keep the rest of the mesh dense (the old global search did)."""
    from faqem.pipeline import simplify_auto
    P, F = shapes.sphere(160)
    P = P.copy()
    rng = np.random.default_rng(1)
    spot = P[:, 1] > 0.9 * P[:, 1].max()                      # a small cap near the pole
    P[spot] += rng.normal(scale=0.004, size=(spot.sum(), 3))
    mesh = _Mesh(P, F)
    res, m = simplify_auto(mesh, 0.002)
    assert res["stats"]["auto_within"] and m["hausdorff"] <= 0.002
    assert res["stats"]["output_faces"] < 0.25 * len(mesh.prep.faces)


# ---- best result for a face count (Percentage / Face count targets) ------------------------
def test_best_for_count_reaches_target_and_beats_plain():
    from faqem.pipeline import simplify_to_count
    mesh = _Mesh(*shapes.detail_torus(160))
    n = 2000
    res, m = simplify_to_count(mesh, n)
    assert n - 2 <= len(res["faces"]) <= n
    plain = simplify(mesh.prep.positions, mesh.prep.faces, n)
    mp = compare_meshes(mesh.prep.positions, mesh.prep.faces, plain["positions"], plain["faces"], bvh_a=mesh.bvh)
    assert m["hausdorff"] <= mp["hausdorff"] * 1.0001
    assert res["stats"]["best_for_count"] and abs(res["stats"]["plain_hausdorff"] - mp["hausdorff"]) < 1e-12


def test_fast_is_plain_fa_qem(tmp_path):
    import trimesh
    from faqem.pipeline import LoadedMesh, run
    P, F = shapes.sphere(60)
    path = str(tmp_path / "s.obj")
    trimesh.Trimesh(P, F, process=False).export(path)
    mesh = LoadedMesh(path)
    out = run(mesh, 800, str(tmp_path / "o"), bake_color=False, bake_normal=False, fast=True)
    plain = simplify(mesh.prep.positions, mesh.prep.faces, 800)
    assert out["stats"]["output_faces"] == len(plain["faces"]) and not out["stats"].get("best_for_count")
