"""Bit-exact parity of the C++ engine (faqem_cpp) with the Python reference (faqem).

    .venv/bin/python -m pytest -q tests/test_cpp_parity.py

Every comparison is exact: float arrays are compared bit for bit (including the sign of zero).
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# the Python-parity build of the C++ engine (cmake -DFAQEM_PYTHON_PARITY=ON -B cpp/build-parity)
sys.path.insert(0, os.path.join(ROOT, "cpp", "build-parity"))

faqem_cpp = pytest.importorskip("faqem_cpp_parity")
if getattr(faqem_cpp, "BUILD_MODE", "") != "python-parity":
    pytest.skip("needs the python-parity build (cpp/build-parity)", allow_module_level=True)

from faqem import core, metrics, pipeline, prep  # noqa: E402
from faqem.bvh import BVH  # noqa: E402
from tests import shapes  # noqa: E402

C = faqem_cpp._compat


def same(a, b):
    """Bitwise equality of two arrays (float: identical bits)."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind == "f" or b.dtype.kind == "f":
        a = np.ascontiguousarray(a, np.float64).view(np.uint64)
        b = np.ascontiguousarray(b, np.float64).view(np.uint64)
    return bool(np.array_equal(a, b))


def first_diff(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if a.shape != b.shape:
        return f"shape {a.shape} vs {b.shape}"
    d = np.nonzero(~((a == b) & (np.signbit(a) == np.signbit(b)) if a.dtype.kind == "f" else (a == b)))[0]
    return f"{len(d)} differ, first at {d[0]}: {a[d[0]]!r} vs {b[d[0]]!r}" if len(d) else "equal"


def assert_same(a, b, what):
    assert same(a, b), f"{what}: {first_diff(a, b)}"


# ---------------------------------------------------------------------------------------------
# compat layer
# ---------------------------------------------------------------------------------------------
def test_pairwise_sum():
    rng = np.random.default_rng(3)
    for n in [0, 1, 7, 8, 9, 127, 128, 129, 1000, 8191, 8192, 8193, 20000, 100001]:
        a = rng.normal(size=n) * 10.0 ** rng.integers(-8, 8)
        assert C.sum(a) == np.sum(a), n


def test_argsort_ties():
    rng = np.random.default_rng(4)
    for n in [1, 2, 5, 16, 17, 40, 100, 1000, 5000]:
        for hi in (3, 10, n + 1):
            a = rng.integers(0, hi, size=n)
            assert_same(C.argsort_i64(a), np.argsort(a), f"int n={n}")
            f = rng.integers(0, hi, size=n).astype(np.float64)
            assert_same(C.argsort_f64(f), np.argsort(f), f"float n={n}")
    # adversarial (heapsort fallback)
    a = np.tile(np.arange(64), 64)[::-1].copy()
    assert_same(C.argsort_i64(a), np.argsort(a), "tiled")


def test_rng():
    for seed in (0, 7, 11, 12345, 2 ** 40 + 3):
        g = np.random.default_rng(seed)
        assert_same(C.rng_random(seed, 1000), g.random(1000), f"random {seed}")
        for n in (1, 2, 10, 1000, 70000):
            assert_same(C.rng_permutation(seed, n), np.random.default_rng(seed).permutation(n), f"perm {n}")
        p = np.random.default_rng(seed + 1).random(500)
        p = p / p.sum()
        assert_same(C.rng_choice(seed, 3000, p), np.random.default_rng(seed).choice(500, size=3000, p=p), "choice")


def test_linspace_pow():
    for a, b in [(100, 1), (627666, 1), (5, 5), (12345, 4000)]:
        assert_same(C.linspace(a, b, 40), np.linspace(a, b, 40), f"linspace {a} {b}")
    for x in np.random.default_rng(0).random(2000) * 3:
        assert C.py_pow(float(x), 2.0) == float(x) ** 2


# ---------------------------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------------------------
SHAPES = list(shapes.ALL)


def _prep_both(P, F):
    a = prep.prepare_mesh(P, F)
    b = faqem_cpp.prepare_mesh(P, F)
    return a, b


@pytest.mark.parametrize("name", SHAPES)
def test_prepare_mesh(name):
    P, F = shapes.ALL[name]()
    a, b = _prep_both(P, F)
    assert_same(b.positions, a.positions, "positions")
    assert_same(b.faces, a.faces, "faces")
    assert_same(b.face_orig, a.face_orig, "face_orig")
    assert_same(b.input_to_vertex, a.input_to_vertex, "input_to_vertex")
    assert_same(b.center, a.center, "center")
    assert b.diag == a.diag
    assert dict(b.removed) == a.removed


@pytest.mark.parametrize("name", SHAPES)
@pytest.mark.parametrize("preset", ["Recommended", "Plain QEM (ablation baseline)"])
def test_quadrics(name, preset):
    m = prep.prepare_mesh(*shapes.ALL[name]())
    o = dict(core.DEFAULTS, **core.PRESETS[preset])
    edges, counts = prep.unique_edges(m.faces)
    ref = core.build_quadrics(m.positions, m.faces, edges, counts, o)
    got = faqem_cpp.build_quadrics(m.positions, m.faces, {k: v for k, v in o.items()})
    for i, what in enumerate(["Q", "mean_area", "mean_edge", "n_bverts", "Qgeo", "Ageo"]):
        assert_same(got[i], ref[i], what)


def _check_result(got, ref):
    assert_same(got["faces"], ref["faces"], "faces")
    assert_same(got["positions"], ref["positions"], "positions")
    assert_same(got["vertex_map"], ref["vertex_map"], "vertex_map")
    assert_same(got["history"], ref["history"], "history")
    for k in ("output_faces", "collapses", "rejected_flip", "rejected_link", "rejected_duplicate",
              "rejected_error", "stale", "rebuilds", "boundary_vertices"):
        assert got["stats"][k] == ref["stats"][k], k


@pytest.mark.parametrize("name", SHAPES)
@pytest.mark.parametrize("ratio", [0.1, 0.02])
def test_simplify_target(name, ratio):
    m = prep.prepare_mesh(*shapes.ALL[name]())
    n = max(4, int(len(m.faces) * ratio))
    _check_result(faqem_cpp.simplify(m.positions, m.faces, n), core.simplify(m.positions, m.faces, n))


@pytest.mark.parametrize("name", SHAPES)
def test_simplify_local_tol(name):
    m = prep.prepare_mesh(*shapes.ALL[name]())
    cm = faqem_cpp.prepare_mesh(*shapes.ALL[name]())
    n = max(4, len(m.faces) // 20)
    o = dict(local_tol=0.002, max_error=0.002)
    _check_result(faqem_cpp.simplify(m.positions, m.faces, n, o, cm),
                  core.simplify(m.positions, m.faces, n, o, reference=BVH(m.positions, m.faces)))
    o = dict(auto=True, local_tol=0.001, max_error=0.001)
    _check_result(faqem_cpp.simplify(m.positions, m.faces, 1, o, cm),
                  core.simplify(m.positions, m.faces, 1, o, reference=BVH(m.positions, m.faces)))


@pytest.mark.parametrize("name", SHAPES)
def test_metrics(name):
    m = prep.prepare_mesh(*shapes.ALL[name]())
    r = core.simplify(m.positions, m.faces, max(4, len(m.faces) // 10))
    assert_same(faqem_cpp.sample_surface(m.positions, m.faces, 5000, 7),
                metrics.sample_surface(m.positions, m.faces, 5000, 7), "samples")
    a = metrics.compare_meshes(m.positions, m.faces, r["positions"], r["faces"])
    b = faqem_cpp.compare_meshes(m.positions, m.faces, r["positions"], r["faces"])
    assert a == b


class _Mesh:
    def __init__(self, P, F):
        self.prep = prep.prepare_mesh(P, F)
        self.bvh = BVH(self.prep.positions, self.prep.faces)


@pytest.mark.parametrize("name", SHAPES)
@pytest.mark.parametrize("level", ["Low", "High"])
def test_simplify_auto(name, level):
    P, F = shapes.ALL[name]()
    ref, mref = pipeline.simplify_auto(_Mesh(P, F), pipeline.DETAIL_LEVELS[level])
    got, mgot = faqem_cpp.simplify_auto(faqem_cpp.prepare_mesh(P, F), pipeline.DETAIL_LEVELS[level])
    assert got["stats"]["auto_passes"] == ref["stats"]["auto_passes"]
    assert mgot == mref
    _check_result(got, ref)


@pytest.mark.parametrize("name", ["sphere", "bowl", "noisy_sphere", "rounded_box"])
def test_simplify_to_count(name):
    P, F = shapes.ALL[name]()
    mesh = _Mesh(P, F)
    n = len(mesh.prep.faces) // 25
    ref, mref = pipeline.simplify_to_count(mesh, n)
    got, mgot = faqem_cpp.simplify_to_count(faqem_cpp.prepare_mesh(P, F), n)
    assert got["stats"]["count_passes"] == ref["stats"]["count_passes"]
    assert mgot == mref
    _check_result(got, ref)


# ---------------------------------------------------------------------------------------------
# Accelerate-backed numpy calls used by the trimesh loader emulation
# ---------------------------------------------------------------------------------------------
def test_accelerate_wrappers():
    from trimesh import transformations as tf
    rng = np.random.default_rng(9)
    for n in [1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 100, 1001, 50000]:
        for forder in (False, True):
            V = rng.normal(size=(n, 3)).astype(np.float32).astype(np.float64)
            T = rng.normal(size=(4, 4))
            Tn = np.asfortranarray(T) if forder else T
            ref = V @ Tn[:3, :3].T + Tn[:3, 3]
            assert_same(C.transform_points(V, T, forder), ref, f"transform n={n} f={forder}")
    for _ in range(200):
        A, B = rng.normal(size=(4, 4)), rng.normal(size=(4, 4))
        for fa in (False, True):
            for fb in (False, True):
                An = np.asfortranarray(A) if fa else A
                Bn = np.asfortranarray(B) if fb else B
                assert_same(C.dot4(A, fa, B, fb), np.dot(An, Bn), "dot4")
        q = rng.normal(size=4)
        assert_same(C.quaternion_matrix(q), tf.quaternion_matrix(q), "quaternion")
        assert C.det3_negative(A) == (np.linalg.det(A[:3, :3]) < 0)
        # near-rigid matrices (float32-rounded rotations) trigger the SVD repair
        R = tf.quaternion_matrix(q).astype(np.float32).astype(np.float64)
        R[:3, 3] = rng.normal(size=3)
        for forder in (False, True):
            Rn = np.asfortranarray(R) if forder else R
            assert_same(C.fix_rigid(R, forder), tf.fix_rigid(Rn), "fix_rigid")


# ---------------------------------------------------------------------------------------------
# loaders: faqem_cpp.load_asset == faqem.io.load_asset
# ---------------------------------------------------------------------------------------------
import glob  # noqa: E402

DL = os.path.expanduser("~/Downloads")
CORPUS = sorted(set(
    glob.glob(os.path.join(ROOT, "checking copy", "*.glb"))
    + glob.glob(os.path.join(ROOT, "*.glb"))
    + glob.glob(os.path.join(ROOT, "compare", "**", "*.glb"), recursive=True)
    + glob.glob(os.path.join(ROOT, "samples", "**", "*.glb"), recursive=True)
    + glob.glob(os.path.join(ROOT, "samples", "**", "*.obj"), recursive=True)
    + glob.glob(os.path.join(DL, "*.stl")) + glob.glob(os.path.join(DL, "*.glb"))
    + glob.glob(os.path.join(DL, "Hunyuan3D-2", "*.glb"))
))


def _assert_asset(got, ref, what):
    from faqem import io as fio  # noqa: F401
    assert_same(got["positions"], ref.positions, f"{what} positions")
    assert_same(got["faces"], ref.faces, f"{what} faces")
    assert_same(got["face_material"], ref.face_material, f"{what} face_material")
    for k in ("corner_uv", "corner_color"):
        r = getattr(ref, k)
        assert (got[k] is None) == (r is None), f"{what} {k} presence"
        if r is not None:
            assert_same(got[k], r, f"{what} {k}")
    assert len(got["materials"]) == len(ref.materials), what
    for gm, rm in zip(got["materials"], ref.materials):
        assert_same(gm["color"], rm.color, f"{what} material color")
        for k in ("texture", "mr_texture"):
            r = getattr(rm, k)
            assert (gm[k] is None) == (r is None), f"{what} {k} presence"
            if r is not None:
                assert_same(gm[k], r, f"{what} {k} pixels")
        assert gm["metallic"] == rm.metallic and gm["roughness"] == rm.roughness, what
        assert gm["alpha_mode"] == rm.alpha_mode and gm["double_sided"] == rm.double_sided, what
    assert got["has_texture"] == ref.has_texture
    assert got["has_appearance"] == ref.has_appearance
    assert got["has_mr_variation"] == ref.has_mr_variation


@pytest.mark.parametrize("path", CORPUS, ids=[os.path.basename(p) for p in CORPUS])
def test_load_asset_corpus(path):
    from faqem.io import load_asset
    try:
        ref = load_asset(path)
    except ValueError as e:  # e.g. a lines-only GLB: both backends must refuse it
        with pytest.raises(RuntimeError, match="No triangle mesh"):
            faqem_cpp.load_asset(path)
        assert "No triangle mesh" in str(e)
        return
    _assert_asset(faqem_cpp.load_asset(path), ref, os.path.basename(path))


def _synthetic_files(d):
    """Small OBJ / PLY / OFF / STL files exercising the loaders' quirks."""
    import trimesh
    from PIL import Image
    files = []
    P, F = shapes.sphere(12)
    P = np.asarray(P, np.float64)
    F = np.asarray(F)
    m = trimesh.Trimesh(P, F, process=False)
    for name, kw in [("bin.stl", {}), ("ascii.stl", dict(file_type="stl_ascii"))]:
        p = os.path.join(d, name)
        data = m.export(file_type=kw.get("file_type", "stl"))
        open(p, "wb").write(data if isinstance(data, bytes) else data.encode())
        files.append(p)
    # OFF with comments, quads and a pentagon
    p = os.path.join(d, "mixed.off")
    open(p, "w").write("OFF\n# a comment\n6 4 0\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n0.5 2 0.25\n0.5 0.5 1\n"
                       "3 0 1 5\n4 0 1 2 3\n4 0 1 2 4 # tail\n3 1 2 5\n")
    files.append(p)
    p = os.path.join(d, "penta.off")
    open(p, "w").write("OFF\n6 2 0\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n0.5 2 0.25\n0.5 0.5 1\n"
                       "5 0 1 2 4 3\n5 5 1 2 4 0\n")
    files.append(p)
    p = os.path.join(d, "quads.off")
    open(p, "w").write("OFF\n4 2 0\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n4 0 1 2 3\n4 3 2 1 0\n")
    files.append(p)
    # PLY ascii / binary with colours, and quads
    for enc in ("ascii", "binary_little_endian"):
        p = os.path.join(d, f"col_{enc}.ply")
        mm = trimesh.Trimesh(P, F, process=False,
                             vertex_colors=(np.random.default_rng(0).random((len(P), 4)) * 255).astype(np.uint8))
        open(p, "wb").write(mm.export(file_type="ply", encoding="ascii" if enc == "ascii" else "binary"))
        files.append(p)
    p = os.path.join(d, "quad.ply")
    open(p, "w").write("ply\nformat ascii 1.0\nelement vertex 5\nproperty float x\nproperty float y\nproperty float z\n"
                       "element face 3\nproperty list uchar int vertex_indices\nend_header\n"
                       "0 0 0\n1 0 0\n1 1 0.1\n0 1 0.2\n0.3 0.7 1\n4 0 1 2 3\n3 0 1 4\n3 1 2 4\n")
    files.append(p)
    # OBJ: v/vt/vn, quads, two materials (one textured), negative indices, vertex colours
    Image.fromarray((np.random.default_rng(1).random((8, 8, 3)) * 255).astype(np.uint8)).save(os.path.join(d, "tex.png"))
    open(os.path.join(d, "m.mtl"), "w").write("newmtl red\nKd 0.8 0.1 0.1\nNs 20\nnewmtl tex\nKd 1 1 1\nmap_Kd tex.png\n")
    rng = np.random.default_rng(2)
    lines = ["mtllib m.mtl"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in P]
    lines += [f"vt {u:.5f} {v:.5f}" for u, v in rng.random((len(P), 2))]
    lines += [f"vn {x:.5f} {y:.5f} {z:.5f}" for x, y, z in P / np.linalg.norm(P, axis=1, keepdims=True)]
    half = len(F) // 2
    lines.append("usemtl red")
    lines += [f"f {a + 1}/{a + 1}/{a + 1} {b + 1}/{b + 1}/{b + 1} {c + 1}/{c + 1}/{c + 1}" for a, b, c in F[:half]]
    lines.append("usemtl tex")
    lines += [f"f {a + 1}/{(a * 7) % len(P) + 1} {b + 1}/{(b * 7) % len(P) + 1} {c + 1}/{(c * 7) % len(P) + 1}"
              for a, b, c in F[half:]]
    p = os.path.join(d, "mat.obj")
    open(p, "w").write("\n".join(lines) + "\n")
    files.append(p)
    lines = [f"v {x:.6f} {y:.6f} {z:.6f} {abs(x):.3f} {abs(y):.3f} {abs(z):.3f}" for x, y, z in P]
    lines += ["f 1 2 3 4", "f -1 -2 -3", "f 5 6 7"] + [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in F[:20]]
    p = os.path.join(d, "colors_mixed.obj")
    open(p, "w").write("\n".join(lines) + "\n")
    files.append(p)
    return files


def test_load_asset_synthetic(tmp_path):
    from faqem.io import load_asset
    for p in _synthetic_files(str(tmp_path)):
        _assert_asset(faqem_cpp.load_asset(p), load_asset(p), os.path.basename(p))
    # a file the Python reference cannot load must fail in C++ as well
    p = os.path.join(str(tmp_path), "ragged.off")
    open(p, "w").write("OFF\n6 2 0\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n0.5 2 0.25\n0.5 0.5 1\n3 0 1 5\n5 0 1 2 4 3\n")
    with pytest.raises(TypeError):
        load_asset(p)
    with pytest.raises(RuntimeError):
        faqem_cpp.load_asset(p)


# ---------------------------------------------------------------------------------------------
# stage 2: normals, bake, exports, full run()
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", SHAPES)
def test_trimesh_vertex_normals(name):
    import trimesh
    m = prep.prepare_mesh(*shapes.ALL[name]())
    r = core.simplify(m.positions, m.faces, max(4, len(m.faces) // 10))
    P = r["positions"] * 3.7 + 0.25
    ref = trimesh.Trimesh(P, r["faces"], process=False).vertex_normals
    assert_same(faqem_cpp.trimesh_vertex_normals(P, r["faces"]), ref, "vertex_normals")


BAKE_SKUS = [p for p in [
    os.path.join(ROOT, "checking copy", "buwch_Boots (1).glb"),
    os.path.join(ROOT, "checking copy", "Versa AI Model.glb"),
    os.path.join(ROOT, "tripo3-wooden+dresser+3d+model.glb.glb"),
    os.path.join(ROOT, "checking copy", "2026-0806-084116-872806-2026-0806-084114-570258.glb"),
] if os.path.exists(p)]


@pytest.mark.parametrize("path", BAKE_SKUS, ids=[os.path.basename(p) for p in BAKE_SKUS])
def test_bake(path):
    from faqem.bake import bake
    from faqem.pipeline import LoadedMesh
    pm = LoadedMesh(path)
    cm = faqem_cpp.LoadedMesh(path)
    r = core.simplify(pm.prep.positions, pm.prep.faces, 1500)
    ref = bake(r["positions"], r["faces"], pm.prep, pm.asset, r["vertex_map"], orig_bvh=pm.bvh)
    got = cm.bake(r["positions"], r["faces"], r["vertex_map"])
    for k in ("positions", "normals", "uv"):
        assert_same(got[k], ref[k], k)
    for k in ("color", "normal_map", "mr"):
        assert (got[k] is None) == (ref[k] is None), k
        if ref[k] is not None:
            assert_same(got[k], ref[k], f"{k} pixels")
    assert got["atlas_size"] == ref["atlas_size"] and got["texels_per_unit"] == ref["texels_per_unit"]
    assert tuple(got["base_color"]) == tuple(ref["base_color"])


def _glb_arrays(path):
    """Decoded accessor arrays + material values + decoded images of a GLB (layout-independent)."""
    import io as _io
    import json
    import struct
    from PIL import Image
    data = open(path, "rb").read()
    off, js, binary = 12, None, b""
    while off < len(data):
        ln, typ = struct.unpack_from("<II", data, off)
        chunk = data[off + 8: off + 8 + ln]
        if typ == 0x4E4F534A:
            js = json.loads(chunk)
        else:
            binary = chunk
        off += 8 + ln
    dt = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}
    nc = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}

    def acc(i):
        a = js["accessors"][i]
        v = js["bufferViews"][a["bufferView"]]
        n = a["count"] * nc[a["type"]]
        return np.frombuffer(binary, dt[a["componentType"]], n, v.get("byteOffset", 0) + a.get("byteOffset", 0))

    out = []
    for mesh in js["meshes"]:
        for p in mesh["primitives"]:
            item = {k: acc(v) for k, v in p["attributes"].items()}
            item["indices"] = acc(p["indices"]) if "indices" in p else None
            item["mode"] = p.get("mode", 4)
            if "material" in p:
                mat = dict(js["materials"][p["material"]])
                for key in ("baseColorTexture", "metallicRoughnessTexture"):
                    t = mat.get("pbrMetallicRoughness", {}).pop(key, None)
                    if t is not None:
                        img = js["images"][js["textures"][t["index"]]["source"]]
                        v = js["bufferViews"][img["bufferView"]]
                        b = binary[v.get("byteOffset", 0): v.get("byteOffset", 0) + v["byteLength"]]
                        item[key] = np.asarray(Image.open(_io.BytesIO(b)))
                t = mat.pop("normalTexture", None)
                if t is not None:
                    img = js["images"][js["textures"][t["index"]]["source"]]
                    v = js["bufferViews"][img["bufferView"]]
                    b = binary[v.get("byteOffset", 0): v.get("byteOffset", 0) + v["byteLength"]]
                    item["normalTexture"] = np.asarray(Image.open(_io.BytesIO(b)))
                mat.pop("name", None)
                item["material"] = mat
            out.append(item)
    return out


def _assert_glb(got_path, ref_path):
    g, r = _glb_arrays(got_path), _glb_arrays(ref_path)
    assert len(g) == len(r), (got_path, len(g), len(r))
    for a, b in zip(g, r):
        assert set(a) == set(b), (set(a) ^ set(b))
        for k in a:
            if k in ("material", "mode"):
                assert a[k] == b[k], (k, a[k], b[k])
            elif a[k] is None or b[k] is None:
                assert a[k] is None and b[k] is None, k
            else:
                assert_same(a[k], b[k], f"{os.path.basename(got_path)} {k}")


RUN_CASES = [
    (os.path.join(ROOT, "checking copy", "buwch_Boots (1).glb"), dict(target_faces=3000, fast=True)),
    (os.path.join(ROOT, "checking copy", "Versa AI Model.glb"), dict(target_faces=2000, fast=True)),
    (os.path.join(ROOT, "tripo3-wooden+dresser+3d+model.glb.glb"), dict(target_faces=None, auto_tolerance=0.005)),
]


@pytest.mark.parametrize("path,kw", RUN_CASES, ids=[os.path.basename(p) for p, _ in RUN_CASES])
def test_run_outputs(path, kw, tmp_path):
    from faqem.pipeline import LoadedMesh, run
    pdir, cdir = str(tmp_path / "py"), str(tmp_path / "cpp")
    ref = run(LoadedMesh(path), kw["target_faces"], pdir, auto_tolerance=kw.get("auto_tolerance"),
              fast=kw.get("fast", False))
    got = faqem_cpp.LoadedMesh(path).run(kw["target_faces"], cdir, auto_tolerance=kw.get("auto_tolerance"),
                                          fast=kw.get("fast", False))
    assert got["stats"]["output_faces"] == ref["stats"]["output_faces"]
    assert got["metrics"] == ref["metrics"]
    assert open(got["geometry_obj"], "rb").read() == open(ref["geometry_obj"], "rb").read()
    for k in ("geometry_glb", "textured_glb", "clay_wire_glb", "lines_glb", "textured_wire_glb"):
        assert (got.get(k) is None) == (ref.get(k) is None), k
        if ref.get(k):
            _assert_glb(got[k], ref[k])


# ---------------------------------------------------------------------------------------------
# virtual edges (scipy cKDTree) and the Paper Table 1 preset
# ---------------------------------------------------------------------------------------------
VE_MESHES = [p for p in [
    os.path.join(ROOT, "checking copy", "buwch_Boots (1).glb"),
    os.path.join(ROOT, "tripo3-wooden+dresser+3d+model.glb.glb"),
    os.path.join(ROOT, "checking copy", "02_textured_improved (10).glb"),
] if os.path.exists(p)]


@pytest.mark.parametrize("src", ["components", "non_manifold_fins", "box_with_knob"] + VE_MESHES)
def test_virtual_edges(src):
    from faqem.io import load_asset
    from faqem.virtual_edges import find_virtual_edges
    if src in shapes.ALL:
        P, F = shapes.ALL[src]()
    else:
        a = load_asset(src)
        P, F = a.positions, a.faces
    m = prep.prepare_mesh(P, F)
    for tau in (0.01, 0.03):
        assert_same(C.find_virtual_edges(m.positions, m.faces, tau), find_virtual_edges(m.positions, m.faces, tau),
                    f"pairs tau={tau}")
    o = dict(core.PRESETS["Paper Table 1 (exact)"])
    n = max(4, len(m.faces) // 20)
    _check_result(faqem_cpp.simplify(m.positions, m.faces, n, o), core.simplify(m.positions, m.faces, n, o))
