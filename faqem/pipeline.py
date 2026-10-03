"""End-to-end FA-QEM pipeline: load -> prepare -> simplify (stage 1) -> bake (stage 2) -> export."""
import os
import time

import numpy as np
import trimesh

from .bake import bake
from .bvh import BVH
from .core import simplify
from .io import export_geometry, export_textured, load_asset
from .metrics import compare_meshes
from .prep import mesh_stats, prepare_mesh
from .wireframe import add_wireframe


class LoadedMesh:
    """An uploaded asset prepared once, so it can be simplified repeatedly."""

    def __init__(self, path):
        t = time.perf_counter()
        self.path = path
        self.asset = load_asset(path)
        self.prep = prepare_mesh(self.asset.positions, self.asset.faces)
        self.stats = mesh_stats(self.prep.positions, self.prep.faces)
        self._bvh = None
        self.load_time = time.perf_counter() - t

    @property
    def bvh(self):
        if self._bvh is None:
            self._bvh = BVH(self.prep.positions, self.prep.faces)
        return self._bvh


DETAIL_LEVELS = {  # Automatic mode: allowed worst-case deviation, fraction of the bbox diagonal
    "Low": 0.005, "Medium": 0.002, "High": 0.001, "Ultra": 0.0005,
}


def simplify_auto(mesh, tolerance, options=None, progress=None, max_passes=4):
    """Automatic face count with a guaranteed worst-case deviation.

    Every collapse is checked locally (core.py: local_tol): the original points around it must
    stay within the tolerance of the new faces, and the new faces within the tolerance of the
    original. So each area is simplified exactly as far as its own shape allows - one noisy
    spot no longer decides the face count of the whole mesh. The real worst-case (Hausdorff)
    deviation is then measured; points between the tracked ones can exceed the local bound a
    little, so the local tolerance starts at 80 % of the limit and is adjusted (usually 1-2
    passes). The smallest result within `tolerance` (x bbox diagonal) is kept."""
    t0 = time.perf_counter()
    prep = mesh.prep
    o = dict(options or {}, auto=True)
    passes, best_ok, best_any = [], None, None
    f, f_ok, f_bad = 0.8, None, None                # local factor; largest passing / smallest failing
    for k in range(max_passes):
        o["local_tol"] = f * tolerance
        o["max_error"] = f * tolerance
        res = simplify(prep.positions, prep.faces, 1, o, reference=mesh.bvh,
                       progress=(lambda p, n, k=len(passes): progress(k, p, n)) if progress else None)
        m = compare_meshes(prep.positions, prep.faces, res["positions"], res["faces"], bvh_a=mesh.bvh)
        faces = res["stats"]["output_faces"]
        H = m["hausdorff"]
        ok = H <= tolerance
        passes.append(dict(factor=f, faces=faces, hausdorff=H, ok=ok))
        if ok and (best_ok is None or faces < best_ok[0]["stats"]["output_faces"]):
            best_ok = (res, m)
        if best_any is None or H < best_any[1]["hausdorff"]:
            best_any = (res, m)
        if ok:
            f_ok = f if f_ok is None else max(f_ok, f)
            if H > 0.85 * tolerance or f >= 1.0:
                break                               # the allowance is used
        else:
            f_bad = f if f_bad is None else min(f_bad, f)
        if f_ok is not None and f_bad is not None:
            if f_bad / f_ok < 1.12:
                break                               # bracket tight enough
            f = float(np.sqrt(f_ok * f_bad))        # bisect (the response is not smooth)
        elif ok:
            f = min(1.0, f * min(1.25, 0.97 * tolerance / max(H, 1e-12)))
        else:
            f *= max(0.5, 0.95 * tolerance / H)
    res, m = best_ok if best_ok is not None else best_any
    res["stats"].update(auto=True, auto_tolerance=tolerance, auto_passes=passes,
                        auto_within=best_ok is not None, time_total=time.perf_counter() - t0)
    return res, m


def simplify_to_count(mesh, target_faces, options=None, progress=None, max_passes=6):
    """Best result for a face count: the Automatic logic aimed at `target_faces`.

    Plain FA-QEM to the target is the first pass (and the fallback, so this is never worse).
    Then the local worst-case check (core.py: local_tol) is switched on and its tolerance tau is
    searched: a pass that still reaches the target with a smaller tau spends the faces where the
    shape needs them, so its worst deviation drops. tau starts at 60 % of the plain result's
    worst deviation, halves while the target is reached, grows while it is not, then bisects.
    The reached result with the smallest measured worst (Hausdorff) deviation is kept."""
    t0 = time.perf_counter()
    prep = mesh.prep
    P, F = prep.positions, prep.faces
    n = int(target_faces)
    res0 = simplify(P, F, n, options,
                    progress=(lambda p, k: progress(0, p, k)) if progress else None)
    m0 = compare_meshes(P, F, res0["positions"], res0["faces"], bvh_a=mesh.bvh)
    best = (res0, m0)
    passes = [dict(tau=0.0, faces=res0["stats"]["output_faces"], hausdorff=m0["hausdorff"], reached=True)]
    tau, lo, hi = 0.6 * m0["hausdorff"], None, None   # lo: largest failing tau, hi: smallest passing
    for k in range(max_passes):
        if tau <= 0:
            break
        o = dict(options or {}, local_tol=tau, max_error=tau)
        res = simplify(P, F, n, o, reference=mesh.bvh,
                       progress=(lambda p, c, k=k: progress(k + 1, p, c)) if progress else None)
        reached = res["stats"]["output_faces"] <= n
        m = None
        if reached:
            m = compare_meshes(P, F, res["positions"], res["faces"], bvh_a=mesh.bvh)
            if m["hausdorff"] < best[1]["hausdorff"]:
                best = (res, m)
            hi = tau if hi is None else min(hi, tau)
        else:
            lo = tau if lo is None else max(lo, tau)
        passes.append(dict(tau=tau, faces=res["stats"]["output_faces"],
                           hausdorff=m["hausdorff"] if m else None, reached=reached))
        if lo is not None and hi is not None:
            if hi / lo < 1.15:
                break                                    # bracket tight enough
            tau = float(np.sqrt(lo * hi))
        elif reached:
            tau *= 0.5
        else:
            tau *= 1.6
    res, m = best
    res["stats"].update(best_for_count=True, count_passes=passes, plain_hausdorff=m0["hausdorff"],
                        time_total=time.perf_counter() - t0)
    return res, m


def run(mesh, target_faces, out_dir, options=None, bake_color=True, bake_normal=True,
        atlas_size=None, crease_angle=60.0, compute_metrics=True, progress=None, auto_tolerance=None,
        fast=False):
    """target_faces=None together with auto_tolerance (fraction of the bbox diagonal) selects
    the Automatic mode: the face count follows from the allowed deviation. With a face target the
    best result for that count is searched (simplify_to_count); fast=True is one plain FA-QEM pass."""
    if isinstance(mesh, str):
        mesh = LoadedMesh(mesh)
    os.makedirs(out_dir, exist_ok=True)
    prep, asset = mesh.prep, mesh.asset
    report = lambda f, msg: progress(f, msg) if progress else None

    auto_metrics = None
    if auto_tolerance:
        report(0.02, "Automatic: simplifying while keeping detail within the limit")
        res, auto_metrics = simplify_auto(
            mesh, float(auto_tolerance), options,
            progress=lambda k, p, n: report(0.05 + 0.1 * k + 0.08 * p, f"Automatic pass {k + 1}: {n:,} faces"))
    elif not fast:
        target_faces = int(np.clip(target_faces, 4, len(prep.faces)))
        report(0.02, "Best result for this face count: searching")
        res, auto_metrics = simplify_to_count(
            mesh, target_faces, options,
            progress=lambda k, p, n: report(0.05 + 0.09 * k + 0.08 * p, f"Pass {k + 1}: {n:,} faces"))
    else:
        target_faces = int(np.clip(target_faces, 4, len(prep.faces)))
        report(0.02, "Simplifying (FA-QEM)")
        res = simplify(prep.positions, prep.faces, target_faces, options,
                       progress=lambda p, n: report(0.05 + 0.6 * p, f"Simplifying: {n:,} faces"))
    S, SF = res["positions"], res["faces"]
    st = res["stats"]
    base = os.path.splitext(os.path.basename(mesh.path))[0]
    out = dict(stats=st, input=mesh.stats)

    # geometry-only low-poly (shared vertices, crease-aware smooth normals)
    Sm = prep.to_model(S)
    tm = trimesh.Trimesh(Sm, SF, process=False)
    geo_obj = export_geometry(os.path.join(out_dir, f"{base}_lowpoly.obj"), Sm, SF)
    geo_glb = export_geometry(os.path.join(out_dir, f"{base}_lowpoly_geometry.glb"), Sm, SF, tm.vertex_normals)
    out.update(geometry_obj=geo_obj, geometry_glb=geo_glb)

    # stage 2: appearance transfer via successive mapping
    textured = None
    t = time.perf_counter()
    if bake_color or bake_normal:
        report(0.7, "Baking appearance (successive mapping)")
        b = bake(S, SF, prep, asset, res["vertex_map"], atlas_size=atlas_size,
                 bake_color=bake_color, bake_normal=bake_normal, crease_angle=crease_angle,
                 orig_bvh=mesh.bvh)
        mat0 = asset.materials[0]
        textured = export_textured(
            os.path.join(out_dir, f"{base}_lowpoly_textured.glb"), prep.to_model(b["positions"]),
            b["normals"], b["uv"], b["color"], b["normal_map"], b["mr"],
            base_color=b["base_color"], metallic=b["metallic"], roughness=b["roughness"],
            double_sided=mat0.double_sided, alpha_mode=mat0.alpha_mode)
        out.update(atlas_size=b["atlas_size"], baked_color=b["color"] is not None,
                   baked_normal=b["normal_map"] is not None)
    out["time_bake"] = time.perf_counter() - t
    out["textured_glb"] = textured

    if auto_metrics is not None:
        out["metrics"] = auto_metrics
    elif compute_metrics:
        report(0.9, "Measuring Hausdorff / Chamfer error")
        out["metrics"] = compare_meshes(prep.positions, prep.faces, S, SF, bvh_a=mesh.bvh)
    # wireframe overlays of the low-poly result (clay and textured)
    out["clay_wire_glb"] = add_wireframe(geo_glb, Sm, SF, os.path.join(out_dir, f"{base}_lowpoly_wireframe.glb"),
                                         color=WIRE_ON_CLAY)
    out["lines_glb"] = add_wireframe(None, Sm, SF, os.path.join(out_dir, f"{base}_lowpoly_lines.glb"),
                                     color=WIRE_ON_CLAY)
    if textured:
        out["textured_wire_glb"] = add_wireframe(
            textured, Sm, SF, os.path.join(out_dir, f"{base}_lowpoly_textured_wireframe.glb"),
            color=WIRE_ON_TEXTURE if out.get("baked_color") else WIRE_ON_CLAY)
    report(1.0, "Done")
    return out


WIRE_ON_CLAY = (0.04, 0.06, 0.16, 1.0)     # dark navy lines on the grey clay model
WIRE_ON_TEXTURE = (0.0, 0.85, 1.0, 1.0)    # cyan lines stay visible on dark and light textures


def original_wireframe(mesh, display_glb, out_path, lines_only=False):
    """Wireframe of the dense input (edges of the welded input mesh): overlaid on its display
    GLB, or as lines alone."""
    prep = mesh.prep
    on_texture = mesh.asset.has_appearance and not lines_only
    return add_wireframe(None if lines_only else display_glb, prep.to_model(prep.positions), prep.faces,
                         out_path, color=WIRE_ON_TEXTURE if on_texture else WIRE_ON_CLAY)
