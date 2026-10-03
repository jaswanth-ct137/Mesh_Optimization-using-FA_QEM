"""FA-QEM Low-Poly Studio - upload a dense triangle mesh, get a feature-preserving low-poly mesh.

    python app.py            # then open http://127.0.0.1:7860
"""
import os
import tempfile
import time

import gradio as gr
import trimesh

from faqem.core import DEFAULTS, PRESETS
from faqem.io import SUPPORTED
from faqem.engine import DETAIL_LEVELS, LABELS, LoadedMesh, available, backend, describe, original_wireframe, run

# While you type into a Slider's number box it briefly holds "" or "0" (typing "0.05"); Gradio
# would raise errors for those. Pass empty as None and clamp in-between values into range.
_slider_preprocess = gr.Slider.preprocess


def _lenient_slider_preprocess(self, payload):
    if payload is None:
        return None
    return _slider_preprocess(self, min(max(payload, self.minimum), self.maximum))


gr.Slider.preprocess = _lenient_slider_preprocess

WORK = os.path.join(tempfile.gettempdir(), "faqem_app")
os.makedirs(WORK, exist_ok=True)
_CACHE = {}
ENGINES = LABELS                                         # internal name -> label
ENGINE_LABEL = "Engine"
_ENGINE = {"name": backend()}                            # engine chosen in the UI (default: C++ if built)


def set_engine(label):
    _ENGINE["name"] = next(k for k, v in ENGINES.items() if v == label)


def _load(path):
    key = (path, os.path.getmtime(path), _ENGINE["name"])
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = LoadedMesh(path, engine=_ENGINE["name"])
    return _CACHE[key]


_ORIG_WIRE = {}
MAX_OVERLAY_FACES = 1_500_000  # beyond this the original falls back to the viewer's own wireframe mode


def _display_copy(path):
    """Model3D renders .glb/.obj/.stl natively; convert everything else to GLB."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".glb", ".stl", ".obj"):
        return path
    return _as_glb(path)


def _as_glb(path):
    if path.lower().endswith(".glb"):
        return path
    out = os.path.join(WORK, os.path.splitext(os.path.basename(path))[0] + "_view.glb")
    if not os.path.exists(out):
        trimesh.load(path, force="scene", process=False).export(out)
    return out


def _original_wire(path, lines_only=False):
    key = (path, lines_only)
    if key not in _ORIG_WIRE:
        m = _load(path)
        suffix = "_lines.glb" if lines_only else "_wireframe.glb"
        out = os.path.join(WORK, os.path.splitext(os.path.basename(path))[0] + suffix)
        _ORIG_WIRE[key] = original_wireframe(m, _as_glb(path), out, lines_only=lines_only)
    return _ORIG_WIRE[key]


def on_upload(file):
    if file is None:
        return None, "Upload a mesh to begin.", gr.update(), gr.update()
    path = file if isinstance(file, str) else file.name
    try:
        m = _load(path)
    except Exception as e:  # noqa: BLE001
        raise gr.Error(f"Could not read this mesh: {e}")
    s, a = m.stats, m.asset.info
    appearance = "texture" if a["textured"] else ("vertex colours" if a["vertex_colors"] else "plain material")
    info = (
        f"**{os.path.basename(path)}**  \n"
        f"{s['faces']:,} faces · {s['vertices']:,} vertices · {s['components']} component(s)  \n"
        f"{s['boundary_edges']:,} boundary edges · {s['non_manifold_edges']:,} non-manifold edges  \n"
        f"Appearance: {appearance} · {a['parts']} part(s), {a['materials']} material(s)  \n"
        f"Welded {m.prep.removed['welded']:,} duplicate vertices, removed "
        f"{m.prep.removed['degenerate'] + m.prep.removed['duplicate']:,} degenerate/duplicate faces "
        f"({m.load_time:.1f}s, {ENGINES[getattr(m, 'backend', 'python')]} engine)"
    )
    default_target = max(100, s["faces"] // 10)
    return _display_copy(path), info, gr.update(value=10.0), gr.update(value=default_target)


def sync_count(file, pct):
    if file is None or pct is None:
        return gr.update()
    m = _load(file if isinstance(file, str) else file.name)
    return gr.update(value=max(4, int(round(m.stats["faces"] * pct / 100.0))))


def apply_preset(name):
    o = {**DEFAULTS, **PRESETS[name]}
    return (o["w_area"], o["w_boundary"], o["w_normal"], o["w_plane_area"], o["virtual_edges"])


def simplify_ui(file, mode, pct, count, bake_color, bake_normal, atlas, crease,
                w_area, w_boundary, w_normal, w_plane_area, virtual, tau, flip_thr, topo, metrics,
                detail, auto_dev, fast, progress=gr.Progress()):
    if file is None:
        raise gr.Error("Please upload a mesh first.")
    path = file if isinstance(file, str) else file.name
    m = _load(path)
    auto_mode = mode == TARGET_AUTO
    auto_tol = (float(auto_dev) / 100.0 if auto_dev and float(auto_dev) > 0 else DETAIL_LEVELS[detail]) \
        if auto_mode else None
    if mode == "Percentage" and pct is None:
        raise gr.Error("Please enter a percentage of faces to keep (0.001 – 90).")
    if mode == "Face count" and not count:
        raise gr.Error("Please enter a target face count.")
    target = int(count) if mode == "Face count" else int(round(m.stats["faces"] * pct / 100.0))
    target = max(4, min(target, m.stats["faces"]))
    out_dir = os.path.join(WORK, f"run_{int(time.time() * 1000)}")
    options = dict(w_area=w_area, w_boundary=w_boundary, w_normal=w_normal, w_plane_area=w_plane_area,
                   virtual_edges=virtual, virtual_tau=tau, flip_threshold=flip_thr,
                   preserve_topology=topo)
    t0 = time.perf_counter()
    res = run(m, None if auto_mode else target, out_dir, options=options, bake_color=bake_color,
              bake_normal=bake_normal, auto_tolerance=auto_tol,
              atlas_size=None if atlas == "Auto" else int(atlas), crease_angle=crease,
              compute_metrics=metrics, fast=bool(fast), progress=lambda f, msg: progress(f, desc=msg))
    total = time.perf_counter() - t0
    st = res["stats"]
    lines = [
        f"### {st['input_faces']:,} → **{st['output_faces']:,} faces**  "
        f"({100 * st['output_faces'] / st['input_faces']:.3g}% · {st['input_faces'] / max(1, st['output_faces']):.1f}× fewer)",
        f"Vertices {st['input_vertices']:,} → {st['output_vertices']:,}",
        f"Engine: **{describe(m)}** · time: simplification {st['time_total']:.2f}s · "
        f"appearance bake {res['time_bake']:.2f}s · total {total:.2f}s",
    ]
    if "metrics" in res:
        mt = res["metrics"]
        lines.append(
            f"Fidelity (relative to bbox diagonal): Hausdorff **{mt['hausdorff'] * 100:.3f}%** · "
            f"mean deviation {mt['mean_distance'] * 100:.4f}% · Chamfer {mt['chamfer']:.2e}")
    lines.append(
        f"Collapses {st['collapses']:,} (virtual {st['virtual_collapses']:,}) · rejected: flips "
        f"{st['rejected_flip']:,}, link {st['rejected_link']:,}, error limit {st.get('rejected_error', 0):,}")
    if auto_mode:
        ps = st.get("auto_passes", [])
        ok = "✓ within the limit" if st.get("auto_within") else "⚠️ closest result found is above the limit"
        lines.append(
            f"Mode: **automatic** — allowed deviation {auto_tol * 100:.3g}% of size → the face count follows "
            f"from the detail: {st['output_faces']:,} faces, measured worst deviation "
            f"{res['metrics']['hausdorff'] * 100:.3f}% ({ok}); passes: "
            + ", ".join(f"{p['faces']:,} faces / {p['hausdorff'] * 100:.3f}%" for p in ps))
    elif st.get("best_for_count"):
        gain = st["plain_hausdorff"] / max(res["metrics"]["hausdorff"], 1e-12)
        lines.append(
            f"Mode: **best for this face count** — worst deviation **{res['metrics']['hausdorff'] * 100:.3f}%** "
            f"(plain FA-QEM at the same count: {st['plain_hausdorff'] * 100:.3f}%"
            + (f", {gain:.1f}× lower" if gain > 1.01 else "") + f"); {len(st.get('count_passes', []))} passes. "
            f"Tick *Fast* for a single plain FA-QEM pass.")
    if not auto_mode and not st["reached_target"]:
        why = ("the error limit" if st.get("rejected_error") else
               "the remaining collapses would fold or tear the surface")
        lines.append(f"⚠️ Target not fully reached: stopped by {why}.")
    if res.get("textured_glb"):
        lines.append(f"Baked {res['atlas_size']}² atlas: "
                     + ", ".join(k for k, v in (("colour", res.get("baked_color")),
                                                ("normal map", res.get("baked_normal"))) if v))
    views = dict(textured=res.get("textured_glb") or res["geometry_glb"], clay=res["geometry_glb"],
                 textured_wire=res.get("textured_wire_glb") or res["clay_wire_glb"],
                 clay_wire=res["clay_wire_glb"], lines=res["lines_glb"])
    files = [f for f in (res.get("textured_glb"), res["geometry_glb"], res["geometry_obj"],
                         res["clay_wire_glb"]) if f]
    return "\n\n".join(lines), files, views


WIRE_OFF, WIRE_OVERLAY, WIRE_ONLY = "Off", "Overlay on surface", "Wireframe only"
TARGET_AUTO = "Automatic (best face count)"
VIEWER_BG = (0.93, 0.94, 0.96, 1.0)


def update_views(file, wire, shading, views):
    """Apply the wireframe / shading choice to both viewers."""
    orig_upd = gr.update()
    if file is not None:
        path = file if isinstance(file, str) else file.name
        big = _load(path).stats["faces"] > MAX_OVERLAY_FACES
        if wire == WIRE_OFF:
            orig_upd = gr.update(value=_display_copy(path), display_mode="solid")
        elif big:  # too many edges to ship as lines: use the viewer's own wireframe mode
            orig_upd = gr.update(value=_display_copy(path), display_mode="wireframe")
        else:
            orig_upd = gr.update(value=_original_wire(path, lines_only=wire == WIRE_ONLY), display_mode="solid")
    res_upd = gr.update()
    if views:
        key = "textured" if shading == "Textured" else "clay"
        if wire == WIRE_OVERLAY:
            res_upd = gr.update(value=views[key + "_wire"], display_mode="solid")
        elif wire == WIRE_ONLY:
            res_upd = gr.update(value=views["lines"], display_mode="solid")
        else:
            res_upd = gr.update(value=views[key], display_mode="solid")
    return orig_upd, res_upd


CSS = """
#title {margin-bottom: 0}
.viewer {min-height: 520px}
"""

with gr.Blocks(title="FA-QEM Low-Poly Studio") as demo:
    views_state = gr.State({})
    gr.Markdown(
        "# FA-QEM Low-Poly Studio\n"
        "Feature-Aware Quadric Error Metric simplification "
        "(Bhosikar et al., *Fast and Robust Mesh Simplification for Generated and Real-World 3D Assets*). "
        "Upload a dense triangle mesh (GLB, GLTF, OBJ, STL, PLY, OFF); the low-poly result keeps sharp "
        "features, open-boundary rims and fine detail, and the original appearance is transferred by "
        "successive mapping.", elem_id="title")
    with gr.Row():
        with gr.Column(scale=1, min_width=320):
            engine = gr.Radio([ENGINES[e] for e in available()], value=ENGINES[_ENGINE["name"]],
                              label=ENGINE_LABEL)
            file = gr.File(label="Dense mesh", file_types=list(SUPPORTED), type="filepath")
            info = gr.Markdown("Upload a mesh to begin.")
            mode = gr.Radio(["Percentage", "Face count", TARGET_AUTO], value="Percentage", label="Target")
            with gr.Group():
                detail = gr.Radio(list(DETAIL_LEVELS), value="High",
                                  label="Automatic: detail level (allowed deviation Low 0.5% · Medium 0.2% · "
                                        "High 0.1% · Ultra 0.05% of size)")
                auto_dev = gr.Number(0, label="…or exact max deviation (% of size, 0 = use the level)")
            pct = gr.Slider(0.001, 90, value=10, step=0.001, label="Keep % of faces (0.001 – 90)")
            count = gr.Number(value=10000, precision=0, label="Target face count")
            fast = gr.Checkbox(False, label="Fast (plain FA-QEM, one pass) — Percentage / Face count only; "
                                            "off = best result for the face count (slower)")
            with gr.Row():
                bake_color = gr.Checkbox(True, label="Transfer colour / texture")
                bake_normal = gr.Checkbox(True, label="Bake normal map (fine detail)")
            with gr.Accordion("FA-QEM parameters", open=False):
                preset = gr.Dropdown(list(PRESETS), value="Recommended", label="Preset")
                w_area = gr.Number(DEFAULTS["w_area"], label="w_area – boundary area preservation")
                w_boundary = gr.Number(DEFAULTS["w_boundary"], label="w_boundary – curvature boundary penalty")
                w_normal = gr.Number(DEFAULTS["w_normal"], label="w_normal – normal preservation")
                w_plane_area = gr.Number(DEFAULTS["w_plane_area"], label="w_plane_area – inverse-area weighting")
                virtual = gr.Checkbox(DEFAULTS["virtual_edges"], label="Virtual edges (merge nearby components)")
                tau = gr.Number(DEFAULTS["virtual_tau"], label="Virtual edge distance τ (× bbox diagonal)")
                flip_thr = gr.Slider(-0.5, 0.9, value=DEFAULTS["flip_threshold"], step=0.05,
                                     label="Normal-flip threshold (cos)")
                topo = gr.Checkbox(DEFAULTS["preserve_topology"], label="Preserve topology (link condition)")
                atlas = gr.Dropdown(["Auto", "1024", "2048", "4096", "8192"], value="Auto", label="Texture atlas size")
                crease = gr.Slider(10, 90, value=60, step=1, label="Shading crease angle (°)")
                metrics = gr.Checkbox(True, label="Compute Hausdorff / Chamfer error")
            go = gr.Button("Simplify", variant="primary")
        with gr.Column(scale=3):
            with gr.Row():
                wire = gr.Radio([WIRE_OFF, WIRE_OVERLAY, WIRE_ONLY], value=WIRE_OFF,
                                label="Wireframe (both viewers)")
                shading = gr.Radio(["Textured", "Clay (geometry only)"], value="Textured",
                                   label="Low-poly shading")
            with gr.Row():
                # fixed light background so dark wireframe lines stay visible in dark browser themes
                orig = gr.Model3D(label="Original (dense)", height=560, elem_classes="viewer",
                                  clear_color=VIEWER_BG)
                result = gr.Model3D(label="Low-poly result", height=560, elem_classes="viewer",
                                    clear_color=VIEWER_BG)
            gr.Markdown("Tip: dense originals are drawn with translucent edges — zoom in (scroll) to see "
                        "individual triangles. The two viewers have separate cameras; ↺ resets a view.")
            summary = gr.Markdown()
            downloads = gr.File(label="Downloads", file_count="multiple")

    engine.change(set_engine, [engine], None).then(on_upload, [file], [orig, info, pct, count])
    preset.change(apply_preset, [preset], [w_area, w_boundary, w_normal, w_plane_area, virtual])
    file.change(on_upload, [file], [orig, info, pct, count]).then(
        lambda: ({}, None), None, [views_state, result]).then(
        update_views, [file, wire, shading, views_state], [orig, result])
    pct.change(sync_count, [file, pct], [count])
    go.click(simplify_ui,
             [file, mode, pct, count, bake_color, bake_normal, atlas, crease, w_area, w_boundary, w_normal,
              w_plane_area, virtual, tau, flip_thr, topo, metrics, detail, auto_dev, fast],
             [summary, downloads, views_state]).then(
        update_views, [file, wire, shading, views_state], [orig, result])
    wire.change(update_views, [file, wire, shading, views_state], [orig, result])
    shading.change(update_views, [file, wire, shading, views_state], [orig, result])


if __name__ == "__main__":
    demo.queue().launch(css=CSS, theme=gr.themes.Soft(), allowed_paths=[WORK],
                        server_name=os.environ.get("HOST", "127.0.0.1"),
                        server_port=int(os.environ.get("PORT", 7860)))
