"""Generate sample outputs + same-camera comparison renders into samples/."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image, ImageDraw
from faqem.pipeline import LoadedMesh, run
from faqem.core import simplify
from faqem.prep import prepare_mesh
from faqem.metrics import compare_meshes
from tests.render import render
from tests import shapes

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")
D = "/Users/jaswanthandhavarapu/Downloads/"
SAMPLES = [
    ("hunyuan_doll", D + "Hunyuan3D-2/output_user6_textured_hq.glb", 0.02, (0, 0.1, 1.7)),
    ("boots", D + "buwch_Boots (1).glb", 0.10, (0.5, 0.35, 1.5)),
    ("versa", D + "Versa AI Model.glb", 0.05, (0, 0.15, 1.7)),
    ("stl_scan", D + "2026-0806-125028-457187.stl", 0.10, (0.9, 0.9, 1.2)),
]

def label(img, text):
    d = ImageDraw.Draw(img); d.rectangle([0, 0, img.width, 34], fill=(245, 246, 250)); d.text((12, 10), text, fill=(20, 20, 40))
    return img

def strip(images):
    W = sum(i.width for i in images); out = Image.new("RGB", (W, images[0].height), "white"); x = 0
    for i in images: out.paste(i, (x, 0)); x += i.width
    return out

summary = {}
for name, path, ratio, eye in SAMPLES:
    d = os.path.join(OUT, name); os.makedirs(d, exist_ok=True)
    m = LoadedMesh(path)
    res = run(m, int(len(m.prep.faces) * ratio), d)
    P, F = m.prep.positions, m.prep.faces
    import trimesh
    g = trimesh.load(res["geometry_obj"], process=False)
    S = (np.asarray(g.vertices) - m.prep.center) / m.prep.diag; SF = np.asarray(g.faces)
    st, mt = res["stats"], res["metrics"]
    ims = [label(render(P, F, eye, np.zeros(3), size=(600, 600)), f"Original: {st['input_faces']:,} faces"),
           label(render(S, SF, eye, np.zeros(3), size=(600, 600)), f"FA-QEM low-poly: {st['output_faces']:,} faces ({ratio*100:g}%)"),
           label(render(S, SF, eye, np.zeros(3), size=(600, 600), wire=True), f"Wireframe - Hausdorff {mt['hausdorff']*100:.3f}% of size")]
    strip(ims).save(os.path.join(d, "geometry_comparison.png"))
    summary[name] = dict(input_faces=st["input_faces"], output_faces=st["output_faces"], hausdorff_pct=round(mt["hausdorff"] * 100, 4),
                         time_simplify_s=round(st["time_total"], 2), time_bake_s=round(res["time_bake"], 2))
    print(name, summary[name], flush=True)

# open-boundary shape: plain QEM vs FA-QEM (what the paper's boundary terms buy)
pm = prepare_mesh(*shapes.bowl(260)); P, F = pm.positions, pm.faces; t = len(F) // 50
ims = [label(render(P, F, (0.2, 1.0, 1.1), np.zeros(3), size=(600, 600), wire=False), f"Original bowl: {len(F):,} faces")]
for nm, opt in [("Plain QEM", dict(w_area=0, w_boundary=0, w_normal=0, w_plane_area=0)), ("FA-QEM", {})]:
    r = simplify(P, F, t, opt); c = compare_meshes(P, F, r["positions"], r["faces"])
    ims.append(label(render(r["positions"], r["faces"], (0.2, 1.0, 1.1), np.zeros(3), size=(600, 600), wire=True),
                     f"{nm}: {len(r['faces']):,} faces, Hausdorff {c['hausdorff']*100:.2f}%"))
os.makedirs(os.path.join(OUT, "open_boundary_bowl"), exist_ok=True)
strip(ims).save(os.path.join(OUT, "open_boundary_bowl", "plain_qem_vs_faqem.png"))
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
