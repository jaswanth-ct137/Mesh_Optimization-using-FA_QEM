# FA-QEM Low-Poly Studio

A Python implementation of **FA-QEM** — *Feature-Aware Quadric Error Metric* mesh simplification
from Bhosikar et al., "Fast and Robust Mesh Simplification for Generated and Real-World 3D Assets"
(arXiv:2605.14029, `FA-QEM.pdf` in this folder). It ships with a web app: upload a dense triangle mesh
and get back a low-poly mesh, with its texture and fine surface detail transferred onto it.

## Quick start

```bash
python3.12 -m venv .venv            # or: uv venv --python 3.12 .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py             # open http://127.0.0.1:7860
```

In the app:

1. Upload a mesh (GLB, GLTF, OBJ, STL, PLY, OFF; Draco-compressed GLB also works).
2. Choose a target:
   * **Percentage** of faces to keep (0.001 – 90 %; type small values into the box next to the slider),
   * **Face count**, or
   * **Automatic (best face count)** with a detail level; the face count then follows from the
     allowed deviation (see below).
3. Click **Simplify**.

The two viewers show the original and the result side by side. Above them:

* **Wireframe (both viewers)**: *Off*, *Overlay on surface* (edges drawn over the model, in cyan
  on textured models and dark blue on clay) or *Wireframe only* (just the edges). This lets you compare
  the dense input triangulation with the adaptive low-poly one. Switching modes is instant and needs
  no re-run.
* **Low-poly shading**: *Textured* (baked colour and normal map) or *Clay* (geometry only).

You can download:

* `*_lowpoly_textured.glb`: the low-poly mesh with a baked colour texture, a normal map and, when the
  source has them, metallic-roughness maps
* `*_lowpoly_geometry.glb` and `*_lowpoly.obj`: the plain low-poly geometry with shared vertices
* `*_lowpoly_wireframe.glb`: the low-poly model with its edges drawn in

Command line:

```bash
.venv/bin/python cli.py model.glb --ratio 0.05 -o out/
.venv/bin/python cli.py scan.stl --faces 5000 --no-normal-map
.venv/bin/python cli.py model.glb --w-plane-area 1 --virtual-edges true   # the paper's exact Table 1
```

Tests: `.venv/bin/python -m pytest -q tests`

## C++ engine (bit-identical)

`cpp/` contains a C++ port of everything except the web UI. Build it once (see `cpp/README.md`) and
`app.py`, `cli.py` and the TripoSG adapter use it automatically. Its results are identical to the
Python code here, down to every vertex bit, every Automatic search pass and every baked texel.
`cpp/build/faqem` is the command-line tool, with the same flags as `cli.py`. Set
`FAQEM_BACKEND=python` to use the Python engine.

## What is implemented (paper → code)

| Paper | Code |
|---|---|
| Composite quadric Q_gf = Q_base + Q_boundary + Q_normal (Eq. 4) | `core.build_quadrics` |
| Inverse-area weighted base quadric (Eq. 5), `w_plane_area` | `core.build_quadrics` |
| Boundary curvature κ (Eq. 6), dual planes p1/p2 (Eq. 7–9), `w_boundary` | `core.build_quadrics` |
| Normal preservation tangent plane (Eq. 10), `w_normal` | `core.build_quadrics` |
| Lindstrom–Turk boundary area cost Q_A (Eq. 11), `w_area`; total cost (Eq. 2) | `core._add_area_edge`, `core._eval_edge` |
| Optimal v′ from a 3×3 solve, falling back to {v_i, v_j, midpoint} | `core._solve_q`, `core._eval_edge` |
| Priority-queue edge collapse loop, neighbour cost updates (Algorithm 1) | `core.run_collapses` (Numba) |
| Normal-flip veto; non-manifold-safe adjacency (Sec. 3.3) | `core._flip_ok`, `core._can_collapse` |
| Virtual edges between nearby components via KD-tree ball queries (Sec. 3.3, supp. 6.2) | `virtual_edges.py` |
| Vertex welding, tiny-edge infinite-cost guard, degenerate-solve fallback (supp. 6.2) | `prep.py`, `core.py` |
| Collapse history → successive mapping → texture bake with per-triangle atlas (Sec. 3.2.3) | `core` (`history`, `vertex_map`), `bake.py` |
| Hausdorff / Chamfer evaluation (Sec. 4.1) | `metrics.py` |

Performance: the hot loops are Numba-compiled, the same approach the paper takes. A 627k-face
Hunyuan3D model simplifies to 31k faces in about 6 s, and the bake takes about 7 s. The paper reports
28 minutes for a 240k-face mesh.

## Where this differs from the paper, and why

The paper leaves some scales and units unspecified. Every choice below was measured on real
AI-generated and scanned meshes, and on synthetic stress shapes (`tests/shapes.py`), using the paper's
own metrics.

1. **Normalised space.** The mesh is scaled to a unit bounding-box diagonal, face areas in Eq. 5 are
   measured relative to the mean face area, and Q_A is divided by the squared mean edge length. This
   way the fixed Table 1 weights behave the same on millimetre CAD parts and on metre-scale scans.
2. **κ in Eq. 6 is made dimensionless** (κ × half the chord length, roughly the turning angle). With
   κ in model units, the boundary term on dense, zig-zagging AI-generated rims grew several orders of
   magnitude larger than the base quadric. That froze the rims and roughly doubled the error at 2% targets.
   With the dimensionless κ, the paper's own w_boundary = 500 works well.
3. **Two terms are off by default: Eq. 5 inverse-area weighting (w_plane_area = 0) and virtual edges.**
   In our ablation both increased error on real meshes, for example on a textured boot model at 5%
   faces: Hausdorff 0.31% without them, 0.81% with Eq. 5 alone, 0.54% with virtual edges alone,
   and 1.39% with both. They are still fully
   implemented. Choose the **"Paper Table 1 (exact)"** preset (app) or pass the flags shown above (CLI)
   to reproduce the paper exactly.
4. **Non-manifold input is split into manifold sheets** after welding. AI-generated meshes often
   contain layers that touch, which welding turns into non-manifold edges. Collapses around those edges
   stacked faces into folded "fins" that every normal-flip check still accepted. After the split, the
   touching layers become coincident open rims, which the boundary terms protect.
5. **Topology is preserved "as much as possible", as the paper says:** a collapse that would change
   topology is re-queued at 10 × its cost rather than forbidden. Forbidding such collapses outright
   destroyed one AI mesh (12.6% Hausdorff error); with the penalty, the same run reaches its target at
   2.2%.
6. **Flip retry.** If the optimal v′ would fold a face, the cheapest valid position among
   {v*, v_i, v_j, midpoint} is re-queued rather than dropping the edge.
7. **Appearance transfer** follows the paper's successive-mapping idea. Each texel is matched to the
   closest point on those original triangles whose vertices collapsed into the corners of the texel's
   low-poly triangle, with a check that the surfaces face the same way, so thin walls don't pick up the
   colour of their back side. This makes the paper's "closer parent vertex" heuristic exact.
   **Normal-map baking is an extension** beyond the paper: it recovers the dense mesh's fine surface
   detail on the low-poly geometry.

Measured results (Hausdorff error as % of the bbox diagonal; lower is better):

| mesh @ keep | plain QEM | paper Table 1 exact | **recommended (default)** |
|---|---|---|---|
| boots (129k faces) @ 5% | 0.52 | 1.39 | **0.31** |
| boots @ 2% | 0.90 | 2.43 | **0.88** |
| bowl, open rim @ 10% | 2.68 | 0.14 | **0.08** |
| star disk, open rim @ 2% | 36.5 | 0.00 | **0.00** |
| AI doll, non-manifold @ 2% | 2.17 | 5.55 | 2.49 |

## Automatic mode: the face count follows from the detail

Choose the target **Automatic (best face count)** and a **detail level** instead of a number:

| level | allowed worst-case deviation |
|---|---|
| Low | 0.5 % of the model size |
| Medium | 0.2 % |
| High (default) | 0.1 % |
| Ultra | 0.05 % |

For a custom limit, fill in *exact max deviation*.

**How the guarantee is enforced: locally, at every collapse** (`core.py: _lh_check`).
Every original vertex (plus face centres on inputs up to 400k faces) is a tracked point, attached to
one face of the current mesh. A collapse is accepted only if:

* every point attached to the faces it changes stays within the tolerance of the new faces (each
  point then moves to its nearest new face), and
* the new faces stay within the tolerance of the original. The new vertex is checked, and so are
  points spread over each new face; big faces get more of them, up to about 45.

So each area is simplified as far as its own shape allows. A noisy spot or a thin carving stays
dense locally instead of setting the face count for the whole mesh, which is what a single global
error cap did. Surface between tracked points can exceed the local bound slightly, so the real
worst-case (Hausdorff) deviation is still measured. The local tolerance starts at 80 % of the limit,
is bracketed and bisected (usually 1–3 passes), and the smallest result within the limit is kept.

Measured on 7 SKUs. The face count is chosen automatically, and all 28 runs are within the limit
(the earlier global-cap version is in brackets):

| mesh (input faces) | Low 0.5 % | Medium 0.2 % | High 0.1 % | Ultra 0.05 % |
|---|---|---|---|---|
| wooden rack (854,828) | 216 (502) | 396 (1,910) | 654 (5,032) | 1,414 (8,406) |
| L-shaped sofa (860,218) | 398 (3,150) | 782 (14,602) | 1,656 (23,339) | 3,631 (25,179) |
| STL scan (11,420) | 392 (2,706) | 866 (7,152) | 1,790 (7,826) | 3,474 (8,970) |
| cartoon koala (1,790,758) | 1,556 (4,960) | 3,406 (12,970) | 6,922 (25,802) | 14,598 (49,030) |
| Hunyuan doll (627,666) | 980 (2,984) | 2,316 (6,368) | 4,542 (11,802) | 9,232 (25,006) |
| boots (129,094) | 2,412 (3,909) | 5,491 (12,967) | 13,218 (19,728) | 18,313 (42,760) |
| Versa scan (93,126) | 4,821 (8,102) | 11,857 (72,913) | 25,221 (76,465) | 48,861 (83,524) |

That is 2–8× fewer faces at the same guaranteed worst case. The noisy Versa scan improves most at
Medium/High, where one rough spot used to keep 78–82 % of its faces. A pass takes about 0.3 s for
small meshes and 15–35 s for 0.8–1.8 M faces. Automatic mode takes 1–100 s in total. CLI:
`cli.py model.glb --auto High` or `--auto-deviation 0.1` (percent).

## Percentage and Face count: the best result for a face count

Percentage and Face count use the same logic as Automatic, aimed at your face count instead of a
deviation (`pipeline.simplify_to_count`):

1. **One plain FA-QEM pass to the target.** It gives the starting worst deviation and is the fallback,
   so the result is never worse than plain FA-QEM.
2. **Search for the smallest local tolerance that still reaches the count.** The per-collapse check
   from Automatic is switched on. The search halves the tolerance while the face count is still
   reached and grows it while it isn't, then bisects.
3. **Keep the best.** Of the results that reach the count, the one with the smallest measured worst
   deviation is kept.

With a smaller allowed error, collapses can no longer pile all the error into a few spots. The faces
go where the shape needs them, and the worst case drops 2–4× at the same face count:

| mesh, face count | plain FA-QEM (Fast) | best for the count | time |
|---|---|---|---|
| boots, 3,000 | 0.651 % | 0.361 % | 1 s → 24 s |
| boots, 10,000 | 0.279 % | 0.084 % | 1 s → 48 s |
| Hunyuan doll, 5,000 | 0.337 % | 0.085 % | ~3 s → 93 s |
| Hunyuan doll, 12,553 | 0.117 % | 0.039 % | ~3 s → 103 s |

The face count comes out exact, or one or two fewer. Tick **Fast** in the app, or pass `--fast` on the
CLI, for the old single plain pass.

## Layout

```
app.py              Gradio web app
cli.py              command-line tool
faqem/core.py       FA-QEM quadrics + Numba collapse loop (Algorithm 1) + local worst-case check
faqem/prep.py       welding, cleanup, non-manifold splitting, consistent orientation, normalisation
faqem/virtual_edges.py
faqem/bake.py       successive-mapping texture / normal / PBR bake
faqem/pipeline.py   load / simplify / Automatic search / bake / export
faqem/bvh.py        closest-point BVH (Numba)
faqem/metrics.py    Hausdorff / Chamfer
faqem/io.py         loading (trimesh) and GLB/OBJ export
faqem/wireframe.py  wireframe overlays (edges written as a glTF LINES primitive)
tests/              pytest suite, procedural stress shapes, software renderer
                    (test_cpp_parity.py: bit-exact C++ vs Python checks)
faqem/engine.py     picks the C++ engine when built (FAQEM_BACKEND=cpp|python)
cpp/                C++ engine: library, `faqem` CLI, `faqem_cpp` Python module (cpp/README.md)
tools/compare_backends.py   runs both CLIs on the SKUs and compares every output
```

## Limitations

* Very aggressive targets on messy AI meshes (for example about 1–2k faces for a figure with hands and
  hair) still produce spiky triangles. The independent `fast-simplification` QEM library behaves the
  same way on those inputs.
* The baked atlas gives each triangle pair its own chart, so the textured GLB stores 3 vertices per
  face. The geometry downloads keep shared vertices.
* KTX2/Basis-compressed textures are not decoded.
* Automatic mode on very noisy scans keeps many faces at High / Ultra: the noise itself is larger
  than the allowed deviation.
