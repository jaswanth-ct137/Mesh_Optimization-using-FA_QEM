# Replace TripoSG's QEM mesh simplification with FA-QEM (Automatic)

You are working in a TripoSG project. It simplifies its generated meshes with ordinary QEM. Replace that
QEM with **FA-QEM (Automatic, best face count)**, which the user developed and tested in another project.
Everything else in TripoSG stays as it is.

This file comes with **`faqem_for_triposg.zip`**. The zip holds the FA-QEM code and one adapter function,
so you never have to work inside FA-QEM itself.

---

## 1. What FA-QEM (Automatic) is

**FA-QEM** is "Fast and Robust Mesh Simplification for Generated and Real-World 3D Assets",
arXiv:2605.14029. It is a quadric edge-collapse simplifier:

* The collapse cost is a composite quadric: base, boundary-curvature and normal terms, plus an area
  term.
* The cheapest collapse is done first. The merged vertex goes to the quadric optimum.
* No collapse may flip a face, break the link condition or pass the edge-valence guard.

**Automatic mode** adds the following on top:

1. **No face target.** It keeps collapsing until no allowed collapse is left.
2. **A local worst-case check on every collapse.** Every original vertex is tracked. A collapse is
   refused if any original point near it would end up further than the allowed deviation from the new
   faces, or if the new faces would end up further than that from the original surface.
3. **Parts and rims are protected.** Separate parts cannot vanish and open rims cannot shrink.
4. **Search.** The real worst-case (Hausdorff) deviation is measured and the local limit is tuned over
   1–3 passes. The smallest result that is within the limit is kept.

The user picks a **detail level**, and the face count follows from it:

| level | max worst-case deviation (fraction of the bbox diagonal) |
|---|---|
| Low | 0.5 % |
| Medium (default) | 0.2 % |
| High | 0.1 % |
| Ultra | 0.05 % |

Tested results, all within the limit: a 627k-face AI-generated doll gives Low 980, Medium 2,316,
High 4,542 and Ultra 9,232 faces. Earlier QEM-style searches needed 2–5× more faces for the same
worst case.

**With a face target** (`target_faces=N`), the same logic is aimed at N instead of at a deviation:

1. One plain FA-QEM pass to N gives the starting worst deviation, and is the fallback.
2. The per-collapse check is switched on and its tolerance searched, down to the smallest value
   that still reaches N.
3. Of the results that reach N, the one with the smallest measured worst deviation is kept.

At the same face count this has a **2–4× smaller worst-case error** than plain FA-QEM (doll at
5,000 faces: 0.085 % instead of 0.337 %). It takes several passes, about 25–100 s instead of 1–3 s.
`fast=True` gives the old single plain pass.

---

## 2. Required behaviour (decided by the user)

| input | behaviour |
|---|---|
| `--detail Low\|Medium\|High\|Ultra` (default **Medium**) | FA-QEM Automatic at that level |
| `--faces N` with N > 0 | best FA-QEM result for N faces; takes precedence over `--detail` |
| `--faces N --fast` | one plain FA-QEM pass to N faces (much faster, larger worst-case error) |
| `--detail none` | no simplification (the old "no decimation" behaviour) |
| `--simplifier qem` | the old PyMeshLab / QEM path, kept as a fallback (default: `faqem`) |

Scale, position, orientation and output file format must stay exactly as they are today.

---

## 3. Step 0: explore first and change nothing yet

Find every place where TripoSG decimates or simplifies a mesh. In the public repo it is
`simplify_mesh()` in `scripts/inference_triposg.py`, which calls PyMeshLab's
`meshing_decimation_quadric_edge_collapse(targetfacenum=...)` when `--faces > 0`. Your copy may differ,
so search:

```bash
grep -rn -e "pymeshlab" -e "meshing_decimation_quadric_edge_collapse" -e "simplify_mesh" \
  -e "decimat" -e "fast_simplification" -e "simplify_quadric_decimation" -e "open3d" \
  -e "\-\-faces" --include="*.py" .
```

For each hit, note three things:

* the file and function
* the mesh type going in and out (`trimesh.Trimesh`, `pymeshlab.Mesh`, or numpy arrays)
* the CLI arguments involved (`--faces` and similar)

Report this to the user before editing anything. If there is more than one decimation call (for
example in a Gradio app or API server as well as the script), ask which ones to replace.

---

## 4. Step 1: install and run the smoke test

1. Unzip `faqem_for_triposg.zip` into the TripoSG repo, for example at `third_party/`:

   ```
   third_party/faqem_for_triposg/
   ├── faqem/                 FA-QEM code (do NOT edit)
   ├── faqem_adapter.py       the function to call
   ├── requirements_faqem.txt
   ├── smoke_test.py
   └── README.txt
   ```

2. Install the dependencies into TripoSG's Python environment:

   ```bash
   pip install -r third_party/faqem_for_triposg/requirements_faqem.txt
   ```

   The list is numpy, scipy, numba ≥ 0.60, trimesh and pillow. Python must be 3.10 or newer.

3. Run the smoke test. It must end with `SMOKE TEST PASSED`:

   ```bash
   python third_party/faqem_for_triposg/smoke_test.py            # sphere + box
   python third_party/faqem_for_triposg/smoke_test.py some.glb   # optional: a real TripoSG output
   ```

   The first run compiles the Numba code, which takes 30–60 s; after that it is cached.

---

## 5. Step 2: replace the QEM call

The adapter is the only interface you use:

```python
import sys, os
sys.path.insert(0, os.path.join(<repo_root>, "third_party", "faqem_for_triposg"))
from faqem_adapter import simplify_mesh_faqem

V2, F2, info = simplify_mesh_faqem(vertices, faces, detail="Medium")      # Automatic
V2, F2, info = simplify_mesh_faqem(vertices, faces, target_faces=20000)   # best result for 20,000 faces
V2, F2, info = simplify_mesh_faqem(vertices, faces, target_faces=20000, fast=True)  # one plain pass
# info = {'mode', 'input_faces', 'output_faces', 'deviation_percent', 'within_limit', 'seconds'}
```

* **Input:** `vertices` (n, 3) and triangle `faces` (m, 3), numpy arrays or anything array-like, at any
  scale.
* **Output:** vertices and faces in the same coordinate space as the input.
* **Optional:** `tolerance=0.003` sets an exact limit (a fraction of the bbox diagonal) instead of a
  level.

Inside TripoSG's existing simplify function (for example `simplify_mesh`):

1. Convert to numpy. For trimesh: `mesh.vertices`, `mesh.faces`. For pymeshlab:
   `m.vertex_matrix()`, `m.face_matrix()`.
2. Call `simplify_mesh_faqem(...)` following the rules in section 2.
3. Rebuild the same mesh type the function returned before, for example
   `trimesh.Trimesh(V2, F2, process=False)`, so the rest of the pipeline is unaffected.
4. Log `info`, for example:
   `FA-QEM auto Medium: 612,440 -> 2,904 faces, deviation 0.18 % (within limit), 21 s`.

Add these CLI arguments next to the existing `--faces`:

* `--detail {none,Low,Medium,High,Ultra}`, default `Medium`
* `--fast`: with `--faces`, one plain FA-QEM pass (`fast=True`)
* `--simplifier {faqem,qem}`, default `faqem`

Keep the old PyMeshLab code as the `qem` branch.

### Rules

* Do **not** edit anything inside `faqem/`. If something seems wrong in it, report it instead.
* Keep the diff minimal and don't refactor unrelated code.
* Keep the old QEM path working through `--simplifier qem`.
* Don't change output paths, file formats, scaling or orientation.

---

## 6. Step 3: verify

1. The smoke test passes in TripoSG's environment.
2. Run TripoSG inference on its example images with `--detail Low`, `Medium`, `High` and `Ultra`, and
   with `--faces 20000`. Check that:
   * every output loads (trimesh or Blender) and sits in the same place and at the same scale as before;
   * `deviation_percent` is at or below the level in every Automatic run, and `within_limit` is `True`;
   * face counts rise from Low to Ultra;
   * `--faces 20000` gives 20,000 faces (or one or two fewer), with a smaller deviation than `--faces 20000 --fast`.
3. For one image, compare against the old path at the same face count:
   `--simplifier qem --faces <the count Medium chose>`.
4. Report a table to the user:

   | image | mode | faces | worst deviation | time |
   |---|---|---|---|---|

---

## 7. Troubleshooting

| problem | fix |
|---|---|
| Numba cache or permission errors | `export NUMBA_CACHE_DIR=/tmp/numba_cache` |
| slow | FA-QEM runs on the CPU (Numba), taking about 15–35 s per pass at around 1 M faces, and Automatic makes 1–3 passes. Large perfectly flat CAD faces are slower than organic shapes. |
| mesh with 0 faces or non-triangles | the adapter returns the input unchanged for 0 faces; triangulate polygons before calling it |
| output looks shifted or scaled | you probably used internal faqem arrays; use only the adapter's returned `V2`, `F2` |

---

## 8. Report back to the user

* the files you changed, with a short diff summary
* the commands you ran
* the results table from section 6
* anything that did not work, or any decision you had to make

---

## Optional: the C++ engine

The FA-QEM project now also has a C++ engine (`cpp/` there) that gives **bit-identical** results:
the same face counts, vertices and deviations. If `faqem_cpp*.so` sits next to `faqem_adapter.py`,
`simplify_mesh_faqem()` uses it automatically. Nothing changes in TripoSG.

* `info["engine"]` reports `"cpp"` or `"python"`.
* `FAQEM_BACKEND=python` forces the Python engine.
* `FAQEM_BACKEND=cpp` makes a missing build an error instead of a silent fallback.

The `.so` is built for and verified on macOS arm64 with Python 3.12. It does **not** build on Linux
yet, because it replays numpy's calls into Apple's Accelerate library. A Linux GPU server needs a
port of `cpp/src/compat/accelerate.cpp` to OpenBLAS, which numpy uses there, plus its own parity run.
Until then, TripoSG on Linux uses the Python engine automatically.
