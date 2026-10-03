"""One entry point for using FA-QEM from another project (e.g. TripoSG).

    from faqem_adapter import simplify_mesh_faqem
    V2, F2, info = simplify_mesh_faqem(mesh.vertices, mesh.faces, detail="Medium")      # Automatic
    V2, F2, info = simplify_mesh_faqem(mesh.vertices, mesh.faces, target_faces=20000)   # face target

* detail / tolerance -> "Automatic (best face count)": the face count is chosen so the worst-case
  (Hausdorff) deviation from the input stays within the level. Low 0.5 %, Medium 0.2 %,
  High 0.1 %, Ultra 0.05 % of the bounding-box diagonal (or an exact `tolerance` fraction).
* target_faces -> FA-QEM down to about that many faces (no accuracy search).

Input: vertices (n, 3) and triangle faces (m, 3), any scale. Output: vertices and faces in the
same coordinate space (scale / position / orientation unchanged) and an info dict.
Do not edit the files inside faqem/; this adapter is the supported interface.

Engine: when the compiled C++ engine `faqem_cpp` (cpp/ in the FA-QEM project; copy the built
faqem_cpp*.so next to this file) can be imported, it is used; its results are bit-identical to the
Python code in faqem/. FAQEM_BACKEND=python forces the Python engine, FAQEM_BACKEND=cpp requires C++.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from faqem.bvh import BVH  # noqa: E402
from faqem.core import simplify  # noqa: E402
from faqem.metrics import compare_meshes  # noqa: E402
from faqem.pipeline import DETAIL_LEVELS, simplify_auto  # noqa: E402
from faqem.prep import prepare_mesh  # noqa: E402


def _cpp():
    want = os.environ.get("FAQEM_BACKEND", "auto").strip().lower()
    if want == "python":
        return None
    try:
        import faqem_cpp
        return faqem_cpp
    except ImportError:
        if want == "cpp":
            raise
        return None


def _simplify_cpp(fc, vertices, faces, detail, target_faces, tolerance, measure):
    prep = fc.prepare_mesh(vertices, faces)
    if target_faces is not None and int(target_faces) > 0:
        res = fc.simplify(prep.positions, prep.faces, int(target_faces))
        dev = None
        if measure:
            dev = fc.compare_meshes(prep.positions, prep.faces, res["positions"], res["faces"])["hausdorff"]
        mode, within = f"faces={int(target_faces)}", None
    else:
        tol = float(tolerance) if tolerance else DETAIL_LEVELS[detail]
        res, m = fc.simplify_auto(prep, tol)
        dev = m["hausdorff"]
        mode = f"auto {detail} ({tol * 100:g}%)" if not tolerance else f"auto {tol * 100:g}%"
        within = bool(res["stats"]["auto_within"])
    return prep.to_model(res["positions"]), np.asarray(res["faces"], dtype=np.int64), mode, dev, within


class _Mesh:
    """The interface simplify_auto expects (.prep, .bvh)."""

    def __init__(self, vertices, faces):
        self.prep = prepare_mesh(vertices, faces)
        self.bvh = BVH(self.prep.positions, self.prep.faces)


def simplify_mesh_faqem(vertices, faces, detail="Medium", target_faces=None, tolerance=None,
                        measure=True):
    """Simplify a triangle mesh with FA-QEM.

    detail: "Low" | "Medium" | "High" | "Ultra" (Automatic mode, used when target_faces is None)
    target_faces: int > 0 -> FA-QEM face target instead of Automatic
    tolerance: exact allowed worst-case deviation as a fraction of the bbox diagonal (overrides detail)
    measure: also measure the worst-case deviation for the face-target mode (Automatic always does)

    Returns (vertices, faces, info) with info = dict(mode, input_faces, output_faces,
    deviation_percent, within_limit, seconds).
    """
    t0 = time.perf_counter()
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) == 0:
        return vertices, faces, dict(mode="none", input_faces=0, output_faces=0,
                                     deviation_percent=0.0, within_limit=True, seconds=0.0)
    fc = _cpp()
    if fc is not None:
        V, F, mode, dev, within = _simplify_cpp(fc, vertices, faces, detail, target_faces, tolerance, measure)
        info = dict(mode=mode.strip(), input_faces=int(len(faces)), output_faces=int(len(F)),
                    deviation_percent=None if dev is None else round(dev * 100, 4),
                    within_limit=within, seconds=round(time.perf_counter() - t0, 2), engine="cpp")
        return V, F, info
    mesh = _Mesh(vertices, faces)
    prep = mesh.prep
    if target_faces is not None and int(target_faces) > 0:
        res = simplify(prep.positions, prep.faces, int(target_faces))
        dev = None
        if measure:
            dev = compare_meshes(prep.positions, prep.faces, res["positions"], res["faces"],
                                 bvh_a=mesh.bvh)["hausdorff"]
        mode, within = f"faces={int(target_faces)}", None
    else:
        tol = float(tolerance) if tolerance else DETAIL_LEVELS[detail]
        res, m = simplify_auto(mesh, tol)
        dev = m["hausdorff"]
        mode = f"auto {detail} ({tol * 100:g}%)" if not tolerance else f"auto {tol * 100:g}%"
        within = bool(res["stats"]["auto_within"])
    V = prep.to_model(res["positions"])
    F = np.asarray(res["faces"], dtype=np.int64)
    info = dict(mode=mode.strip(), input_faces=int(len(faces)), output_faces=int(len(F)),
                deviation_percent=None if dev is None else round(dev * 100, 4),
                within_limit=within, seconds=round(time.perf_counter() - t0, 2), engine="python")
    return V, F, info
