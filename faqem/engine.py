"""Engine selection for the app, the CLI and integrations.

Engines:
  cpp-mt    C++ portable build using all CPU cores (the per-collapse checks are split across cores;
            the result is the same bits as `cpp`). FAQEM_THREADS caps the core count.
  cpp       C++ portable build (cpp/build, module faqem_cpp): the same bits on macOS, Linux and
            Windows; very close to (not bit-equal to) the Python reference.
  cpp-mac   C++ Python-parity build (cpp/build-parity, module faqem_cpp_parity; macOS arm64):
            bit-identical to the Python reference.
  python    the Python reference (this package).

FAQEM_BACKEND=cpp-mt|cpp|cpp-mac|python picks the default (auto: cpp-mt, else cpp-mac, else python).

    from faqem.engine import LoadedMesh, run, original_wireframe
"""
import importlib
import os
import sys

from . import pipeline as _py

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PORTABLE = os.environ.get("FAQEM_CPP_BUILD") or os.path.join(_ROOT, "cpp", "build")
_BUILDS = {  # engine -> (module, build folder)
    "cpp-mt": ("faqem_cpp", _PORTABLE),
    "cpp": ("faqem_cpp", _PORTABLE),
    "cpp-mac": ("faqem_cpp_parity", os.path.join(_ROOT, "cpp", "build-parity")),
}
LABELS = {"cpp-mt": "C++ (all cores)", "cpp": "C++ (portable)", "cpp-mac": "C++ (macOS, identical to Python)",
          "python": "Python"}
_modules = {}


def _load_cpp(engine="cpp"):
    """The C++ module of an engine, or None when it is not built."""
    if engine not in _modules:
        name, folder = _BUILDS[engine]
        mod = None
        try:
            mod = importlib.import_module(name)
        except ImportError:
            if os.path.isdir(folder) and folder not in sys.path:
                sys.path.insert(0, folder)
                try:
                    mod = importlib.import_module(name)
                except ImportError:
                    mod = None
        _modules[engine] = mod
    return _modules[engine]


def available():
    """Engines that can run here, in order of preference."""
    return [e for e in ("cpp-mt", "cpp", "cpp-mac") if _load_cpp(e) is not None] + ["python"]


def backend():
    """The default engine according to FAQEM_BACKEND and what is built."""
    want = os.environ.get("FAQEM_BACKEND", "auto").strip().lower()
    if want in ("python", "cpp-mt", "cpp", "cpp-mac"):
        if want != "python" and _load_cpp(want) is None:
            raise ImportError(f"FAQEM_BACKEND={want} but that C++ build is missing (see cpp/README.md)")
        return want
    return available()[0]


def cpp_build_mode(engine="cpp"):
    """'portable', 'python-parity' or None when that C++ build is missing."""
    cpp = _load_cpp(engine)
    return getattr(cpp, "BUILD_MODE", None) if cpp is not None else None


class _Prep:
    """The parts of faqem.prep.PreparedMesh the app uses, backed by the C++ engine."""

    def __init__(self, cpp_mesh):
        self._m = cpp_mesh
        self._p = None
        self.removed = dict(cpp_mesh.removed)

    def _prepared(self):
        if self._p is None:
            self._p = self._m.prepared()
        return self._p

    @property
    def positions(self):
        return self._prepared().positions

    @property
    def faces(self):
        return self._prepared().faces

    def to_model(self, p):
        return self._prepared().to_model(p)


class _AssetInfo:
    def __init__(self, cpp_mesh):
        self.info = dict(cpp_mesh.asset_info)
        self.has_appearance = bool(cpp_mesh.has_appearance)


class CppLoadedMesh:
    """faqem.pipeline.LoadedMesh interface on top of faqem_cpp(.parity).LoadedMesh."""

    def __init__(self, path, engine="cpp"):
        self.path = path
        self.backend = engine
        self._mod = _load_cpp(engine)
        self._m = self._mod.LoadedMesh(path)
        self.stats = dict(self._m.stats)
        self.load_time = self._m.load_time
        self.prep = _Prep(self._m)
        self.asset = _AssetInfo(self._m)


def LoadedMesh(path, engine=None):
    """engine: 'cpp' | 'cpp-mac' | 'python' | None (= FAQEM_BACKEND / auto)."""
    engine = engine or backend()
    if engine in _BUILDS:
        if _load_cpp(engine) is None:
            raise ImportError(f"the {LABELS[engine]} engine is not built (see cpp/README.md)")
        return CppLoadedMesh(path, engine)
    m = _py.LoadedMesh(path)
    m.backend = "python"
    return m


def run(mesh, target_faces, out_dir, options=None, bake_color=True, bake_normal=True, atlas_size=None,
        crease_angle=60.0, compute_metrics=True, progress=None, auto_tolerance=None, fast=False):
    """faqem.pipeline.run with the same arguments and result keys, on the selected engine."""
    if isinstance(mesh, str):
        mesh = LoadedMesh(mesh)
    if not isinstance(mesh, CppLoadedMesh):
        return _py.run(mesh, target_faces, out_dir, options=options, bake_color=bake_color, bake_normal=bake_normal,
                       atlas_size=atlas_size, crease_angle=crease_angle, compute_metrics=compute_metrics,
                       progress=progress, auto_tolerance=auto_tolerance, fast=fast)
    opts = {k: v for k, v in (options or {}).items() if v is not None}
    _configure_threads(mesh)
    return mesh._m.run(None if auto_tolerance else int(target_faces), out_dir, options=opts, bake_color=bake_color,
                       bake_normal=bake_normal, atlas_size=atlas_size, crease_angle=crease_angle,
                       compute_metrics=compute_metrics,
                       auto_tolerance=float(auto_tolerance) if auto_tolerance else None, fast=fast,
                       progress=progress)


def _configure_threads(mesh):
    """cpp-mt splits each collapse check across all cores; the other C++ engines keep it on one core
    (their metrics and bake use all cores either way). The result does not depend on this."""
    mod = mesh._mod
    if hasattr(mod, "set_threads"):
        mod.set_threads(0)
        mod.set_collapse_parallel(mesh.backend == "cpp-mt")


def describe(mesh):
    """Engine label for display, with the core count for cpp-mt."""
    name = getattr(mesh, "backend", "python")
    if name == "cpp-mt" and hasattr(mesh._mod, "num_threads"):
        return f"C++ (all {mesh._mod.num_threads()} cores)"
    return LABELS[name]


def original_wireframe(mesh, display_glb, out_path, lines_only=False):
    """faqem.pipeline.original_wireframe on the selected engine."""
    if not isinstance(mesh, CppLoadedMesh):
        return _py.original_wireframe(mesh, display_glb, out_path, lines_only=lines_only)
    on_texture = mesh.asset.has_appearance and not lines_only
    color = _py.WIRE_ON_TEXTURE if on_texture else _py.WIRE_ON_CLAY
    p = mesh.prep
    return mesh._mod.add_wireframe(None if lines_only else display_glb, p.to_model(p.positions), p.faces, out_path,
                                     list(color))


DETAIL_LEVELS = _py.DETAIL_LEVELS
