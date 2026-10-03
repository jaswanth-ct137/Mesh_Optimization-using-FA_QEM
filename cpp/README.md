# FA-QEM C++ engine

This is a C++ port of the whole `faqem/` package except the web UI. That includes loading (GLB/glTF with
Draco, OBJ+MTL, STL, PLY, OFF), preparation, FA-QEM simplification, the **Automatic (best face count)**
and best-for-a-face-count searches, Hausdorff/Chamfer metrics, the texture/normal/PBR bake, GLB/OBJ
export and the wireframes.

## Two build modes

| | **portable** (default) | **python-parity** |
|---|---|---|
| runs on | macOS, Linux, Windows | macOS arm64 only |
| identical bits | on every OS (checked with golden digests) | with the Python reference on this Mac |
| vs Python | very close (same face counts on the SKUs, rare few-face differences possible) | bit-identical |
| math | correctly rounded CORE-MATH, fixed-order linear algebra | the platform libm and Apple Accelerate, replaying numpy |
| codecs | pinned versions built from source (`cpp/deps`) | Homebrew (the versions Pillow bundles) |
| build folder used here | `cpp/build` (what the app uses) | `cpp/build-parity` |

## Build: portable (macOS / Linux; Windows: see `build_windows.md`)

```bash
cmake -S cpp/deps -B cpp/build-deps -G Ninja && cmake --build cpp/build-deps      # pinned libraries, once
cmake -S cpp -B cpp/build -G Ninja -DFAQEM_DEPS=$PWD/cpp/build-deps/install \
      -DPython3_EXECUTABLE=$PWD/.venv/bin/python -DCMAKE_PREFIX_PATH=$(brew --prefix)   # pybind11
cmake --build cpp/build
python3 tools/portable_golden.py        # must print "all 36 runs identical"
```

The result is `cpp/build/faqem`. It depends only on the system C/C++ runtime: the codecs are linked in
statically.

## Build: python-parity (macOS arm64)

```bash
brew install cmake ninja pybind11 nlohmann-json draco jpeg-turbo webp libpng
cmake -S cpp -B cpp/build-parity -G Ninja -DFAQEM_PYTHON_PARITY=ON -DCMAKE_PREFIX_PATH=$(brew --prefix) \
      -DPython3_EXECUTABLE=$PWD/.venv/bin/python -DPYBIND11_FINDPYTHON=ON
cmake --build cpp/build-parity
```

Use it from Python with `FAQEM_CPP_BUILD=$PWD/cpp/build-parity`.

## Use

```bash
cpp/build/faqem model.glb --auto High -o out/          # Automatic, max deviation 0.1 %
cpp/build/faqem scan.stl --faces 5000 --no-normal-map  # best result for 5,000 faces
cpp/build/faqem model.glb --faces 5000 --fast          # one plain FA-QEM pass
cpp/build/faqem model.glb --w-plane-area 1 --virtual-edges true   # Paper Table 1 (exact)
```

`app.py`, `cli.py` and `handoff_triposg/faqem_for_triposg/faqem_adapter.py` use the C++ engine
automatically once it is built, through `faqem/engine.py`. Set `FAQEM_BACKEND=python` to force the
Python reference or `FAQEM_BACKEND=cpp` to require C++. The app shows the engine in use under the
upload summary.

From Python:

```python
import faqem_cpp
m = faqem_cpp.LoadedMesh("model.glb")
out = m.run(None, "out/", auto_tolerance=0.001)                # same result dict as pipeline.run
prep = faqem_cpp.prepare_mesh(V, F)
res, metrics = faqem_cpp.simplify_auto(prep, 0.002)          # == pipeline.simplify_auto
```

For TripoSG, copy `faqem_cpp*.so` next to `faqem_adapter.py`. `simplify_mesh_faqem()` keeps its
signature and reports `info["engine"]`.

## Using all CPU cores

`faqem --threads N` sets the core count; the default 0 means all cores. `FAQEM_THREADS=N` caps it, for
example on a shared server. From Python, use `faqem_cpp.set_threads(n)`.

The simplifier keeps one global collapse order. Only the work inside each collapse's local check is
split across cores: the tracked points against the new triangles, and the samples on the new triangles
against the original surface. Each check is an AND of independent tests, so **the result is the same
bits for any core count**; `tools/portable_golden.py --threads 1/3/0` checks this. The metrics and the
bake also use all cores.

Measured on this 8-core M2 (Automatic High):

| mesh | 1 core | all 8 cores |
|---|---|---|
| boots, 129k faces | 26.4 s | 13.5 s (about 2×) |
| koala, 1.79M faces | 58.9 s | 54.0 s (about 1.1×) |

The koala gains little: above 400k faces only the original vertices are tracked, so each check is a
handful of tests (about 31 µs per collapse) and too small to split. More cores (e.g. 16) help most on
meshes like boots.

App and `cli.py --engine`:

| choice | engine |
|---|---|
| `cpp-mt` | C++ (all cores) |
| `cpp` | C++ (portable): one core for the checks |
| `cpp-mac` | C++ (macOS, identical to Python) |
| `python` | Python |

All four give the same faces and numbers.

## Verifying

```bash
python3 tools/portable_golden.py                          # portable: same bits as the macOS golden
.venv/bin/python tools/compare_backends.py --approx --cpp cpp/build/faqem   # portable vs Python
.venv/bin/python -m pytest -q tests/test_cpp_parity.py   # parity build: ~230 bit-exact checks, ~30 min
.venv/bin/python tools/compare_backends.py               # both CLIs on the 7 SKUs x 4 detail levels
```

The Python side must be the pinned environment in `requirements-lock.txt`.

## How the python-parity build matches Python (rules for anyone editing this code)

The Python reference is not plain IEEE scalar code everywhere, so the C++ reproduces each library's
exact arithmetic.

* **No FMA, except where Python has one.** The project builds with `-ffp-contract=off` and without
  fast-math. Numba never fuses, but clang would by default. `np.einsum('ij,ij->i')` is
  `fma(a2,b2,a0*b0)+a1*b1` (`np::einsum3`), and scipy's sparse product in trimesh's vertex normals fuses,
  so those places call `std::fma` explicitly.
* **`pow`.** Python/numpy scalar `**2` and array `**3` / `**2.4` call libm `pow`, which is not `x*x`.
  Use `np::py_pow`. Numba `x**2` and numpy array `**2` are `x*x`.
* **numpy reductions.**
  * 1-D sum/mean: pairwise summation in 8192-element chunks (`np::sum`).
  * `norm(axis=1)`: `sqrt((x²+y²)+z²)`.
  * min/max: order −0 below +0 (`np::fmin0`/`fmax0`).
  * `np.round`: rounds half to even.
* **Sorting.** The default `argsort` is numpy's introsort, ported in `np::argsort_quick` (ties matter).
  `kind="stable"` and `np.unique` are stable sorts.
* **scipy.**
  * csgraph label order and BFS order: `compat/csgraph`.
  * cKDTree: the vendored scipy 1.15.0 sources in `third_party/ckdtree`, compiled like the scipy wheel.
* **numpy RNG.** SeedSequence, PCG64, `random`, `choice(p)` and `permutation` are reimplemented in
  `compat/nprandom`. They are used by the metrics sampling and the bake.
* **Accelerate.** trimesh's transform maths (`np.dot`, `V @ T.T`, `fix_rigid`'s SVD, `unitize`) runs
  on Apple's BLAS/LAPACK. `compat/accelerate` makes the same calls with numpy's arguments and even
  numpy's buffer layout, because alignment changes Accelerate's results.
* **Loaders and decoders.**
  * Loaders reproduce trimesh 5.1.0 rule by rule: node order, float32 UV flip, colour quantisation,
    OBJ/PLY vertex splitting, and OFF's comment and polygon quirks.
  * JPEG uses libjpeg-turbo and WebP uses libwebp 1.6, the same decoders Pillow bundles.
* **Threads.** Only the per-point metrics and the per-chart bake are multithreaded. Both are
  order-independent. The collapse loop stays sequential, as in Python.

## How the portable build stays identical across systems

* **Transcendental functions.** `atan2`, `acos`, `cos` and `pow` are CORE-MATH's correctly rounded
  functions (`third_party/core-math`, MIT). They were checked against MPFR: 0 errors on 1.6 M inputs.
  Apple's libm differs from the correct result on up to 14.5 % of inputs, and glibc and Windows differ
  on other inputs.
* **Linear algebra.** `compat/linalg_portable.cpp` uses fixed-order scalar code and replaces Accelerate.
  `fix_rigid` uses a deterministic polar decomposition instead of LAPACK's SVD.
* **Sorting.** The KD-tree build uses `std::stable_partition` and `std::stable_sort`, whose results
  are fully specified. Other sorts are stable or sort unique keys.
* **Text.** `std::to_chars` writes the `.obj` numbers.
* **Dependencies.** They are pinned and built from source with the same flags and with SIMD paths
  off.
* **Compiler.** The compiler is gcc or clang (clang-cl on Windows) with `-ffp-contract=off` and no
  fast-math.

## Not supported in C++ (the loader raises a clear error)

* glTF `KHR_materials_pbrSpecularGlossiness` (deprecated; trimesh converts it with Pillow resampling)
* PLY files with texture coordinates or a texture file
* 3MF and DAE files

## Speed

* The collapse loop runs about 1.3–1.5× faster than the Numba code.
* Metrics and bake are multithreaded.
* There is no Numba JIT warm-up, which takes 30–60 s on a cold cache.
