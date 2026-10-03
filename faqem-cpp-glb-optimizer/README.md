# FA-QEM C++ GLB Optimizer

A native macOS viewer and command-line tool for FA-QEM mesh simplification. It opens a model, simplifies it
in the background, shows the original and the result side by side, and writes the same output files as
our FA-QEM app.

The simplification is done by `engine/`. It is an unchanged copy of our C++ FA-QEM engine (the portable
build that our app and CLI use), so **results are identical to ours down to the last bit**: the same face
counts, Automatic search passes, Hausdorff/Chamfer values and output files. The build checks this (see
*Tests*).

## What changed from the first version of this folder

The first version had its own partial FA-QEM in `src/optimizer.cpp`, `native_bvh.cpp` and `glb_io.cpp`.
Its face counts and outputs did not match ours. It was missing the adaptive 1–4 pass Automatic search,
mesh repair (non-manifold split, reorientation), virtual edges, boundary validation, Chamfer and mean
distance, the texture bake, and formats other than GLB. Those files are replaced by:

| Before | Now |
|---|---|
| `optimizer.cpp`, `native_bvh.cpp` (own FA-QEM) | `engine/`: our engine, all of `faqem/` except the web UI |
| `glb_io.cpp` (tinygltf) | the engine's loaders: GLB/glTF (with Draco), OBJ+MTL, STL, PLY, OFF |
| one geometry `.glb` | the full output set, including baked colour, normal and PBR textures |
| Automatic = one local-tolerance pass | Automatic (best face count): up to 4 passes, final measured Hausdorff |
| Target faces = one pass | best result for a face count (up to 6 passes), or `--fast` for one pass |
| Eigen, tinygltf | not needed |

The viewer (`src/viewer.cpp`) keeps its window, camera, keys and statistics table. It now calls the engine
through `src/engine_bridge.cpp`.

## Build (macOS)

Requirements: CMake 3.21+, Xcode command-line tools (clang), and internet access the first time. The
first build downloads GLFW, Dear ImGui and the engine's pinned image/Draco libraries.

```bash
cd faqem-cpp-glb-optimizer
./build.sh                 # build, run the tests, open the viewer
./build.sh --build-only    # build and test only
```

The first build also compiles the pinned libraries into `build-deps/` (about 2 minutes, once). To reuse
an existing install, set `FAQEM_DEPS=/path/to/install`.

This builds two programs:

- `build/faqem_optimizer`: the viewer, which also has a batch mode
- `build/engine/faqem`: the engine's own command line, with the same flags and report as our `cli.py`

## Viewer

```bash
./build/faqem_optimizer                      # empty window; click "Open model..."
./build/faqem_optimizer model.glb            # opens the model and optimizes it (Automatic, High)
./build/faqem_optimizer model.glb --faces 5000
```

Settings in the panel:

- **Target**:
  - *Automatic (best face count)* with a detail level: Low 0.5%, Medium 0.2%, High 0.1%, Ultra 0.05% of
    the bounding-box diagonal, or a custom limit
  - *Face count (best result)*
  - *Face count (fast, one pass)*
- **Preset**: Recommended (default), Paper Table 1 (exact), Plain QEM.
- **Bake colour** / **Bake normal map**.

Loading and optimizing run in the background, and the progress bar shows the engine's own steps. When a
run finishes, the panel shows:

- the original and optimized counts
- Hausdorff, Chamfer and mean distance
- the search passes
- the output files, plus an **Open output folder** button

**Stop** discards the running result. The engine can't be interrupted part-way through a pass, so the
current pass finishes in the background before you can optimize again.

Keys: left-drag orbit, wheel zoom, `W` wireframe, `Space` original/optimized, `R` reset camera, `Esc` quit.

## Batch mode

```bash
./build/faqem_optimizer model.glb --no-viewer                      # Automatic, High
./build/faqem_optimizer model.glb --no-viewer --detail Medium
./build/faqem_optimizer model.glb --no-viewer --tolerance 0.003    # custom limit, 0.3%
./build/faqem_optimizer model.glb --no-viewer --faces 20000        # best result for 20,000 faces
./build/faqem_optimizer model.glb --no-viewer --ratio 0.05 --fast  # one pass to 5% of the faces
./build/faqem_optimizer --select ~/models --no-viewer              # pick a model from a folder
```

Other options:

- `-o DIR`: output folder (default `<model dir>/<stem>_faqem/`)
- `--preset NAME`
- `--no-color`, `--no-normal-map`
- `--threads N`: results are identical for any N
- any FA-QEM option by name, e.g. `--w-area 100`, `--w-boundary 500`, `--virtual-edges true`

Run `--help` for the full list.

These flags changed meaning from the first version:

- `-o` is now an output **folder**
- `--tolerance` starts Automatic mode with an exact limit
- the default detail is High, as in our app

Each run writes:

- `<stem>_lowpoly_textured.glb`: low-poly mesh with baked colour, normal map and metallic-roughness when
  the source has them
- `<stem>_lowpoly_geometry.glb` and `<stem>_lowpoly.obj`: plain geometry
- `<stem>_lowpoly_wireframe.glb`, `<stem>_lowpoly_textured_wireframe.glb`, `<stem>_lowpoly_lines.glb`:
  wireframes

## Tests

`ctest --test-dir build --output-on-failure` (`build.sh` runs it):

- `engine_smoke_test`: prepare, simplify and the Automatic search on small synthetic meshes
- `glb_roundtrip_test`: GLB export → load → full run (search, bake, every output) → reload
- `golden_digest_test`: runs `engine/faqem --digest` on `tests/data/` (the dresser model, Automatic
  Medium) and compares 12 SHA-256 digests (result arrays, search passes, metrics and every output file)
  with those recorded by our engine. If it fails, the results no longer match ours.

## Notes

- Do not edit `engine/`. It must stay identical to our engine; see `PORTING_STATUS.md`.
- The viewer is macOS-only (Cocoa file dialog). The engine itself also builds on Linux and Windows.
- The viewer draws flat-coloured geometry. Textures are in the exported `_textured.glb`.
