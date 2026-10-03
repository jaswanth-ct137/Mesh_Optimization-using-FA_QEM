# Engine status

`engine/` is a copy of our FA-QEM C++ engine (`cpp/` in the FA-QEM repository), taken on 2026-10-03:

| Here | Source | Changes |
|---|---|---|
| `engine/src/` | `cpp/src/` | none (only `compat/accelerate.cpp` left out, which only the macOS Python-parity build uses) |
| `engine/third_party/` | `cpp/third_party/` (scipy cKDTree, CORE-MATH) | none |
| `engine/deps/CMakeLists.txt` | `cpp/deps/CMakeLists.txt` | none |
| `engine/cli/main.cpp` | `cpp/cli/main.cpp` | none |
| `engine/CMakeLists.txt` | `cpp/CMakeLists.txt` | Python module off by default |
| `engine/tests/golden_portable.json` | `cpp/tests/golden_portable.json` | none |

The engine ports the whole Python `faqem/` package except the web UI:

- loading
- preparation and repair
- FA-QEM simplification, virtual edges
- the Automatic and best-for-count searches
- metrics
- the texture/normal/PBR bake
- export and wireframes

## Verified

- `golden_digest_test` in this folder: dresser model, Automatic Medium. All 12 digests are identical to
  our engine's.
- Our full corpus (`tools/portable_golden.py --faqem build/engine/faqem`): all 36 runs identical. That
  covers 7 models × 4 detail levels, face-count and `--fast` runs, the Paper Table 1 preset and OBJ inputs.
- `faqem_optimizer --no-viewer` against our CLI on the boots model (Automatic High, `--faces 3000`,
  `--ratio 0.05 --fast`): the same face counts and metrics, and byte-identical output files.

## Updating the engine

Copy `cpp/src`, `cpp/third_party`, `cpp/deps` and `cpp/cli` from our repository over `engine/` again.
Keep `compat/accelerate.cpp` out. Then rebuild and run the tests. Do not change files in `engine/` here
or its compiler flags (`-ffp-contract=off`, no fast-math). Either change would make results differ from
ours.
