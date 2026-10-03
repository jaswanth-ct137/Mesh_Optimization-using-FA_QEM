// GLB export -> load -> full FA-QEM run (search, bake, every output file) -> reload.
#include "engine_bridge.hpp"

#include "io/export.hpp"

#include <cassert>
#include <filesystem>
#include <iostream>

int main() {
  const faqem::VecD P = {1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1};
  const faqem::VecI F = {0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5};
  const auto dir = std::filesystem::temp_directory_path() / "faqem_glb_roundtrip";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const auto path = dir / "octahedron.glb";
  faqem::export_geometry(path.string(), P, F);
  assert(std::filesystem::file_size(path) > 20);

  auto model = load_model(path);
  assert(model.mesh->stats.vertices == 6);
  assert(model.mesh->stats.faces == 8);
  assert(display_stats(model.scene).triangles == 8);

  OptimizationOptions options;
  options.mode = FaqemMode::TargetFaces;
  options.target_faces = 4;
  const auto out_dir = dir / "out";
  const auto result = optimize(*model.mesh, options, out_dir);
  const auto &out = result.output;
  assert(out.best_for_count);
  assert(result.output.res.output_faces() > 0);
  for (const auto *file : {&out.geometry_glb, &out.geometry_obj, &out.textured_glb, &out.clay_wire_glb,
                           &out.lines_glb})
    assert(!file->empty() && std::filesystem::exists(*file));
  assert(load_model(out.geometry_glb).mesh->stats.faces == out.res.output_faces());
  std::filesystem::remove_all(dir);
  std::cout << "GLB export/optimization/round-trip passed\n";
}
