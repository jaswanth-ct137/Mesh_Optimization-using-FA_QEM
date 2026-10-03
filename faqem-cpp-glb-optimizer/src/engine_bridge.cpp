#include "engine_bridge.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

float display_channel(double linear) {
  return static_cast<float>(std::pow(std::clamp(linear, 0.0, 1.0), 1.0 / 2.2));
}

// one display part per material; positions and faces in model space
MeshScene to_scene(const faqem::VecD &positions, const faqem::VecI &faces,
                   const faqem::VecI &face_material,
                   const std::vector<faqem::Material> &materials) {
  MeshScene scene;
  const std::size_t nV = positions.size() / 3, nF = faces.size() / 3;
  const std::size_t nM = std::max<std::size_t>(materials.size(), 1);
  std::vector<std::vector<std::uint32_t>> indices(nM);
  for (std::size_t f = 0; f < nF; ++f) {
    std::size_t m = f < face_material.size() ? static_cast<std::size_t>(face_material[f]) : 0;
    if (m >= nM)
      m = 0;
    for (int c = 0; c < 3; ++c)
      indices[m].push_back(static_cast<std::uint32_t>(faces[3 * f + c]));
  }
  std::vector<Vertex> vertices(nV);
  for (std::size_t v = 0; v < nV; ++v)
    vertices[v].position = {positions[3 * v], positions[3 * v + 1], positions[3 * v + 2]};
  for (std::size_t m = 0; m < nM; ++m) {
    if (indices[m].empty())
      continue;
    MeshPart part;
    part.name = "material " + std::to_string(m);
    part.vertices = vertices;
    part.indices = std::move(indices[m]);
    if (m < materials.size()) {
      const auto &c = materials[m].color;
      part.color = {display_channel(c[0]), display_channel(c[1]), display_channel(c[2]), 1.0f};
    }
    scene.parts.push_back(std::move(part));
  }
  return scene;
}

}  // namespace

LoadedModel load_model(const std::filesystem::path &path) {
  LoadedModel model;
  model.mesh = std::make_shared<faqem::LoadedMesh>(path.string());
  const auto &asset = model.mesh->asset;
  model.scene = to_scene(asset.positions, asset.faces, asset.face_material, asset.materials);
  return model;
}

double detail_tolerance(const std::string &detail) {
  const auto &levels = faqem::detail_levels();
  const auto it = levels.find(detail);
  if (it == levels.end())
    throw std::runtime_error("Detail must be Low, Medium, High, or Ultra");
  return it->second;
}

std::filesystem::path default_output_dir(const std::filesystem::path &input) {
  return input.parent_path() / (input.stem().string() + "_faqem");
}

OptimizationResult optimize(faqem::LoadedMesh &mesh, const OptimizationOptions &options,
                            const std::filesystem::path &out_dir,
                            const faqem::Progress &progress) {
  faqem::RunOptions ro;
  ro.options = faqem::preset(options.preset);
  for (const auto &[key, value] : options.overrides)
    if (!ro.options.set(key, value))
      throw std::runtime_error("Unknown FA-QEM option: " + key);
  ro.bake_color = options.bake_color;
  ro.bake_normal = options.bake_normal;
  if (options.mode == FaqemMode::Automatic) {
    if (!(options.tolerance > 0.0))
      throw std::runtime_error("Max deviation must be positive");
    ro.auto_tolerance = options.tolerance;
  } else {
    if (options.target_faces <= 0)
      throw std::runtime_error("Target faces must be positive");
    ro.target_faces = options.target_faces;
    ro.fast = options.mode == FaqemMode::Fast;
  }

  OptimizationResult result;
  result.out_dir = out_dir;
  result.output = faqem::run(mesh, out_dir.string(), ro, progress);
  const auto &res = result.output.res;
  // the low-poly mesh has a single material; show it in the first material's colour
  std::vector<faqem::Material> materials(1);
  if (!mesh.asset.materials.empty())
    std::copy(mesh.asset.materials[0].color, mesh.asset.materials[0].color + 4, materials[0].color);
  result.scene = to_scene(mesh.prep.to_model(res.positions), res.faces, {}, materials);
  return result;
}
