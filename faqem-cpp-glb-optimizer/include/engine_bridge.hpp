#pragma once
// Connects the viewer to the FA-QEM engine in engine/ (the same engine as our app and CLI).

#include "mesh_data.hpp"

#include "pipeline.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

enum class FaqemMode { Automatic, TargetFaces, Fast };

struct OptimizationOptions {
  FaqemMode mode = FaqemMode::Automatic;
  std::string detail = "High";  // Low | Medium | High | Ultra | Custom
  double tolerance = 0.001;     // Automatic: max deviation, fraction of the bbox diagonal
  long long target_faces = 20000;
  std::string preset = "Recommended";
  std::vector<std::pair<std::string, double>> overrides;  // FA-QEM options by Python name
  bool bake_color = true;
  bool bake_normal = true;
};

struct LoadedModel {
  std::shared_ptr<faqem::LoadedMesh> mesh;
  MeshScene scene;  // for display, model space
};

struct OptimizationResult {
  MeshScene scene;  // for display, model space
  faqem::RunOutput output;
  std::filesystem::path out_dir;
};

// load any supported file (glb, gltf, obj, stl, ply, off)
LoadedModel load_model(const std::filesystem::path &path);

// detail level -> tolerance (faqem::detail_levels)
double detail_tolerance(const std::string &detail);

// default output folder: <model dir>/<stem>_faqem
std::filesystem::path default_output_dir(const std::filesystem::path &input);

// faqem::run with the viewer's options; writes every output file into out_dir
OptimizationResult optimize(faqem::LoadedMesh &mesh, const OptimizationOptions &options,
                            const std::filesystem::path &out_dir,
                            const faqem::Progress &progress = nullptr);
