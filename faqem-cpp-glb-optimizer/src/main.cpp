#include "engine_bridge.hpp"
#include "viewer.hpp"

#include "compat/fmath.hpp"
#include "parallel.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void usage(const char *executable) {
  std::cout << "Usage: " << executable << R"USAGE( [model] [options]
  model                              .glb .gltf .obj .stl .ply .off (none: open the viewer empty)
  --select DIRECTORY                 choose a model from a folder interactively
  -o, --output DIR                   output folder (default: <model dir>/<stem>_faqem)
  --auto [Low|Medium|High|Ultra]     Automatic (best face count) mode (default, High)
  --detail Low|Medium|High|Ultra     detail level for Automatic mode
  --tolerance FRACTION               Automatic mode with an exact max deviation (0.001 = 0.1%)
  --faces N                          best result for N faces (multi-pass search)
  --ratio FRACTION                   same, with N = round(input faces * FRACTION)
  --fast                             with --faces/--ratio: one plain FA-QEM pass
  --preset NAME                      Recommended | "Paper Table 1 (exact)" | "Plain QEM (ablation baseline)"
  --w-area X, --w-boundary X, ...    any FA-QEM option (see engine/src/core.hpp Options)
  --no-color, --no-normal-map        skip the colour / normal-map bake
  --threads N                        cores to use (0 = all; results are identical for any N)
  --no-viewer                        optimize and export without the UI

The engine in engine/ is the same code as our FA-QEM app and CLI, so results are identical.
Viewer: left-drag orbit, wheel zoom, W wireframe, Space original/optimized, R reset.
)USAGE";
}

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

std::filesystem::path select_model(const std::filesystem::path &directory) {
  if (!std::filesystem::is_directory(directory))
    throw std::runtime_error("Model directory does not exist: " + directory.string());
  static const std::set<std::string> extensions = {".glb", ".gltf", ".obj", ".stl", ".ply", ".off"};
  std::vector<std::filesystem::path> files;
  for (const auto &entry : std::filesystem::recursive_directory_iterator(directory))
    if (entry.is_regular_file() && extensions.count(lower(entry.path().extension().string())))
      files.push_back(entry.path());
  std::sort(files.begin(), files.end());
  if (files.empty())
    throw std::runtime_error("No models found in: " + directory.string());

  std::cout << "\nAvailable models in " << directory << ":\n";
  for (std::size_t index = 0; index < files.size(); ++index)
    std::cout << "  " << index + 1 << ") " << std::filesystem::relative(files[index], directory) << '\n';
  std::cout << "\nSelect model [1-" << files.size() << "] or 0 to cancel: " << std::flush;
  std::size_t selection = 0;
  if (!(std::cin >> selection) || selection == 0)
    throw std::runtime_error("Selection cancelled");
  if (selection > files.size())
    throw std::runtime_error("Invalid selection");
  return files[selection - 1];
}

std::string thousands(long long v) {
  std::string s = std::to_string(v), o;
  for (std::size_t i = 0; i < s.size(); i++) {
    if (i && (s.size() - i) % 3 == 0)
      o += ',';
    o += s[i];
  }
  return o;
}

} // namespace

int main(int argc, char **argv) {
  try {
    std::filesystem::path input, output, selection_directory;
    OptimizationOptions options;
    bool viewer = true;
    double ratio = 0.0;
    int modes = 0;  // --auto/--tolerance, --faces, --ratio are exclusive

    for (int index = 1; index < argc; ++index) {
      const std::string argument = argv[index];
      auto next_value = [&]() {
        if (++index >= argc)
          throw std::runtime_error("Missing value after " + argument);
        return std::string(argv[index]);
      };
      if (argument == "-h" || argument == "--help") {
        usage(argv[0]);
        return 0;
      } else if (argument == "--select") {
        selection_directory = next_value();
      } else if (argument == "-o" || argument == "--output") {
        output = next_value();
      } else if (argument == "--auto") {
        options.mode = FaqemMode::Automatic;
        if (index + 1 < argc && faqem::detail_levels().count(argv[index + 1]))
          options.detail = next_value();
        options.tolerance = detail_tolerance(options.detail);
        ++modes;
      } else if (argument == "--detail") {
        options.detail = next_value();
        options.tolerance = detail_tolerance(options.detail);
      } else if (argument == "--tolerance") {
        options.mode = FaqemMode::Automatic;
        options.tolerance = std::stod(next_value());
        options.detail = "Custom";
        ++modes;
      } else if (argument == "--faces") {
        options.target_faces = std::stoll(next_value());
        if (options.mode != FaqemMode::Fast)
          options.mode = FaqemMode::TargetFaces;
        ++modes;
      } else if (argument == "--ratio") {
        ratio = std::stod(next_value());
        if (!std::isfinite(ratio) || ratio <= 0.0 || ratio > 1.0)
          throw std::runtime_error("Ratio must be in (0, 1]");
        if (options.mode != FaqemMode::Fast)
          options.mode = FaqemMode::TargetFaces;
        ++modes;
      } else if (argument == "--fast") {
        options.mode = FaqemMode::Fast;
      } else if (argument == "--preset") {
        options.preset = next_value();
      } else if (argument == "--no-color") {
        options.bake_color = false;
      } else if (argument == "--no-normal-map") {
        options.bake_normal = false;
      } else if (argument == "--threads") {
        faqem::set_num_threads(std::stoi(next_value()));
      } else if (argument == "--no-viewer") {
        viewer = false;
      } else if (argument.rfind("--", 0) == 0) {
        // any FA-QEM option by its Python name, e.g. --w-area 100 --virtual-edges true
        std::string key = argument.substr(2);
        std::replace(key.begin(), key.end(), '-', '_');
        const auto keys = faqem::Options::keys();
        if (std::find(keys.begin(), keys.end(), key) == keys.end() || key == "auto" ||
            key == "max_error" || key == "local_tol")
          throw std::runtime_error("Unknown argument: " + argument);
        const std::string value = next_value();
        const std::string lv = lower(value);
        options.overrides.emplace_back(
            key, faqem::Options::is_bool(key) ? (lv == "1" || lv == "true" || lv == "yes" ? 1.0 : 0.0)
                                               : std::stod(value));
      } else if (input.empty()) {
        input = argument;
      } else {
        throw std::runtime_error("Unknown argument: " + argument);
      }
    }
    if (modes > 1)
      throw std::runtime_error("Use only one of --auto/--tolerance, --faces, --ratio");
    if (options.mode == FaqemMode::Fast && modes == 0)
      throw std::runtime_error("--fast needs --faces or --ratio");

    if (!selection_directory.empty()) {
      if (!input.empty())
        throw std::runtime_error("Use an input file or --select, not both");
      input = select_model(selection_directory);
    }
    if (input.empty()) {
      if (!viewer)
        throw std::runtime_error("An input model is required with --no-viewer");
      run_viewer({}, options, output, false);
      return 0;
    }
    if (!std::filesystem::is_regular_file(input))
      throw std::runtime_error("Input file does not exist: " + input.string());
    if (viewer) {
      if (ratio > 0.0)
        std::cerr << "Note: --ratio applies to --no-viewer; set the face count in the viewer.\n";
      run_viewer(input, options, output, true);
      return 0;
    }

    auto model = load_model(input);
    const auto &s = model.mesh->stats;
    std::printf("loaded %s: %s vertices, %s faces, %s components (%.1fs)\n", input.string().c_str(),
                thousands(s.vertices).c_str(), thousands(s.faces).c_str(),
                thousands(s.components).c_str(), model.mesh->load_time);
    std::printf("engine: C++ (%s build, %d threads)\n", faqem::fm::build_mode(), faqem::num_threads());
    if (ratio > 0.0)
      options.target_faces = static_cast<long long>(std::nearbyint(static_cast<double>(s.faces) * ratio));
    if (output.empty())
      output = default_output_dir(input);
    const auto result = optimize(*model.mesh, options, output, [](double f, const std::string &msg) {
      std::printf("  [%5.1f%%] %s\n", f * 100, msg.c_str());
      std::fflush(stdout);
    });
    const auto &out = result.output;
    auto stats = out.res.stats;
    std::printf("\n%s -> %s faces in %.2fs (bake %.2fs)\n",
                thousands(static_cast<long long>(stats["input_faces"])).c_str(),
                thousands(static_cast<long long>(stats["output_faces"])).c_str(), stats["time_total"],
                out.time_bake);
    if (out.auto_mode)
      std::printf("automatic: max deviation %.3f%%, %s, %zu passes\n", options.tolerance * 100,
                  out.auto_within ? "within the limit" : "NOT within the limit", out.passes.size());
    if (out.best_for_count)
      std::printf("best for this face count: worst deviation %.3f%% (plain FA-QEM %.3f%%), %zu passes\n",
                  out.metrics.hausdorff * 100, out.plain_hausdorff * 100, out.passes.size());
    if (out.has_metrics)
      std::printf("fidelity: hausdorff %.7f, chamfer %.7f, mean_distance %.7f\n", out.metrics.hausdorff,
                  out.metrics.chamfer, out.metrics.mean_distance);
    std::printf("outputs in %s\n", std::filesystem::absolute(output).lexically_normal().string().c_str());
    for (const auto *path : {&out.textured_glb, &out.geometry_glb, &out.geometry_obj, &out.clay_wire_glb,
                             &out.textured_wire_glb, &out.lines_glb})
      if (!path->empty())
        std::printf("  %s\n", std::filesystem::path(*path).filename().string().c_str());
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
