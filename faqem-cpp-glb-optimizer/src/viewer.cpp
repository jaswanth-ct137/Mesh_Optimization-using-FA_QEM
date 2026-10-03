#include "viewer.hpp"
#include "file_dialog.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl2.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>

namespace {
struct State {
  const MeshScene *original;
  const MeshScene *optimized;
  bool showOptimized = true;
  bool wireframe = false;
  bool dragging = false;
  double lastX = 0.0;
  double lastY = 0.0;
  float yaw = 25.0f;
  float pitch = -20.0f;
  float zoom = 1.0f;
};

struct GeometryStats {
  std::size_t parts = 0, vertices = 0, edges = 0, triangles = 0;
};

// One engine call (load or optimize) on a background thread. Only one runs at a time: the engine's
// LoadedMesh builds its BVH lazily. The engine cannot be interrupted, so Stop abandons the job and the
// thread finishes on its own.
struct Job {
  std::mutex mutex;
  double fraction = 0.0;
  std::string message;
  bool done = false;
  std::atomic_bool abandoned{false};
  std::optional<LoadedModel> loaded;
  std::optional<OptimizationResult> result;
  std::filesystem::path input;
  std::string error;
};

GeometryStats geometry_stats(const MeshScene &scene) {
  GeometryStats result;
  result.parts = scene.parts.size();
  std::unordered_set<std::uint64_t> edges;
  for (const auto &part : scene.parts) {
    // parts of one model share vertex numbering, so edges are counted across parts
    result.triangles += part.indices.size() / 3;
    for (std::size_t i = 0; i + 2 < part.indices.size(); i += 3) {
      const std::uint32_t triangle[3] = {part.indices[i], part.indices[i + 1], part.indices[i + 2]};
      for (int e = 0; e < 3; ++e) {
        const auto a = std::min(triangle[e], triangle[(e + 1) % 3]);
        const auto b = std::max(triangle[e], triangle[(e + 1) % 3]);
        edges.insert((static_cast<std::uint64_t>(a) << 32U) | b);
      }
    }
  }
  result.edges = edges.size();
  if (!scene.parts.empty())
    result.vertices = scene.parts[0].vertices.size();
  return result;
}

double reduction(std::size_t before, std::size_t after) {
  return before == 0 ? 0.0 : 100.0 * (1.0 - static_cast<double>(after) / before);
}

void key(GLFWwindow *window, int key, int, int action, int) {
  if (action != GLFW_PRESS)
    return;
  if (ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureKeyboard)
    return;
  auto *state = static_cast<State *>(glfwGetWindowUserPointer(window));
  if (key == GLFW_KEY_ESCAPE)
    glfwSetWindowShouldClose(window, 1);
  if (key == GLFW_KEY_W)
    state->wireframe = !state->wireframe;
  if (key == GLFW_KEY_SPACE)
    state->showOptimized = !state->showOptimized;
  if (key == GLFW_KEY_R) {
    state->yaw = 25.0f;
    state->pitch = -20.0f;
    state->zoom = 1.0f;
  }
}

void scroll(GLFWwindow *window, double, double y) {
  if (ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureMouse)
    return;
  auto *state = static_cast<State *>(glfwGetWindowUserPointer(window));
  state->zoom = std::clamp(state->zoom * static_cast<float>(std::pow(0.9, y)), 0.1f, 10.0f);
}

void update_orbit(GLFWwindow *window, State &state) {
  double x, y;
  glfwGetCursorPos(window, &x, &y);
  const bool down = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
  if (down && !ImGui::GetIO().WantCaptureMouse) {
    if (state.dragging) {
      state.yaw += static_cast<float>(x - state.lastX) * 0.35f;
      state.pitch =
          std::clamp(state.pitch + static_cast<float>(y - state.lastY) * 0.35f, -89.0f, 89.0f);
    }
    state.dragging = true;
  } else {
    state.dragging = false;
  }
  state.lastX = x;
  state.lastY = y;
}

void bounds(const MeshScene &scene, double *center, double &radius) {
  double low[3] = {1e300, 1e300, 1e300}, high[3] = {-1e300, -1e300, -1e300};
  for (const auto &part : scene.parts)
    for (const auto &vertex : part.vertices)
      for (int j = 0; j < 3; ++j) {
        low[j] = std::min(low[j], vertex.position[j]);
        high[j] = std::max(high[j], vertex.position[j]);
      }
  radius = 0.0;
  for (int j = 0; j < 3; ++j) {
    center[j] = (low[j] + high[j]) * 0.5;
    radius = std::max(radius, (high[j] - low[j]) * 0.5);
  }
  if (radius <= 0.0)
    radius = 1.0;
}

void stats_column(const char *title, const GeometryStats &stats) {
  ImGui::TextUnformatted(title);
  ImGui::Separator();
  ImGui::Text("Parts:     %zu", stats.parts);
  ImGui::Text("Vertices:  %zu", stats.vertices);
  ImGui::Text("Edges:     %zu", stats.edges);
  ImGui::Text("Triangles: %zu", stats.triangles);
}

const char *const kDetails[] = {"Low", "Medium", "High", "Ultra", "Custom"};
const char *const kPresets[] = {"Recommended", "Paper Table 1 (exact)",
                                "Plain QEM (ablation baseline)"};

int index_of(const char *const *items, int count, const std::string &value, int fallback) {
  for (int i = 0; i < count; ++i)
    if (value == items[i])
      return i;
  return fallback;
}
} // namespace

void run_viewer(const std::filesystem::path &input, OptimizationOptions options,
                std::filesystem::path out_dir, bool optimize_on_open) {
  if (!glfwInit())
    throw std::runtime_error("GLFW initialization failed");
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
  GLFWwindow *window = glfwCreateWindow(1280, 800, "FA-QEM C++ Mesh Optimizer", nullptr, nullptr);
  if (!window) {
    glfwTerminate();
    throw std::runtime_error("OpenGL window creation failed");
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  std::shared_ptr<faqem::LoadedMesh> mesh;
  std::filesystem::path modelPath;
  MeshScene original, optimized;
  std::optional<OptimizationResult> result;
  State state{&original, &optimized};
  GeometryStats originalStats, optimizedStats;

  int mode = options.mode == FaqemMode::Automatic ? 0 : options.mode == FaqemMode::TargetFaces ? 1 : 2;
  int detail = index_of(kDetails, 5, options.detail, 4);
  double tolerancePercent = options.tolerance * 100.0;
  int targetFaces = static_cast<int>(std::min<long long>(options.target_faces, 1 << 30));
  int preset = index_of(kPresets, 3, options.preset, 0);
  bool bakeColor = options.bake_color, bakeNormal = options.bake_normal;
  const std::filesystem::path requestedOutDir = out_dir;

  std::shared_ptr<Job> job;       // running job whose outcome is shown
  std::shared_ptr<Job> draining;  // stopped job still finishing in the background
  bool jobIsLoad = false;
  bool optimizeAfterLoad = false;
  std::chrono::steady_clock::time_point jobStart;
  std::string operationStatus = "Open a model to begin.";

  auto start_load = [&](const std::filesystem::path &path, bool thenOptimize) {
    job = std::make_shared<Job>();
    job->input = path;
    job->message = "Loading and preparing " + path.filename().string();
    jobIsLoad = true;
    optimizeAfterLoad = thenOptimize;
    jobStart = std::chrono::steady_clock::now();
    operationStatus = job->message;
    std::thread([job = job, path] {
      try {
        auto loaded = load_model(path);
        std::lock_guard<std::mutex> lock(job->mutex);
        job->loaded = std::move(loaded);
      } catch (const std::exception &error) {
        std::lock_guard<std::mutex> lock(job->mutex);
        job->error = error.what();
      }
      std::lock_guard<std::mutex> lock(job->mutex);
      job->done = true;
    }).detach();
  };

  auto start_optimize = [&] {
    options.mode = mode == 0 ? FaqemMode::Automatic : mode == 1 ? FaqemMode::TargetFaces : FaqemMode::Fast;
    options.detail = kDetails[detail];
    options.tolerance =
        detail < 4 ? detail_tolerance(options.detail) : tolerancePercent / 100.0;
    options.target_faces = targetFaces;
    options.preset = kPresets[preset];
    options.bake_color = bakeColor;
    options.bake_normal = bakeNormal;
    out_dir = requestedOutDir.empty() ? default_output_dir(modelPath) : requestedOutDir;
    job = std::make_shared<Job>();
    job->message = "Starting FA-QEM";
    jobIsLoad = false;
    jobStart = std::chrono::steady_clock::now();
    operationStatus = "FA-QEM optimization is running in the background...";
    std::thread([job = job, mesh = mesh, requested = options, dir = out_dir] {
      try {
        auto output = optimize(*mesh, requested, dir, [&job](double fraction, const std::string &msg) {
          std::lock_guard<std::mutex> lock(job->mutex);
          job->fraction = fraction;
          job->message = msg;
        });
        std::lock_guard<std::mutex> lock(job->mutex);
        job->result = std::move(output);
      } catch (const std::exception &error) {
        std::lock_guard<std::mutex> lock(job->mutex);
        job->error = error.what();
      }
      std::lock_guard<std::mutex> lock(job->mutex);
      job->done = true;
    }).detach();
  };

  glfwSetWindowUserPointer(window, &state);
  glfwSetKeyCallback(window, key);
  glfwSetScrollCallback(window, scroll);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL2_Init();
  glEnable(GL_DEPTH_TEST);
  glClearColor(0.055f, 0.065f, 0.085f, 1.0f);

  if (!input.empty())
    start_load(input, optimize_on_open);

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    ImGui_ImplOpenGL2_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    update_orbit(window, state);

    if (draining) {
      std::lock_guard<std::mutex> lock(draining->mutex);
      if (draining->done) {
        draining.reset();
        operationStatus = "Stopped. The abandoned run has finished; you can optimize again.";
      }
    }
    if (job) {
      std::unique_lock<std::mutex> lock(job->mutex);
      if (job->done) {
        const auto finished = job;
        job.reset();
        lock.unlock();
        if (!finished->error.empty()) {
          operationStatus = (jobIsLoad ? "Open failed: " : "Optimization failed: ") + finished->error;
        } else if (jobIsLoad) {
          mesh = finished->loaded->mesh;
          modelPath = finished->input;
          original = std::move(finished->loaded->scene);
          optimized = original;
          result.reset();
          state.showOptimized = false;
          originalStats = geometry_stats(original);
          optimizedStats = originalStats;
          operationStatus = "Model loaded. Choose a mode, then click Optimize.";
          if (optimizeAfterLoad)
            start_optimize();
        } else {
          result = std::move(finished->result);
          optimized = std::move(result->scene);
          optimizedStats = geometry_stats(optimized);
          state.showOptimized = true;
          operationStatus = "Done. Every output file is in the output folder.";
        }
      }
    }
    const bool busy = job || draining;

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    const auto &scene = state.showOptimized ? optimized : original;
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    const double aspect = height ? static_cast<double>(width) / height : 1.0;
    const double nearPlane = 0.01, farPlane = 100.0;
    const double top = nearPlane * std::tan(45.0 * 3.141592653589793 / 360.0);
    glFrustum(-top * aspect, top * aspect, -top, top, nearPlane, farPlane);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    if (!scene.parts.empty()) {
      double center[3], radius;
      bounds(scene, center, radius);
      glTranslated(0, 0, -3.2 * state.zoom);
      glRotated(state.pitch, 1, 0, 0);
      glRotated(state.yaw, 0, 1, 0);
      glScaled(1 / radius, 1 / radius, 1 / radius);
      glTranslated(-center[0], -center[1], -center[2]);
      glPolygonMode(GL_FRONT_AND_BACK, state.wireframe ? GL_LINE : GL_FILL);
      for (const auto &part : scene.parts) {
        glColor4fv(part.color.data());
        glBegin(GL_TRIANGLES);
        for (auto index : part.indices) {
          if (index >= part.vertices.size()) continue;
          const auto &p = part.vertices[index].position;
          glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();
      }
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    ImGui::SetNextWindowPos(ImVec2(16, 16), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(480, 760), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(360, 360), ImVec2(900, 1200));
    ImGui::SetNextWindowBgAlpha(0.92f);
    ImGui::Begin("FA-QEM Mesh Optimizer", nullptr, ImGuiWindowFlags_NoCollapse);
    ImGui::BeginDisabled(busy);
    if (ImGui::Button("Open model...", ImVec2(-1.0f, 0.0f))) {
      const auto selected = choose_model_file();
      if (!selected.empty())
        start_load(selected, false);
    }
    ImGui::EndDisabled();

    if (job || draining) {
      const auto shown = job ? job : draining;
      double fraction;
      std::string message;
      {
        std::lock_guard<std::mutex> lock(shown->mutex);
        fraction = shown->fraction;
        message = shown->message;
      }
      const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - jobStart).count();
      if (job) {
        ImGui::ProgressBar(static_cast<float>(fraction), ImVec2(-1.0f, 0.0f), message.c_str());
        ImGui::Text("Elapsed: %.1f seconds", seconds);
        if (!jobIsLoad && ImGui::Button("Stop current optimization", ImVec2(-1.0f, 0.0f))) {
          job->abandoned = true;
          draining = job;
          job.reset();
          operationStatus = "Stopping: the result will be discarded. The current FA-QEM pass "
                            "finishes in the background first.";
        }
      } else {
        ImGui::ProgressBar(static_cast<float>(fraction), ImVec2(-1.0f, 0.0f), "Stopping...");
      }
    }

    if (original.parts.empty()) {
      ImGui::TextWrapped("Choose a .glb, .gltf, .obj, .stl, .ply or .off file. Loading and "
                         "optimization run in the background.");
      ImGui::TextWrapped("%s", operationStatus.c_str());
      ImGui::End();
      ImGui::Render();
      ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
      glfwSwapBuffers(window);
      continue;
    }
    ImGui::TextWrapped("Model: %s", modelPath.filename().string().c_str());
    ImGui::Text("Showing: %s", state.showOptimized ? "Optimized" : "Original");
    if (ImGui::Button(state.showOptimized ? "Show original" : "Show optimized"))
      state.showOptimized = !state.showOptimized;
    ImGui::SameLine();
    if (ImGui::Button(state.wireframe ? "Solid mode" : "Wireframe mode"))
      state.wireframe = !state.wireframe;
    ImGui::Separator();

    ImGui::TextUnformatted("Optimization settings");
    ImGui::BeginDisabled(busy);
    ImGui::SetNextItemWidth(240.0f);
    ImGui::Combo("Target", &mode, "Automatic (best face count)\0Face count (best result)\0Face count (fast, one pass)\0");
    if (mode == 0) {
      ImGui::SetNextItemWidth(240.0f);
      ImGui::Combo("Detail", &detail, "Low (0.5%)\0Medium (0.2%)\0High (0.1%)\0Ultra (0.05%)\0Custom\0");
      if (detail == 4) {
        ImGui::SetNextItemWidth(240.0f);
        ImGui::InputDouble("Max deviation (%)", &tolerancePercent, 0.01, 0.1, "%.4f");
        tolerancePercent = std::clamp(tolerancePercent, 0.0001, 100.0);
      }
      ImGui::TextWrapped("The face count follows from the allowed worst-case deviation "
                         "(%% of the bounding-box diagonal), checked at every collapse.");
    } else {
      ImGui::SetNextItemWidth(240.0f);
      ImGui::InputInt("Target triangles", &targetFaces, 100, 1000);
      targetFaces = std::max(targetFaces, 4);
      if (mode == 1)
        ImGui::TextWrapped("Searches up to 6 passes for the lowest deviation at this face count.");
    }
    ImGui::SetNextItemWidth(240.0f);
    ImGui::Combo("Preset", &preset, "Recommended\0Paper Table 1 (exact)\0Plain QEM (ablation baseline)\0");
    ImGui::Checkbox("Bake colour", &bakeColor);
    ImGui::SameLine();
    ImGui::Checkbox("Bake normal map", &bakeNormal);
    if (ImGui::Button("Optimize", ImVec2(-1.0f, 0.0f))) {
      try {
        start_optimize();
      } catch (const std::exception &error) {
        operationStatus = "Optimization failed: " + std::string(error.what());
      }
    }
    ImGui::EndDisabled();
    ImGui::BeginDisabled(busy || !result);
    if (ImGui::Button("Open output folder", ImVec2(-1.0f, 0.0f)) && result)
      open_in_finder(result->out_dir);
    ImGui::EndDisabled();
    ImGui::TextWrapped("%s", operationStatus.c_str());
    ImGui::Separator();

    if (ImGui::BeginTable("stats", 2, ImGuiTableFlags_BordersInnerV)) {
      ImGui::TableNextColumn();
      stats_column("Original", originalStats);
      ImGui::TableNextColumn();
      stats_column("Optimized", optimizedStats);
      ImGui::EndTable();
    }
    ImGui::Separator();
    ImGui::Text("Vertex reduction:   %.2f%%", reduction(originalStats.vertices, optimizedStats.vertices));
    ImGui::Text("Triangle reduction: %.2f%%", reduction(originalStats.triangles, optimizedStats.triangles));
    if (result) {
      const auto &out = result->output;
      ImGui::Text("Simplify time:      %.2f s (bake %.2f s)", out.time_total, out.time_bake);
      if (out.has_metrics) {
        ImGui::Text("Hausdorff:     %.4f%% of bbox diagonal", out.metrics.hausdorff * 100.0);
        ImGui::Text("Chamfer:       %.6g", out.metrics.chamfer);
        ImGui::Text("Mean distance: %.4f%% of bbox diagonal", out.metrics.mean_distance * 100.0);
      }
      if (out.auto_mode)
        ImGui::Text("Within the limit: %s", out.auto_within ? "yes" : "no");
      if (out.best_for_count)
        ImGui::Text("Plain FA-QEM pass: %.4f%%", out.plain_hausdorff * 100.0);
      if (!out.passes.empty() && ImGui::TreeNode("Search passes")) {
        if (ImGui::BeginTable("passes", 4, ImGuiTableFlags_RowBg)) {
          ImGui::TableSetupColumn(out.auto_mode ? "factor" : "tau");
          ImGui::TableSetupColumn("faces");
          ImGui::TableSetupColumn("Hausdorff");
          ImGui::TableSetupColumn(out.auto_mode ? "within" : "reached");
          ImGui::TableHeadersRow();
          for (const auto &p : out.passes) {
            ImGui::TableNextColumn();
            ImGui::Text("%.4g", p.factor_or_tau);
            ImGui::TableNextColumn();
            ImGui::Text("%lld", static_cast<long long>(p.faces));
            ImGui::TableNextColumn();
            if (std::isnan(p.hausdorff))
              ImGui::TextUnformatted("-");
            else
              ImGui::Text("%.4f%%", p.hausdorff * 100.0);
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(p.ok ? "yes" : "no");
          }
          ImGui::EndTable();
        }
        ImGui::TreePop();
      }
      if (ImGui::TreeNode("Output files")) {
        for (const auto *path : {&out.textured_glb, &out.geometry_glb, &out.geometry_obj,
                                 &out.clay_wire_glb, &out.textured_wire_glb, &out.lines_glb})
          if (!path->empty())
            ImGui::TextWrapped("%s", std::filesystem::path(*path).filename().string().c_str());
        ImGui::TextDisabled("in %s", result->out_dir.string().c_str());
        ImGui::TreePop();
      }
    }
    ImGui::Spacing();
    ImGui::TextDisabled("Left-drag: orbit | Wheel: zoom | W: wireframe | Space: switch | R: reset");
    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
  }
  ImGui_ImplOpenGL2_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
  // a running engine thread is detached; exiting the process ends it
}
