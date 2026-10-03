#pragma once

#include "engine_bridge.hpp"
#include <filesystem>

// input empty: start with an Open dialog button. Otherwise the model is loaded and, when
// optimize_on_open is set, optimized in the background with `options`.
// out_dir empty: <model dir>/<stem>_faqem
void run_viewer(const std::filesystem::path &input, OptimizationOptions options,
                std::filesystem::path out_dir, bool optimize_on_open);
