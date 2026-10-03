#pragma once
#include <filesystem>

// returns an empty path when cancelled
std::filesystem::path choose_model_file();
void open_in_finder(const std::filesystem::path &path);
