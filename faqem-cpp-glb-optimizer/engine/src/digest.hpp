// SHA-256 digests of results and output files (cross-platform identity checks: faqem --digest).
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace faqem {

class Sha256 {
public:
    Sha256();
    void update(const void* data, size_t n);
    std::string hex();  // finalises

private:
    void block(const uint8_t* p);
    uint32_t h_[8];
    uint8_t buf_[64];
    size_t used_ = 0;
    uint64_t total_ = 0;
};

std::string sha256_hex(const void* data, size_t n);
std::string sha256_file(const std::string& path);

}  // namespace faqem
