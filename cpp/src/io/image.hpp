// Image decode / encode with Pillow semantics: decode(...) == np.asarray(PIL.Image.open(b).convert("RGBA")).
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace faqem {

struct Image {
    int w = 0, h = 0, channels = 4;  // pixel rows top to bottom, interleaved
    std::vector<uint8_t> px;
    uint8_t* at(int x, int y) { return &px[((size_t)y * w + x) * channels]; }
    const uint8_t* at(int x, int y) const { return &px[((size_t)y * w + x) * channels]; }
};
using ImagePtr = std::shared_ptr<Image>;

// PNG / JPEG / WebP bytes -> RGBA image; nullptr if the format is unsupported or broken
ImagePtr decode_image_rgba(const uint8_t* data, size_t n);
// PNG bytes of an 8-bit RGB / RGBA image (channels 3 or 4)
std::vector<uint8_t> encode_png(const Image& img);

}  // namespace faqem
