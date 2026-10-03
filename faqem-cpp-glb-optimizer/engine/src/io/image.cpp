#include "image.hpp"

#include <png.h>
#include <turbojpeg.h>
#include <webp/decode.h>

#include <algorithm>
#include <csetjmp>
#include <cstring>

namespace faqem {

// ------------------------------------------------------------------------------------------------
// PNG (Pillow: PngImagePlugin + convert("RGBA"))
// ------------------------------------------------------------------------------------------------
namespace {
struct PngSrc {
    const uint8_t* d;
    size_t n, off;
};
void png_read_cb(png_structp p, png_bytep out, png_size_t len) {
    PngSrc* s = (PngSrc*)png_get_io_ptr(p);
    if (s->off + len > s->n) png_error(p, "truncated");
    std::memcpy(out, s->d + s->off, len);
    s->off += len;
}

ImagePtr decode_png(const uint8_t* data, size_t n) {
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) return nullptr;
    png_infop info = png_create_info_struct(png);
    ImagePtr img = std::make_shared<Image>();
    std::vector<uint8_t> raw;
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, nullptr);
        return nullptr;
    }
    PngSrc src{data, n, 0};
    png_set_read_fn(png, &src, png_read_cb);
    png_set_crc_action(png, PNG_CRC_QUIET_USE, PNG_CRC_QUIET_USE);
    png_read_info(png, info);
    png_uint_32 w, h;
    int depth, ctype, interlace;
    png_get_IHDR(png, info, &w, &h, &depth, &ctype, &interlace, nullptr, nullptr);
    bool has_trns = png_get_valid(png, info, PNG_INFO_tRNS);
    img->w = (int)w;
    img->h = (int)h;
    img->channels = 4;
    img->px.assign((size_t)w * h * 4, 255);
    if (ctype == PNG_COLOR_TYPE_GRAY && depth == 16) {
        // Pillow mode "I;16" -> RGBA: values clip at 255; tRNS compares the 16-bit value
        png_color_16p tc = nullptr;
        if (has_trns) png_get_tRNS(png, info, nullptr, nullptr, &tc);
        if (interlace != PNG_INTERLACE_NONE) png_set_interlace_handling(png);
        png_read_update_info(png, info);
        size_t rb = png_get_rowbytes(png, info);
        raw.resize(rb * h);
        std::vector<png_bytep> rows(h);
        for (png_uint_32 y = 0; y < h; y++) rows[y] = raw.data() + y * rb;
        png_read_image(png, rows.data());
        for (png_uint_32 y = 0; y < h; y++)
            for (png_uint_32 x = 0; x < w; x++) {
                unsigned v = ((unsigned)rows[y][2 * x] << 8) | rows[y][2 * x + 1];
                uint8_t l = (uint8_t)std::min(v, 255u);
                uint8_t* o = img->at(x, y);
                o[0] = o[1] = o[2] = l;
                o[3] = (has_trns && tc && v == tc->gray) ? 0 : 255;
            }
        png_destroy_read_struct(&png, &info, nullptr);
        return img;
    }
    // 16-bit channels -> high byte (Pillow rawmodes "RGB;16B", "RGBA;16B", "LA;16B")
    if (depth == 16) png_set_strip_16(png);
    if (ctype == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (ctype == PNG_COLOR_TYPE_GRAY && depth < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (has_trns) png_set_tRNS_to_alpha(png);
    if (ctype == PNG_COLOR_TYPE_GRAY || ctype == PNG_COLOR_TYPE_GRAY_ALPHA) png_set_gray_to_rgb(png);
    png_set_filler(png, 0xff, PNG_FILLER_AFTER);
    if (interlace != PNG_INTERLACE_NONE) png_set_interlace_handling(png);
    png_read_update_info(png, info);
    size_t rb = png_get_rowbytes(png, info);
    if (rb != (size_t)w * 4) {
        png_destroy_read_struct(&png, &info, nullptr);
        return nullptr;
    }
    std::vector<png_bytep> rows(h);
    for (png_uint_32 y = 0; y < h; y++) rows[y] = img->px.data() + (size_t)y * w * 4;
    png_read_image(png, rows.data());
    png_destroy_read_struct(&png, &info, nullptr);
    return img;
}

// ------------------------------------------------------------------------------------------------
// JPEG (Pillow: libjpeg-turbo, ISLOW DCT, fancy upsampling; CMYK assumes Adobe inversion)
// ------------------------------------------------------------------------------------------------
inline uint8_t clip8(int v) { return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v)); }
inline int muldiv255(int a, int b) {
    int tmp = a * b + 128;
    return ((tmp >> 8) + tmp) >> 8;
}

ImagePtr decode_jpeg(const uint8_t* data, size_t n) {
    tjhandle h = tj3Init(TJINIT_DECOMPRESS);
    if (!h) return nullptr;
    ImagePtr img;
    if (tj3DecompressHeader(h, data, n) == 0) {
        int w = tj3Get(h, TJPARAM_JPEGWIDTH), ht = tj3Get(h, TJPARAM_JPEGHEIGHT);
        int cs = tj3Get(h, TJPARAM_COLORSPACE);
        int pf = TJPF_RGB, ch = 3;
        if (cs == TJCS_GRAY) {
            pf = TJPF_GRAY;
            ch = 1;
        } else if (cs == TJCS_CMYK || cs == TJCS_YCCK) {
            pf = TJPF_CMYK;
            ch = 4;
        }
        std::vector<uint8_t> buf((size_t)w * ht * ch);
        if (tj3Decompress8(h, data, n, buf.data(), w * ch, pf) == 0 || tj3GetErrorCode(h) == TJERR_WARNING) {
            img = std::make_shared<Image>();
            img->w = w;
            img->h = ht;
            img->channels = 4;
            img->px.resize((size_t)w * ht * 4);
            for (size_t i = 0; i < (size_t)w * ht; i++) {
                uint8_t* o = &img->px[i * 4];
                const uint8_t* s = &buf[i * ch];
                if (ch == 1) {
                    o[0] = o[1] = o[2] = s[0];
                } else if (ch == 3) {
                    o[0] = s[0];
                    o[1] = s[1];
                    o[2] = s[2];
                } else {
                    int c = 255 - s[0], m = 255 - s[1], y = 255 - s[2], k = 255 - s[3];  // "CMYK;I"
                    int nk = 255 - k;
                    o[0] = clip8(nk - muldiv255(c, nk));
                    o[1] = clip8(nk - muldiv255(m, nk));
                    o[2] = clip8(nk - muldiv255(y, nk));
                }
                o[3] = 255;
            }
        }
    }
    tj3Destroy(h);
    return img;
}

ImagePtr decode_webp(const uint8_t* data, size_t n) {
    int w = 0, h = 0;
    uint8_t* out = WebPDecodeRGBA(data, n, &w, &h);
    if (!out) return nullptr;
    ImagePtr img = std::make_shared<Image>();
    img->w = w;
    img->h = h;
    img->channels = 4;
    img->px.assign(out, out + (size_t)w * h * 4);
    WebPFree(out);
    return img;
}

void png_write_cb(png_structp p, png_bytep d, png_size_t len) {
    auto* v = (std::vector<uint8_t>*)png_get_io_ptr(p);
    v->insert(v->end(), d, d + len);
}
void png_flush_cb(png_structp) {}
}  // namespace

ImagePtr decode_image_rgba(const uint8_t* d, size_t n) {
    static const uint8_t png_sig[8] = {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
    if (n >= 8 && std::memcmp(d, png_sig, 8) == 0) return decode_png(d, n);
    if (n >= 3 && d[0] == 0xff && d[1] == 0xd8 && d[2] == 0xff) return decode_jpeg(d, n);
    if (n >= 12 && std::memcmp(d, "RIFF", 4) == 0 && std::memcmp(d + 8, "WEBP", 4) == 0) return decode_webp(d, n);
    return nullptr;
}

std::vector<uint8_t> encode_png(const Image& img) {
    std::vector<uint8_t> out;
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    png_infop info = png_create_info_struct(png);
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        return {};
    }
    png_set_write_fn(png, &out, png_write_cb, png_flush_cb);
    png_set_IHDR(png, info, img.w, img.h, 8, img.channels == 4 ? PNG_COLOR_TYPE_RGBA : PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_compression_level(png, 6);
    png_write_info(png, info);
    for (int y = 0; y < img.h; y++) png_write_row(png, (png_const_bytep)&img.px[(size_t)y * img.w * img.channels]);
    png_write_end(png, nullptr);
    png_destroy_write_struct(&png, &info);
    return out;
}

}  // namespace faqem
