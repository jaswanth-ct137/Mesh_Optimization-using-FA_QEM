// Transcendental functions used by the engine.
//
// Portable build (default): CORE-MATH correctly rounded binary64 functions
// (third_party/core-math). A correctly rounded result is unique, so it is the same on macOS, Linux
// and Windows, whatever the CPU or C library.
//
// Python-parity build (FAQEM_PYTHON_PARITY, macOS arm64 only): the platform libm, exactly what
// numpy / Numba call, so results match the Python reference bit for bit.
#pragma once

namespace faqem {

constexpr double PI = 3.141592653589793;  // np.pi

namespace fm {

double atan2(double y, double x);
double acos(double x);
double cos(double x);
// never folded or inlined (clang would turn pow(x, 2.0) into x * x)
double pow(double x, double y);

// "portable" or "python-parity"
const char* build_mode();

}  // namespace fm
}  // namespace faqem
