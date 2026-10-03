#include "fmath.hpp"

#include <cmath>

#if FAQEM_PYTHON_PARITY
namespace {
// separate translation unit + volatile pointers: never folded, always the libm call numpy makes
double (*volatile g_atan2)(double, double) = &::atan2;
double (*volatile g_acos)(double) = &::acos;
double (*volatile g_cos)(double) = &::cos;
double (*volatile g_pow)(double, double) = &::pow;
}  // namespace
#else
extern "C" {
double cr_atan2(double, double);
double cr_acos(double);
double cr_cos(double);
double cr_pow(double, double);
}
#endif

namespace faqem {
namespace fm {

#if FAQEM_PYTHON_PARITY
double atan2(double y, double x) { return g_atan2(y, x); }
double acos(double x) { return g_acos(x); }
double cos(double x) { return g_cos(x); }
double pow(double x, double y) { return g_pow(x, y); }
const char* build_mode() { return "python-parity"; }
#else
double atan2(double y, double x) { return cr_atan2(y, x); }
double acos(double x) { return cr_acos(x); }
double cos(double x) { return cr_cos(x); }
double pow(double x, double y) { return cr_pow(x, y); }
const char* build_mode() { return "portable"; }
#endif

}  // namespace fm
}  // namespace faqem
