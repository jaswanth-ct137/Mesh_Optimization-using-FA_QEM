// Minimal stand-in for numpy/npy_common.h: the macros scipy's cKDTree sources use.
#pragma once
#include <cassert>  // numpy headers pull this in
#include <cstdint>
typedef intptr_t npy_intp;
#if defined(__GNUC__) || defined(__clang__)
#define NPY_LIKELY(x) __builtin_expect(!!(x), 1)
#define NPY_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define NPY_LIKELY(x) (x)
#define NPY_UNLIKELY(x) (x)
#endif
// numpy only enables __builtin_prefetch inside its own build (HAVE___BUILTIN_PREFETCH); for
// scipy this is a no-op
#define NPY_PREFETCH(x, rw, loc)
