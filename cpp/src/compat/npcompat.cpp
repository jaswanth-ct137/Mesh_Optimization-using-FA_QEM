#include "npcompat.hpp"

#include "fmath.hpp"

namespace faqem {
namespace np {

double py_pow(double x, double y) { return fm::pow(x, y); }

double pairwise_sum(const double* a, i64 n, i64 stride) {
    if (n < 8) {
        double res = 0.;
        for (i64 i = 0; i < n; i++) res += a[i * stride];
        return res;
    } else if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; j++) r[j] = a[j * stride];
        i64 i;
        for (i = 8; i < n - (n % 8); i += 8) {
            for (int j = 0; j < 8; j++) r[j] += a[(i + j) * stride];
        }
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) res += a[i * stride];
        return res;
    } else {
        i64 n2 = n / 2;
        n2 -= n2 % 8;
        return pairwise_sum(a, n2, stride) + pairwise_sum(a + n2 * stride, n - n2, stride);
    }
}

double sum(const double* a, i64 n, i64 stride) {
    double res = 0.0;
    for (i64 s = 0; s < n; s += 8192) {
        i64 k = std::min<i64>(8192, n - s);
        res += pairwise_sum(a + s * stride, k, stride);
    }
    return res;
}

double vmax(const double* a, i64 n) {
    double m = a[0];
    for (i64 i = 1; i < n; i++) m = fmax0(m, a[i]);
    return m;
}
double vmin(const double* a, i64 n) {
    double m = a[0];
    for (i64 i = 1; i < n; i++) m = fmin0(m, a[i]);
    return m;
}

VecD linspace(double start, double stop, i64 num) {
    VecD y(num);
    i64 div = num - 1;
    double delta = stop - start;
    if (div > 0) {
        double step = delta / (double)div;
        if (step == 0) {
            for (i64 i = 0; i < num; i++) y[i] = ((double)i / (double)div) * delta + start;
        } else {
            for (i64 i = 0; i < num; i++) y[i] = (double)i * step + start;
        }
        y[num - 1] = stop;
    } else if (num == 1) {
        y[0] = start;
    }
    return y;
}

// ---- numpy quicksort.cpp: aquicksort_ / aheapsort_ ---------------------------------------------
template <typename T>
static inline bool lt(T a, T b) { return a < b; }
template <>
inline bool lt<double>(double a, double b) { return a < b || (b != b && a == a); }

template <typename T>
static void aheapsort(const T* v, i64* tosort, i64 n) {
    i64* a = tosort - 1;
    i64 i, j, l, tmp;
    for (l = n >> 1; l > 0; --l) {
        tmp = a[l];
        for (i = l, j = l << 1; j <= n;) {
            if (j < n && lt(v[a[j]], v[a[j + 1]])) j += 1;
            if (lt(v[tmp], v[a[j]])) {
                a[i] = a[j];
                i = j;
                j += j;
            } else {
                break;
            }
        }
        a[i] = tmp;
    }
    for (; n > 1;) {
        tmp = a[n];
        a[n] = a[1];
        n -= 1;
        for (i = 1, j = 2; j <= n;) {
            if (j < n && lt(v[a[j]], v[a[j + 1]])) j++;
            if (lt(v[tmp], v[a[j]])) {
                a[i] = a[j];
                i = j;
                j += j;
            } else {
                break;
            }
        }
        a[i] = tmp;
    }
}

static int get_msb(uint64_t n) {
    int depth = 0;
    while (n >>= 1) depth++;
    return depth;
}

template <typename T>
static void aquicksort(const T* v, i64* tosort, i64 num) {
    const int SMALL_QUICKSORT = 15;
    const int PYA_QS_STACK = 128;  // NPY_BITSOF_INTP * 2
    T vp;
    i64* pl = tosort;
    i64* pr = tosort + num - 1;
    i64* stack[PYA_QS_STACK];
    i64** sptr = stack;
    i64 *pm, *pi, *pj, *pk, vi;
    int depth[PYA_QS_STACK];
    int* psdepth = depth;
    int cdepth = get_msb((uint64_t)num) * 2;
    if (num <= 1) return;
    for (;;) {
        if (cdepth < 0) {
            aheapsort(v, pl, pr - pl + 1);
            goto stack_pop;
        }
        while ((pr - pl) > SMALL_QUICKSORT) {
            pm = pl + ((pr - pl) >> 1);
            if (lt(v[*pm], v[*pl])) std::swap(*pm, *pl);
            if (lt(v[*pr], v[*pm])) std::swap(*pr, *pm);
            if (lt(v[*pm], v[*pl])) std::swap(*pm, *pl);
            vp = v[*pm];
            pi = pl;
            pj = pr - 1;
            std::swap(*pm, *pj);
            for (;;) {
                do {
                    ++pi;
                } while (lt(v[*pi], vp));
                do {
                    --pj;
                } while (lt(vp, v[*pj]));
                if (pi >= pj) break;
                std::swap(*pi, *pj);
            }
            pk = pr - 1;
            std::swap(*pi, *pk);
            if (pi - pl < pr - pi) {
                *sptr++ = pi + 1;
                *sptr++ = pr;
                pr = pi - 1;
            } else {
                *sptr++ = pl;
                *sptr++ = pi - 1;
                pl = pi + 1;
            }
            *psdepth++ = --cdepth;
        }
        for (pi = pl + 1; pi <= pr; ++pi) {
            vi = *pi;
            vp = v[vi];
            pj = pi;
            pk = pi - 1;
            while (pj > pl && lt(vp, v[*pk])) {
                *pj-- = *pk--;
            }
            *pj = vi;
        }
    stack_pop:
        if (sptr == stack) break;
        pr = *(--sptr);
        pl = *(--sptr);
        cdepth = *(--psdepth);
    }
}

VecI argsort_quick(const VecI& v) {
    VecI idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    aquicksort(v.data(), idx.data(), (i64)v.size());
    return idx;
}
VecI argsort_quick(const VecD& v) {
    VecI idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    aquicksort(v.data(), idx.data(), (i64)v.size());
    return idx;
}

UniqueRows unique_rows(const i64* a, i64 n, int k) {
    UniqueRows u;
    VecI order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](i64 x, i64 y) {
        for (int j = 0; j < k; j++) {
            if (a[x * k + j] != a[y * k + j]) return a[x * k + j] < a[y * k + j];
        }
        return false;
    });
    u.inverse.assign(n, 0);
    i64 g = -1;
    for (i64 i = 0; i < n; i++) {
        i64 r = order[i];
        bool fresh = i == 0;
        if (!fresh) {
            i64 p = order[i - 1];
            for (int j = 0; j < k; j++)
                if (a[r * k + j] != a[p * k + j]) {
                    fresh = true;
                    break;
                }
        }
        if (fresh) {
            g++;
            u.index.push_back(r);
            u.counts.push_back(0);
            for (int j = 0; j < k; j++) u.rows.push_back(a[r * k + j]);
        }
        u.counts[g]++;
        u.inverse[r] = g;
    }
    return u;
}

}  // namespace np
}  // namespace faqem
