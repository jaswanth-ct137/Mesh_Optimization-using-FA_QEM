#include "nprandom.hpp"

namespace faqem {
namespace np {

namespace {
const uint32_t INIT_A = 0x43b0d7e5u, MULT_A = 0x931e8875u, INIT_B = 0x8b51f9ddu, MULT_B = 0x58f38dedu;
const uint32_t MIX_MULT_L = 0xca01f9ddu, MIX_MULT_R = 0x4973f715u;
const int XSHIFT = 16;
const int POOL_SIZE = 4;

inline uint32_t hashmix(uint32_t value, uint32_t& hash_const) {
    value ^= hash_const;
    hash_const *= MULT_A;
    value *= hash_const;
    value ^= value >> XSHIFT;
    return value;
}
inline uint32_t mix(uint32_t x, uint32_t y) {
    uint32_t result = (MIX_MULT_L * x - MIX_MULT_R * y);
    result ^= result >> XSHIFT;
    return result;
}

// SeedSequence(seed).generate_state(4, uint64)
void seed_sequence_state(uint64_t seed, uint64_t out[4]) {
    std::vector<uint32_t> entropy;
    if (seed == 0) entropy.push_back(0);
    while (seed > 0) {
        entropy.push_back((uint32_t)(seed & 0xffffffffu));
        seed >>= 32;
    }
    uint32_t pool[POOL_SIZE];
    uint32_t hc = INIT_A;
    for (int i = 0; i < POOL_SIZE; i++) {
        pool[i] = hashmix(i < (int)entropy.size() ? entropy[i] : 0u, hc);
    }
    for (int s = 0; s < POOL_SIZE; s++)
        for (int d = 0; d < POOL_SIZE; d++)
            if (s != d) pool[d] = mix(pool[d], hashmix(pool[s], hc));
    for (size_t s = POOL_SIZE; s < entropy.size(); s++)
        for (int d = 0; d < POOL_SIZE; d++) pool[d] = mix(pool[d], hashmix(entropy[s], hc));
    uint32_t words[8];
    uint32_t hb = INIT_B;
    for (int i = 0; i < 8; i++) {
        uint32_t v = pool[i % POOL_SIZE];
        v ^= hb;
        hb *= MULT_B;
        v *= hb;
        v ^= v >> XSHIFT;
        words[i] = v;
    }
    for (int i = 0; i < 4; i++) out[i] = (uint64_t)words[2 * i] | ((uint64_t)words[2 * i + 1] << 32);
}

const unsigned __int128 PCG_MULT =
    ((unsigned __int128)2549297995355413924ULL << 64) | (unsigned __int128)4865540595714422341ULL;
}  // namespace

Generator::Generator(uint64_t seed) {
    uint64_t v[4];
    seed_sequence_state(seed, v);
    unsigned __int128 s = ((unsigned __int128)v[0] << 64) | v[1];
    unsigned __int128 i = ((unsigned __int128)v[2] << 64) | v[3];
    state_ = 0;
    inc_ = (i << 1u) | 1u;
    state_ = state_ * PCG_MULT + inc_;
    state_ += s;
    state_ = state_ * PCG_MULT + inc_;
}

uint64_t Generator::next64() {
    state_ = state_ * PCG_MULT + inc_;
    uint64_t hi = (uint64_t)(state_ >> 64), lo = (uint64_t)state_;
    unsigned rot = (unsigned)(state_ >> 122u);
    uint64_t x = hi ^ lo;
    return (x >> rot) | (x << ((-rot) & 63));
}

uint32_t Generator::next32() {
    if (has_uint32_) {
        has_uint32_ = false;
        return uinteger_;
    }
    uint64_t n = next64();
    has_uint32_ = true;
    uinteger_ = (uint32_t)(n >> 32);
    return (uint32_t)(n & 0xffffffffu);
}

double Generator::random() { return (double)(next64() >> 11) * (1.0 / 9007199254740992.0); }

VecD Generator::random(i64 n) {
    VecD r(n);
    for (i64 i = 0; i < n; i++) r[i] = random();
    return r;
}

VecI Generator::choice(i64 n, i64 size, const VecD& p) {
    (void)n;
    VecD cdf(p.size());
    double acc = 0.0;
    for (size_t i = 0; i < p.size(); i++) {
        acc += p[i];
        cdf[i] = acc;
    }
    double last = cdf.back();
    for (auto& c : cdf) c /= last;
    VecD u = random(size);
    VecI idx(size);
    for (i64 i = 0; i < size; i++) {
        idx[i] = (i64)(std::upper_bound(cdf.begin(), cdf.end(), u[i]) - cdf.begin());
    }
    return idx;
}

uint64_t Generator::interval(uint64_t max) {
    if (max == 0) return 0;
    uint64_t mask = max, value;
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
    mask |= mask >> 32;
    if (max <= 0xffffffffULL) {
        while ((value = (next32() & mask)) > max) {
        }
    } else {
        while ((value = (next64() & mask)) > max) {
        }
    }
    return value;
}

VecI Generator::permutation(i64 n) {
    VecI a(n);
    for (i64 i = 0; i < n; i++) a[i] = i;
    for (i64 i = n - 1; i >= 1; i--) {
        i64 j = (i64)interval((uint64_t)i);
        std::swap(a[i], a[j]);
    }
    return a;
}

}  // namespace np
}  // namespace faqem
