// numpy.random.default_rng(seed) reimplemented bit-exactly: SeedSequence -> PCG64 (XSL-RR 128/64)
// -> Generator.random / choice(p=...) / permutation (numpy 2.2.6).
#pragma once
#include <cstdint>
#include <vector>

#include "npcompat.hpp"

namespace faqem {
namespace np {

class Generator {
public:
    explicit Generator(uint64_t seed);
    uint64_t next64();
    uint32_t next32();
    double random();  // Generator.random(): [0, 1) with 53 bits
    VecD random(i64 n);
    // Generator.choice(n, size, p=p) with replacement
    VecI choice(i64 n, i64 size, const VecD& p);
    // Generator.permutation(n)
    VecI permutation(i64 n);
    uint64_t interval(uint64_t max);  // random_interval

private:
    unsigned __int128 state_ = 0, inc_ = 0;
    bool has_uint32_ = false;
    uint32_t uinteger_ = 0;
};

}  // namespace np
}  // namespace faqem
