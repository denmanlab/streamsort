// Row-major (n x C) float buffer with cheap front erasure.
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>

namespace streamsort {

struct Rows {
    int C = 0;
    std::vector<float> d;
    Rows() = default;
    explicit Rows(int c) : C(c) {}
    int64_t rows() const { return C ? int64_t(d.size()) / C : 0; }
    float* row(int64_t r) { return d.data() + r * C; }
    const float* row(int64_t r) const { return d.data() + r * C; }
    void clear() { d.clear(); }
    void append(const float* src, int64_t n) { d.insert(d.end(), src, src + n * C); }
    void append(const Rows& o, int64_t r0, int64_t r1) {
        if (r1 > r0) d.insert(d.end(), o.d.begin() + r0 * C, o.d.begin() + r1 * C);
    }
    void erase_front(int64_t n) {
        if (n <= 0) return;
        if (n >= rows()) { d.clear(); return; }
        d.erase(d.begin(), d.begin() + n * C);
    }
};

}  // namespace streamsort
