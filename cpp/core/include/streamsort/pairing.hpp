// Pair artifact-detected pulse onsets with stim TTL edges (port of the Phase 2 DecoderProcessor._pair rule).
// Times are on one clock shared by every stream of the processor (block arrival time, seconds).
// A clip with no TTL within tol is dropped; a TTL with no clip within tol is emitted on its own
// (onset_sample = -1, time = TTL time). Without a TTL source every clip is emitted with ttl_index = -1.
#pragma once
#include <cstdint>
#include <deque>
#include <vector>

namespace streamsort {

struct Pulse {
    int64_t onset_sample;   // -1: TTL without a detected artifact
    double t;               // pairing clock (s)
    int64_t ttl_index;      // -1: no TTL source
};

class PulsePairer {
public:
    PulsePairer(bool ttl_routed, double tol_s = 0.05) : ttl_(ttl_routed), tol_(tol_s) {}
    void add_ttl(int64_t index, double t) { ttls_.push_back({index, t}); }
    void add_artifact(int64_t onset_sample, double t) { cand_.push_back({onset_sample, t}); }
    std::vector<Pulse> pair(double now) {
        std::vector<Pulse> out;
        if (!ttl_) {
            for (auto& c : cand_) out.push_back({c.s, c.t, -1});
            cand_.clear();
            return out;
        }
        while (!cand_.empty() && !ttls_.empty()) {
            const auto c = cand_.front(); const auto k = ttls_.front();
            if (c.t < k.t - tol_) cand_.pop_front();                                         // clip, no TTL
            else if (c.t > k.t + tol_) { ttls_.pop_front(); out.push_back({-1, k.t, k.k}); }   // TTL, no clip
            else { cand_.pop_front(); ttls_.pop_front(); out.push_back({c.s, c.t, k.k}); }
        }
        while (!cand_.empty() && now - cand_.front().t > tol_) cand_.pop_front();
        while (!ttls_.empty() && now - ttls_.front().t > tol_) { auto k = ttls_.front(); ttls_.pop_front(); out.push_back({-1, k.t, k.k}); }
        return out;
    }
    size_t n_pending_ttl() const { return ttls_.size(); }

private:
    struct C { int64_t s; double t; };
    struct K { int64_t k; double t; };
    bool ttl_;
    double tol_;
    std::deque<C> cand_;
    std::deque<K> ttls_;
};

}  // namespace streamsort
