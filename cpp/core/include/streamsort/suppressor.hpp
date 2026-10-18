// Streaming causal stimulus-artifact suppression (port of streamsort.preprocess.suppressor).
// Raw-domain float32 counts in, float32 counts out, bounded look-ahead.
#pragma once
#include <cstdint>
#include <deque>
#include <map>
#include <vector>
#include "rows.hpp"

namespace streamsort {

struct SuppressorParams {
    int C = 0;
    int clip = 6190, kmin = 3;
    int L = 0, pre = 0, post = 0, maxspan = 0, nb = 0;
    int n_history = 20, min_history = 5;
    std::vector<float> taper;      // length L
    bool subtract = true, bridge = true;
};

struct ArtifactEvent {
    int64_t onset = 0, span = -1;
    int cond = -1, n_hist = 0;
    bool subtracted = false;
};

class ArtifactSuppressor {
public:
    explicit ArtifactSuppressor(const SuppressorParams& p);
    void reset(int64_t start_sample);
    void set_condition(int key) { cond_ = key; }   // condition of the NEXT event; -1 = none (bridge only)
    void push(const float* raw, int64_t n, Rows& out);
    void flush(Rows& out);
    std::vector<ArtifactEvent> events;
    const SuppressorParams& params() const { return p_; }

private:
    struct Ev {
        bool active = false, resolved = false, bridged = false, has_tpl = false;
        int64_t on = 0, last_clip = 0, sub_upto = 0, b0 = 0, b1 = 0, span = -1;
        int cond = -1, n_hist = 0;
        std::vector<float> tpl;
    };
    void advance(bool final, Rows& out);
    void start_event(int64_t on);
    bool resolve(int64_t end, bool final);
    void apply(int64_t end);
    bool finish_template(int64_t end, bool final);
    const std::vector<float>* tmpl(int key);

    SuppressorParams p_;
    int H_;
    Rows x_, raw_;
    int64_t x0_ = 0, emit_ = 0, scan_ = 0;
    Ev ev_;
    int cond_ = -1;
    std::map<int, std::deque<std::vector<float>>> history_;
    std::map<int, std::vector<float>> tpl_cache_;
};

}  // namespace streamsort
