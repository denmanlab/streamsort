// Online template-matching decoder (port of streamsort.online.decoder.OnlineDecoder, numba path).
// All tables are produced in Python from an OnlineDecoder (streamsort.core_cpp.tables_from_decoder)
// and exported for the plugin by `streamsort bank`; this file holds no signal-path constants.
#pragma once
#include <cstdint>
#include <memory>
#include <vector>
#include "rows.hpp"
#include "suppressor.hpp"
#include "pool.hpp"

namespace streamsort {

struct DecoderTables {
    int C = 0;
    double bit_volts = 0.195;
    int n_sections = 0;
    std::vector<double> sos;        // (S, 6)
    std::vector<double> zi_unit;    // (S, 2)
    // whitening, sparse by output column: y_w[j] = sum_k y[w_row[k]] * w_val[k], k in w_ptr[j]..w_ptr[j+1]
    std::vector<int64_t> w_ptr, w_row;
    std::vector<float> w_val;
    std::vector<float> thr;         // C
    int K = 0;
    std::vector<int64_t> nbr;       // (C, K)
    int dedup = 10, before_need = 0, after_need = 0, lag = 0, bpre = 0, bpost = 0;
    int clip_counts = 6190, clip_min_channels = 3;
    int T = 0, Ks = 0;
    std::vector<int64_t> cand_ptr, cand_units, tro_cand;   // (C+1), (M), (M)
    std::vector<float> Tcat, ttcat;                       // (M, T, Ks), (M)
    std::vector<int64_t> sset;                            // (C, Ks)
    std::vector<int64_t> shifts;
    double amp_lo = 0.5, amp_hi = 1.8;
    float min_energy = 50.f;
    std::vector<int64_t> unit_ids;                        // bank index -> unit id
    int n_before = 30;
    bool has_suppressor = true;
    SuppressorParams sup;
};

struct Spike {
    int64_t frame, unit;
    float score, amp;
    int64_t peak_channel;
};

class Decoder {
public:
    explicit Decoder(const DecoderTables& t, int n_threads = 1);
    void reset();
    void start_at(int64_t sample);
    // raw: n rows x C (float32 counts, or int16 converted by caller); appends finished spikes to out
    void push(const float* raw, int64_t n, std::vector<Spike>& out);
    void flush(std::vector<Spike>& out);
    void set_condition(int key) { if (sup_) sup_->set_condition(key); }
    ArtifactSuppressor* suppressor() { return sup_.get(); }
    // template hot-swap: keep the per-condition artifact history (and event log) of the old decoder
    void adopt_suppressor(Decoder& old) {
        if (old.sup_ && t_.has_suppressor && old.sup_->params().C == t_.C) sup_ = std::move(old.sup_);
    }
    const DecoderTables& tables() const { return t_; }
    int64_t n_blanked = 0;
    double timing[6] = {0, 0, 0, 0, 0, 0};   // suppress, blank, filter, median+whiten, detect, classify (s)

private:
    void release(int64_t n, bool final, std::vector<Spike>& out);
    void detect_and_classify(bool final, std::vector<Spike>& out);
    void classify(const std::vector<int64_t>& g, const std::vector<int64_t>& cc, std::vector<Spike>& out);

    DecoderTables t_;
    std::unique_ptr<ArtifactSuppressor> sup_;
    std::unique_ptr<Pool> pool_;
    int64_t tro_min_ = 0, tro_max_ = 0, sh_min_ = 0, sh_max_ = 0;
    std::vector<int64_t> band_lo_, band_hi_, band_off_;   // whitening as row bands of W^T
    std::vector<float> band_val_;
    Rows pending_, Y_, sup_out_;
    int64_t pend_start_ = 0, blank_until_ = -1, Y_start_ = 0, next_det_ = 0;
    std::vector<float> hold_;
    bool has_hold_ = false, primed_ = false;
    std::vector<double> zi_;        // (S, 2, C)
    std::vector<double> yd_;        // scratch
    std::vector<float> yf_;
};

}  // namespace streamsort
