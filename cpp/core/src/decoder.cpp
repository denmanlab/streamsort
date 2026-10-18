#include "streamsort/decoder.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <chrono>
static inline double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

namespace streamsort {

Decoder::Decoder(const DecoderTables& t, int n_threads) : t_(t), pending_(t.C), Y_(t.C), sup_out_(t.C) {
    if (t_.C <= 0) throw std::invalid_argument("C must be > 0");
    pool_ = std::make_unique<Pool>(n_threads);
    if (!t_.tro_cand.empty()) {
        tro_min_ = *std::min_element(t_.tro_cand.begin(), t_.tro_cand.end());
        tro_max_ = *std::max_element(t_.tro_cand.begin(), t_.tro_cand.end());
    }
    {   // banded W^T: input row k touches output columns [band_lo, band_hi)
        const int C = t_.C;
        band_lo_.assign(C, C); band_hi_.assign(C, 0);
        for (int j = 0; j < C; ++j)
            for (int64_t q = t_.w_ptr[j]; q < t_.w_ptr[j + 1]; ++q) {
                int64_t k = t_.w_row[q];
                band_lo_[k] = std::min<int64_t>(band_lo_[k], j); band_hi_[k] = std::max<int64_t>(band_hi_[k], j + 1);
            }
        band_off_.assign(C + 1, 0);
        for (int k = 0; k < C; ++k) band_off_[k + 1] = band_off_[k] + std::max<int64_t>(band_hi_[k] - band_lo_[k], 0);
        band_val_.assign(size_t(band_off_[C]), 0.f);
        for (int j = 0; j < C; ++j)
            for (int64_t q = t_.w_ptr[j]; q < t_.w_ptr[j + 1]; ++q) {
                int64_t k = t_.w_row[q];
                band_val_[size_t(band_off_[k] + j - band_lo_[k])] = t_.w_val[q];
            }
    }
    if (!t_.shifts.empty()) {
        sh_min_ = *std::min_element(t_.shifts.begin(), t_.shifts.end());
        sh_max_ = *std::max_element(t_.shifts.begin(), t_.shifts.end());
    }
    if (t_.bpre > t_.lag) throw std::invalid_argument("look-ahead must cover the pre-onset blank");
    if (t_.has_suppressor) sup_ = std::make_unique<ArtifactSuppressor>(t_.sup);
    reset();
}

void Decoder::reset() {
    zi_.assign(size_t(t_.n_sections) * 2 * t_.C, 0.0);
    primed_ = false;
    pending_.clear(); Y_.clear();
    pend_start_ = 0; blank_until_ = -1; has_hold_ = false;
    Y_start_ = 0; next_det_ = 0; n_blanked = 0;
}

void Decoder::start_at(int64_t s) {
    reset();
    pend_start_ = Y_start_ = next_det_ = s;
    if (sup_) sup_->reset(s);
}

void Decoder::push(const float* raw, int64_t n, std::vector<Spike>& out) {
    if (sup_) {
        sup_out_.clear();
        double t0 = now_s();
        sup_->push(raw, n, sup_out_);
        timing[0] += now_s() - t0;
        if (!sup_out_.rows()) return;
        pending_.append(sup_out_, 0, sup_out_.rows());
    } else {
        pending_.append(raw, n);
    }
    int64_t n_release = pending_.rows() - t_.lag;
    if (n_release <= 0) return;
    release(n_release, false, out);
}

void Decoder::flush(std::vector<Spike>& out) {
    if (sup_) {
        sup_out_.clear();
        sup_->flush(sup_out_);
        pending_.append(sup_out_, 0, sup_out_.rows());
    }
    if (pending_.rows()) release(pending_.rows(), true, out);
    detect_and_classify(true, out);
}

void Decoder::release(int64_t n, bool final, std::vector<Spike>& out) {
    const int C = t_.C;
    double t0 = now_s();
    const int64_t P = pending_.rows(), g0 = pend_start_;
    const float clip = float(t_.clip_counts);
    // blanking mask over all pending rows
    std::vector<int> diff(P + 1, 0);
    std::vector<char> mask(P, 0);
    if (blank_until_ > g0) std::fill(mask.begin(), mask.begin() + std::min<int64_t>(blank_until_ - g0, P), 1);
    int64_t last_clip = -1;
    for (int64_t r = 0; r < P; ++r) {
        const float* x = pending_.row(r); int k = 0;
        for (int c = 0; c < C; ++c) k += std::fabs(x[c]) >= clip;
        if (k >= t_.clip_min_channels) {
            diff[std::max<int64_t>(r - t_.bpre, 0)] += 1;
            diff[std::min<int64_t>(r + t_.bpost + 1, P)] -= 1;
            last_clip = r;
        }
    }
    if (last_clip >= 0) {
        int run = 0;
        for (int64_t r = 0; r < P; ++r) { run += diff[r]; if (run > 0) mask[r] = 1; }
        blank_until_ = std::max(blank_until_, g0 + last_clip + t_.bpost + 1);
    }
    // rel = pending[:n], blanked rows held at the last clean value
    std::vector<float> rel(pending_.d.begin(), pending_.d.begin() + n * C);
    bool any_m = false, any_clean = false;
    for (int64_t i = 0; i < n; ++i) { if (mask[i]) any_m = true; else any_clean = true; }
    if (any_m) {
        std::vector<float> hold(C, 0.f);
        if (has_hold_) hold = hold_;
        else if (any_clean) {
            for (int64_t i = 0; i < n; ++i) if (!mask[i]) { std::copy(&rel[i * C], &rel[i * C] + C, hold.begin()); break; }
        }
        int64_t last = -1;
        for (int64_t i = 0; i < n; ++i) {
            if (!mask[i]) { last = i; continue; }
            ++n_blanked;
            const float* src = last >= 0 ? &rel[last * C] : hold.data();
            std::copy(src, src + C, &rel[i * C]);
        }
    }
    if (any_clean) {
        for (int64_t i = n - 1; i >= 0; --i)
            if (!mask[i]) { hold_.assign(&rel[i * C], &rel[i * C] + C); has_hold_ = true; break; }
    }
    pending_.erase_front(n);
    pend_start_ = g0 + n;

    double t1 = now_s(); timing[1] += t1 - t0;
    // causal SOS high-pass (DF2T, float64) + median reference -> float32
    const int S = t_.n_sections;
    const double sc = t_.bit_volts;
    if (!primed_) {
        for (int s = 0; s < S; ++s) for (int k = 0; k < 2; ++k) for (int c = 0; c < C; ++c)
            zi_[(size_t(s) * 2 + k) * C + c] = t_.zi_unit[s * 2 + k] * (double(rel[c]) * sc);
        primed_ = true;
    }
    yd_.resize(size_t(n) * C);
    pool_->parallel_for(C, [&](int64_t c0, int64_t c1) {
        const int64_t w = c1 - c0;
        std::vector<double> z0(size_t(S) * w), z1(size_t(S) * w), v(w);
        for (int s = 0; s < S; ++s) for (int64_t c = c0; c < c1; ++c) {
            z0[s * w + (c - c0)] = zi_[(size_t(s) * 2) * C + c]; z1[s * w + (c - c0)] = zi_[(size_t(s) * 2 + 1) * C + c];
        }
        for (int64_t i = 0; i < n; ++i) {
            const float* r = &rel[i * C];
            for (int64_t q = 0; q < w; ++q) v[q] = double(r[c0 + q]) * sc;
            for (int s = 0; s < S; ++s) {
                const double* bq = &t_.sos[s * 6];
                double* Z0 = &z0[s * w]; double* Z1 = &z1[s * w];
                for (int64_t q = 0; q < w; ++q) {
                    double o = bq[0] * v[q] + Z0[q];
                    Z0[q] = bq[1] * v[q] - bq[4] * o + Z1[q];
                    Z1[q] = bq[2] * v[q] - bq[5] * o;
                    v[q] = o;
                }
            }
            double* y = &yd_[i * C];
            for (int64_t q = 0; q < w; ++q) y[c0 + q] = v[q];
        }
        for (int s = 0; s < S; ++s) for (int64_t c = c0; c < c1; ++c) {
            zi_[(size_t(s) * 2) * C + c] = z0[s * w + (c - c0)]; zi_[(size_t(s) * 2 + 1) * C + c] = z1[s * w + (c - c0)];
        }
    });
    double t2 = now_s(); timing[2] += t2 - t1;
    yf_.resize(size_t(n) * C);
    const int64_t y0 = Y_.rows();
    Y_.d.resize(size_t(y0 + n) * C);
    pool_->parallel_for(n, [&](int64_t i0, int64_t i1) {
        std::vector<double> tmp(C);
        for (int64_t i = i0; i < i1; ++i) {
            const double* y = &yd_[i * C];
            std::copy(y, y + C, tmp.begin());
            double m;
            size_t h = C / 2;
            std::nth_element(tmp.begin(), tmp.begin() + h, tmp.end());
            if (C % 2) m = tmp[h];
            else { double hi = tmp[h]; double lo = *std::max_element(tmp.begin(), tmp.begin() + h); m = (lo + hi) / 2.0; }
            float* yfr = &yf_[i * C];
            for (int c = 0; c < C; ++c) yfr[c] = float(y[c] - m);
            float* o = Y_.row(y0 + i);
            std::fill(o, o + C, 0.f);
            for (int k = 0; k < C; ++k) {          // per output j: inputs accumulated in ascending k
                const float yk = yfr[k];
                const float* bv = &band_val_[size_t(band_off_[k])];
                float* oj = o + band_lo_[k];
                const int64_t w = band_hi_[k] - band_lo_[k];
                for (int64_t q = 0; q < w; ++q) oj[q] += yk * bv[q];
            }
        }
    });
    timing[3] += now_s() - t2;
    detect_and_classify(final, out);
}

void Decoder::detect_and_classify(bool final, std::vector<Spike>& out) {
    const int C = t_.C;
    const int64_t Ys = Y_start_, N = Y_.rows(), end = Ys + N;
    const int64_t fin_hi = end - t_.after_need - (final ? 0 : t_.dedup), fin_lo = next_det_;
    if (fin_hi <= fin_lo) return;
    const int64_t a_rel = std::max(fin_lo, Ys + t_.before_need) - Ys, b_rel = fin_hi - Ys;
    double t0 = now_s();
    std::vector<int64_t> g, cc;
    const int nb_blocks = pool_->size();
    std::vector<std::vector<int64_t>> gb(nb_blocks), cb(nb_blocks);
    const int64_t span = b_rel - a_rel;
    if (span > 0) pool_->parallel_for(nb_blocks, [&](int64_t q0, int64_t q1) {
        for (int64_t q = q0; q < q1; ++q) {
            int64_t lo = a_rel + span * q / nb_blocks, hi = a_rel + span * (q + 1) / nb_blocks;
            for (int64_t n = lo; n < hi; ++n) {
                const float* yr = Y_.row(n);
                for (int c = 0; c < C; ++c) {
                    float v = yr[c];
                    if (v >= -t_.thr[c]) continue;
                    int64_t a = std::max<int64_t>(n - t_.dedup, 0), b = std::min<int64_t>(n + t_.dedup + 1, N);
                    bool ok = true;
                    const int64_t* nb = &t_.nbr[size_t(c) * t_.K];
                    for (int64_t m = a; m < b && ok; ++m) {
                        const float* ym = Y_.row(m);
                        for (int k = 0; k < t_.K; ++k) if (ym[nb[k]] < v) { ok = false; break; }
                    }
                    if (ok) { gb[q].push_back(n + Ys); cb[q].push_back(c); }
                }
            }
        }
    });
    for (int q = 0; q < nb_blocks; ++q) { g.insert(g.end(), gb[q].begin(), gb[q].end()); cc.insert(cc.end(), cb[q].begin(), cb[q].end()); }
    double t1 = now_s(); timing[4] += t1 - t0;
    classify(g, cc, out);
    timing[5] += now_s() - t1;
    next_det_ = fin_hi;
    int64_t keep_from = std::max(fin_hi - t_.before_need - t_.dedup - 1, Ys);
    Y_.erase_front(keep_from - Ys);
    Y_start_ = keep_from;
}

void Decoder::classify(const std::vector<int64_t>& g, const std::vector<int64_t>& cc, std::vector<Spike>& out) {
    const int64_t E = int64_t(g.size()), Ys = Y_start_, N = Y_.rows();
    const int T = t_.T, K = t_.Ks, NS = int(t_.shifts.size());
    if (NS > 16) throw std::invalid_argument("at most 16 jitter shifts");
    const size_t base = out.size();
    out.resize(base + size_t(E));
    // window rows needed by any candidate/shift: [gr - tro_max + sh_min, gr - tro_min + sh_max + T)
    const int64_t w_lo = -tro_max_ + sh_min_, w_hi = -tro_min_ + sh_max_ + T;
    pool_->parallel_for(E, [&](int64_t e0, int64_t e1) {
        std::vector<float> win(size_t(w_hi - w_lo) * K);
        for (int64_t e = e0; e < e1; ++e) {
            const int64_t c = cc[e], gr = g[e] - Ys;
            const int64_t* ss = &t_.sset[size_t(c) * K];
            for (int64_t r = 0; r < w_hi - w_lo; ++r) {
                int64_t row = gr + w_lo + r;
                float* wr = &win[size_t(r) * K];
                if (row < 0 || row >= N) { std::fill(wr, wr + K, 0.f); continue; }
                const float* yr = Y_.row(row);
                for (int k = 0; k < K; ++k) wr[k] = yr[ss[k]];
            }
            double bsc = -1.0e30, ba = 0.0; int64_t bj = -1, bs = 0;
            const int64_t TK = int64_t(T) * K;
            for (int64_t j = t_.cand_ptr[c]; j < t_.cand_ptr[c + 1]; ++j) {
                const float tt = t_.ttcat[j];
                const float* Tj = &t_.Tcat[size_t(j) * T * K];
                // all shifts of this candidate at once: independent accumulators, each summed in t,k order
                double dot[16] = {0};
                const float* wv[16];
                bool valid[16];
                for (int si = 0; si < NS; ++si) {
                    const int64_t st = gr - t_.tro_cand[j] + t_.shifts[si];
                    valid[si] = !(st < 0 || st + T > N);
                    wv[si] = &win[size_t(st - gr - w_lo) * K];
                }
                if (NS == 5) {
                    for (int64_t q = 0; q < TK; ++q) {
                        const float tq = Tj[q];
                        float p0 = wv[0][q] * tq, p1 = wv[1][q] * tq, p2 = wv[2][q] * tq, p3 = wv[3][q] * tq, p4 = wv[4][q] * tq;
                        dot[0] += double(p0); dot[1] += double(p1); dot[2] += double(p2); dot[3] += double(p3); dot[4] += double(p4);
                    }
                } else {
                    for (int64_t q = 0; q < TK; ++q) {
                        const float tq = Tj[q];
                        for (int si = 0; si < NS; ++si) { float p = wv[si][q] * tq; dot[si] += double(p); }
                    }
                }
                for (int si = 0; si < NS; ++si) {      // same shift order and tie rule as the reference
                    if (!valid[si]) continue;
                    const double a = dot[si] / double(tt);
                    if (a < t_.amp_lo || a > t_.amp_hi) continue;
                    const double sc_ = dot[si] * a;
                    if (sc_ > bsc) { bsc = sc_; bj = j; ba = a; bs = t_.shifts[si]; }
                }
            }
            Spike& sp = out[base + size_t(e)];
            float bscf = bj >= 0 ? float(bsc) : 0.f;
            bool assigned = bj >= 0 && bscf >= t_.min_energy;
            int64_t u = bj >= 0 ? t_.cand_units[bj] : 0;
            sp.unit = assigned ? t_.unit_ids[u] : -1;
            int64_t tro = assigned ? t_.tro_cand[bj] : t_.n_before;
            sp.frame = g[e] + (assigned ? bs : 0) - tro + t_.n_before;
            sp.score = bscf; sp.amp = bj >= 0 ? float(ba) : 0.f; sp.peak_channel = c;
        }
    });
}

}  // namespace streamsort
