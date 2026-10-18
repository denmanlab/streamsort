#include "streamsort/suppressor.hpp"
#include <algorithm>
#include <cmath>

namespace streamsort {

ArtifactSuppressor::ArtifactSuppressor(const SuppressorParams& p) : p_(p), x_(p.C), raw_(p.C) {
    H_ = p_.nb + p_.pre + 3;
    reset(0);
}

void ArtifactSuppressor::reset(int64_t s) {
    x_.clear(); raw_.clear();
    x0_ = emit_ = scan_ = s;
    ev_ = Ev{};
}

void ArtifactSuppressor::push(const float* raw, int64_t n, Rows& out) {
    x_.append(raw, n); raw_.append(raw, n);
    advance(false, out);
}

void ArtifactSuppressor::flush(Rows& out) { advance(true, out); }

static inline int count_clip(const float* r, int C, float clip) {
    int k = 0;
    for (int c = 0; c < C; ++c) k += std::fabs(r[c]) >= clip;
    return k;
}

void ArtifactSuppressor::advance(bool final, Rows& out) {
    const int C = p_.C;
    const float clip = float(p_.clip);
    int64_t end = x0_ + x_.rows();
    while (true) {
        if (!ev_.active) {
            int64_t lo = scan_ - x0_, hit = -1;
            for (int64_t r = std::max<int64_t>(lo, 0); r < x_.rows(); ++r)
                if (count_clip(x_.row(r), C, clip) >= p_.kmin) { hit = r; break; }
            if (hit < 0) { scan_ = end; break; }
            start_event(x0_ + hit);
        }
        if (!resolve(end, final)) break;
        if (!finish_template(end, final)) break;
    }
    int64_t limit;
    if (final) limit = end;
    else if (!ev_.active) limit = end - (p_.pre + 1);
    else if (!ev_.resolved) limit = ev_.b0;
    else limit = std::min(end, ev_.on + p_.L - (p_.pre + 1));
    limit = std::max(limit, emit_);
    out.append(x_, emit_ - x0_, limit - x0_);
    emit_ = limit;
    int64_t keep = std::max(emit_ - H_, x0_);
    if (ev_.active) keep = std::min(keep, ev_.on - p_.nb);
    keep = std::max(keep, x0_);
    x_.erase_front(keep - x0_); raw_.erase_front(keep - x0_); x0_ = keep;
}

const std::vector<float>* ArtifactSuppressor::tmpl(int key) {
    if (key < 0) return nullptr;
    auto h = history_.find(key);
    if (h == history_.end() || int(h->second.size()) < p_.min_history) return nullptr;
    auto c = tpl_cache_.find(key);
    if (c != tpl_cache_.end()) return &c->second;
    const auto& hs = h->second;
    const size_t n = hs.size(), sz = hs[0].size();
    std::vector<float> med(sz), tmp(n);
    for (size_t i = 0; i < sz; ++i) {
        for (size_t k = 0; k < n; ++k) tmp[k] = hs[k][i];
        std::sort(tmp.begin(), tmp.end());
        float m = (n % 2) ? tmp[n / 2] : (tmp[n / 2 - 1] + tmp[n / 2]) / 2.0f;
        med[i] = m * p_.taper[i / p_.C];
    }
    return &(tpl_cache_[key] = std::move(med));
}

void ArtifactSuppressor::start_event(int64_t on) {
    ev_ = Ev{};
    ev_.active = true; ev_.on = on; ev_.cond = cond_; ev_.last_clip = on; ev_.sub_upto = on; ev_.b0 = on - p_.pre;
    if (p_.subtract) {
        const auto* t = tmpl(cond_);
        if (t) { ev_.tpl = *t; ev_.has_tpl = true; }
    }
    auto h = history_.find(cond_);
    ev_.n_hist = (cond_ >= 0 && h != history_.end()) ? int(h->second.size()) : 0;
}

bool ArtifactSuppressor::resolve(int64_t end, bool final) {
    if (!ev_.resolved) {
        const float clip = float(p_.clip);
        int64_t on = ev_.on, a = ev_.last_clip + 1 - x0_, hi = std::min(end, on + p_.maxspan) - x0_;
        for (int64_t r = hi - 1; r >= std::max<int64_t>(a, 0); --r) {
            if (count_clip(x_.row(r), p_.C, clip) > 0) { ev_.last_clip = x0_ + r; break; }
        }
        bool done_scan = end >= on + p_.maxspan;
        int64_t span = ev_.last_clip + 1 - on, b1 = on + span + p_.post;
        if (!((done_scan && end >= b1 + 3) || final)) return false;
        ev_.resolved = true; ev_.span = span; ev_.b1 = b1;
    }
    apply(end);
    return true;
}

void ArtifactSuppressor::apply(int64_t end) {
    const int C = p_.C;
    if (ev_.has_tpl) {
        int64_t a = ev_.sub_upto, b = std::min(end, ev_.on + p_.L);
        for (int64_t g = a; g < b; ++g) {
            float* xr = x_.row(g - x0_);
            const float* tr = ev_.tpl.data() + (g - ev_.on) * C;
            for (int c = 0; c < C; ++c) xr[c] -= tr[c];
        }
        ev_.sub_upto = std::max(a, b);
    }
    if (p_.bridge && !ev_.bridged && end >= ev_.b1 + 3 && ev_.b0 - 3 >= x0_) {
        int64_t b0 = ev_.b0 - x0_, b1 = ev_.b1 - x0_, n = b1 - b0;
        std::vector<float> l(C, 0.f), r(C, 0.f);
        for (int k = 0; k < 3; ++k) {
            const float* a = x_.row(b0 - 3 + k); const float* b = x_.row(b1 + k);
            for (int c = 0; c < C; ++c) { l[c] += a[c]; r[c] += b[c]; }
        }
        for (int c = 0; c < C; ++c) { l[c] /= 3.0f; r[c] /= 3.0f; }
        const double step = n > 1 ? 1.0 / double(n - 1) : 0.0;
        for (int64_t i = 0; i < n; ++i) {
            float w = (i == n - 1 && n > 1) ? 1.0f : float(double(i) * step);
            float* xr = x_.row(b0 + i);
            for (int c = 0; c < C; ++c) xr[c] = (1.0f - w) * l[c] + w * r[c];
        }
        ev_.bridged = true;
    } else if (!p_.bridge) {
        ev_.bridged = true;
    }
}

bool ArtifactSuppressor::finish_template(int64_t end, bool final) {
    const int C = p_.C;
    int64_t on = ev_.on;
    if (end < on + p_.L && !final) return false;
    if (end >= on + p_.L && ev_.cond >= 0 && on - p_.nb >= x0_) {
        std::vector<float> base(C, 0.f), r(size_t(p_.L) * C);
        for (int k = 0; k < p_.nb; ++k) {
            const float* b = raw_.row(on - p_.nb - x0_ + k);
            for (int c = 0; c < C; ++c) base[c] += b[c];
        }
        for (int c = 0; c < C; ++c) base[c] /= float(p_.nb);
        for (int t = 0; t < p_.L; ++t) {
            const float* s = raw_.row(on - x0_ + t);
            for (int c = 0; c < C; ++c) r[size_t(t) * C + c] = s[c] - base[c];
        }
        auto& h = history_[ev_.cond];
        h.push_back(std::move(r));
        while (int(h.size()) > p_.n_history) h.pop_front();
        tpl_cache_.erase(ev_.cond);
    }
    ArtifactEvent e;
    e.onset = on; e.span = ev_.resolved ? ev_.span : -1; e.cond = ev_.cond; e.n_hist = ev_.n_hist; e.subtracted = ev_.has_tpl;
    events.push_back(e);
    scan_ = on + p_.L;
    ev_ = Ev{};
    cond_ = -1;
    return true;
}

}  // namespace streamsort
