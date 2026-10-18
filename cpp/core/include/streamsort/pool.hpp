// Minimal persistent thread pool with a blocking parallel_for (used inside one decoder).
#pragma once
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace streamsort {

class Pool {
public:
    explicit Pool(int n) : n_(n < 1 ? 1 : n) {
        for (int i = 1; i < n_; ++i) th_.emplace_back([this, i] { loop(i); });
    }
    ~Pool() {
        { std::lock_guard<std::mutex> l(m_); stop_ = true; ++gen_; }
        cv_.notify_all();
        for (auto& t : th_) t.join();
    }
    int size() const { return n_; }
    // fn(lo, hi) over [0, n) split into n_ contiguous chunks; caller thread takes chunk 0
    void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& fn) {
        if (n_ == 1 || n < 2) { if (n > 0) fn(0, n); return; }
        {
            std::lock_guard<std::mutex> l(m_);
            fn_ = &fn; total_ = n; pending_ = n_ - 1; ++gen_;
        }
        cv_.notify_all();
        run_chunk(0);
        std::unique_lock<std::mutex> l(m_);
        done_.wait(l, [this] { return pending_ == 0; });
        fn_ = nullptr;
    }

private:
    void run_chunk(int i) {
        int64_t lo = total_ * i / n_, hi = total_ * (i + 1) / n_;
        if (hi > lo) (*fn_)(lo, hi);
    }
    void loop(int i) {
        uint64_t seen = 0;
        while (true) {
            std::unique_lock<std::mutex> l(m_);
            cv_.wait(l, [&] { return gen_ != seen; });
            seen = gen_;
            if (stop_) return;
            l.unlock();
            run_chunk(i);
            l.lock();
            if (--pending_ == 0) done_.notify_one();
        }
    }
    int n_;
    std::vector<std::thread> th_;
    std::mutex m_;
    std::condition_variable cv_, done_;
    const std::function<void(int64_t, int64_t)>* fn_ = nullptr;
    int64_t total_ = 0;
    int pending_ = 0;
    uint64_t gen_ = 0;
    bool stop_ = false;
};

}  // namespace streamsort
