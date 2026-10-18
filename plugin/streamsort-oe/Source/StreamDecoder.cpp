#include "StreamDecoder.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <sys/stat.h>

static long long fileMtime(const std::string& p) {
    struct stat st;
    if (stat(p.c_str(), &st) != 0) return 0;
#if defined(__APPLE__)
    return (long long)st.st_mtimespec.tv_sec * 1000000000LL + st.st_mtimespec.tv_nsec;
#elif defined(_WIN32)
    return (long long)st.st_mtime * 1000000000LL;
#else
    return (long long)st.st_mtim.tv_sec * 1000000000LL + st.st_mtim.tv_nsec;
#endif
}

static std::string resolveBankFolder(const std::string& ks) {
    // accept the Kilosort folder (bank in <ks>/streamsort) or the bank folder itself
    struct stat st;
    std::string a = ks + "/streamsort";
    if (stat((a + "/decoder_tables.txt").c_str(), &st) == 0) return a;
    return ks;
}

StreamDecoder::StreamDecoder(std::string streamName, int streamId, double sampleRate, std::string bankFolder,
                             int nThreads, bool ttlRouted, ZmqLink* zmq)
    : name_(std::move(streamName)), folder_(resolveBankFolder(bankFolder)), id_(streamId), fs_(sampleRate),
      nThreads_(nThreads), ttlRouted_(ttlRouted), zmq_(zmq) {
    try {
        tables_ = streamsort::load_tables(folder_, &info_);
        if (info_.sample_rate > 0 && std::fabs(info_.sample_rate - fs_) / fs_ > 0.01) {
            status_ = "bank sample rate " + std::to_string(info_.sample_rate) + " Hz does not match stream";
            return;
        }
        tablesMtime_ = fileMtime(tablesFile());
        warm_ = int(std::lround(0.015 * fs_));
        loaded_ = true;
        status_ = std::to_string(info_.n_usable) + "/" + std::to_string(info_.n_units) + " units, " +
                  std::to_string(tables_.C) + " ch";
    } catch (const std::exception& e) {
        status_ = e.what();
    }
}

StreamDecoder::~StreamDecoder() { stop(); }

std::string StreamDecoder::tablesFile() const { return folder_ + "/decoder_tables.txt"; }

std::string StreamDecoder::helloHeader() const {
    std::string u;
    // usable unit ids = units that are candidates on some channel
    std::vector<char> used(tables_.unit_ids.size(), 0);
    for (auto k : tables_.cand_units) used[size_t(k)] = 1;
    for (size_t k = 0; k < used.size(); ++k)
        if (used[k]) u += (u.empty() ? "" : ",") + std::to_string(tables_.unit_ids[k]);
    char b[512];
    std::snprintf(b, sizeof(b), "{\"type\":\"hello\",\"stream\":\"%s\",\"stream_id\":%d,\"sample_rate\":%.6f,"
                  "\"n_channels\":%d,\"bank\":\"%s\",\"fingerprint\":\"%s\",\"t_wall_us\":%lld,\"unit_ids\":[",
                  jsonEscape(name_).c_str(), id_, fs_, tables_.C, jsonEscape(folder_).c_str(),
                  jsonEscape(info_.fingerprint).c_str(), (long long)wallMicros());
    return std::string(b) + u + "]}";
}

void StreamDecoder::start() {
    if (!loaded_ || started_) return;
    dec_ = std::make_unique<streamsort::Decoder>(tables_, nThreads_);
    pairer_ = std::make_unique<streamsort::PulsePairer>(ttlRouted_, 0.05);
    stop_ = false; started_ = false; nEvents_ = 0; arrivals_.clear(); ring_.clear();
    if (zmq_) zmq_->send("STATUS", helloHeader());
    th_ = std::thread([this] { loop(); });
}

void StreamDecoder::stop() {
    if (!th_.joinable()) return;
    stop_ = true; cv_.notify_all(); th_.join();
}

void StreamDecoder::enqueue(Block&& b) {
    {
        std::lock_guard<std::mutex> l(m_);
        if (q_.size() > 2000) { ++overruns_; q_.pop_front(); }   // > ~1 min behind: drop oldest, counted
        q_.push_back(std::move(b));
    }
    cv_.notify_one();
}

void StreamDecoder::addTtl(int64_t index, double t) {
    std::lock_guard<std::mutex> l(ttlM_);
    ttlIn_.push_back({index, t});
}

void StreamDecoder::loop() {
    while (true) {
        Block b;
        {
            std::unique_lock<std::mutex> l(m_);
            cv_.wait(l, [this] { return stop_ || !q_.empty(); });
            if (q_.empty() && stop_) break;
            b = std::move(q_.front()); q_.pop_front();
        }
        handle(b);
    }
    if (dec_ && started_) {
        std::vector<streamsort::Spike> sp;
        dec_->flush(sp);
        publish(sp, head_, wallMicros());
    }
}

void StreamDecoder::handle(Block& b) {
    const int C = tables_.C;
    if (!started_) { dec_->start_at(b.s0); head_ = b.s0; started_ = true; nextSwapCheck_ = b.s0 + int64_t(0.5 * fs_); nextHello_ = b.s0 + int64_t(2.0 * fs_); }
    if (b.s0 != head_) {                       // gap or reset in the stream: restart cleanly
        dec_->start_at(b.s0); head_ = b.s0;
    }
    if (hasCond_.exchange(false)) dec_->set_condition(pendingCond_.load());
    {
        std::lock_guard<std::mutex> l(ttlM_);
        for (auto& k : ttlIn_) pairer_->add_ttl(k.first, k.second);
        ttlIn_.clear();
    }
    std::vector<streamsort::Spike> sp;
    dec_->push(b.rows.data(), b.n, sp);
    head_ = b.s0 + b.n;
    arrivals_.push_back({head_, {b.t_arrival, b.wall_us}});
    while (arrivals_.size() > 256) arrivals_.pop_front();
    // ring of the last warm rows for hot-swap warm-up
    ring_.insert(ring_.end(), b.rows.begin(), b.rows.end());
    if (int64_t(ring_.size()) > int64_t(warm_) * C) ring_.erase(ring_.begin(), ring_.end() - int64_t(warm_) * C);
    // new artifact events -> pairing (arrival time of the block that held the onset)
    if (auto* s = dec_->suppressor()) {
        for (; nEvents_ < s->events.size(); ++nEvents_) {
            const int64_t on = s->events[nEvents_].onset;
            double t = b.t_arrival;
            for (auto& a : arrivals_) if (on < a.first) { t = a.second.first; break; }
            pairer_->add_artifact(on, t);
        }
    }
    for (auto& p : pairer_->pair(b.t_arrival)) {
        if (p.onset_sample >= 0) lastPulse_ = std::max(lastPulse_, p.onset_sample);
        int64_t wall = b.wall_us;
        for (auto& a : arrivals_) if (p.onset_sample >= 0 && p.onset_sample < a.first) { wall = a.second.second; break; }
        char h[384];
        std::snprintf(h, sizeof(h), "{\"type\":\"pulse\",\"stream\":\"%s\",\"onset_sample\":%lld,\"ttl_index\":%lld,"
                      "\"t_wall_us\":%lld,\"approx\":%s}", jsonEscape(name_).c_str(), (long long)p.onset_sample,
                      (long long)p.ttl_index, (long long)wall, p.onset_sample < 0 ? "true" : "false");
        if (zmq_) zmq_->send("PULSE", h);
    }
    publish(sp, head_, b.wall_us);
    if (head_ >= nextSwapCheck_) { nextSwapCheck_ = head_ + int64_t(0.5 * fs_); maybeSwap(head_); }
    if (head_ >= nextHello_) { nextHello_ = head_ + int64_t(2.0 * fs_); if (zmq_) zmq_->send("STATUS", helloHeader()); }   // late-joining workers
}

void StreamDecoder::publish(const std::vector<streamsort::Spike>& sp, int64_t head, int64_t wall_us) {
    if (!zmq_) return;
    std::vector<int64_t> f; std::vector<int32_t> u;
    for (auto& s : sp) if (s.unit >= 0) { f.push_back(s.frame); u.push_back(int32_t(s.unit)); }
    char h[384];
    std::snprintf(h, sizeof(h), "{\"type\":\"spikes\",\"stream\":\"%s\",\"stream_id\":%d,\"message_num\":%llu,\"n\":%zu,"
                  "\"head\":%lld,\"sample_rate\":%.6f,\"t_wall_us\":%lld}", jsonEscape(name_).c_str(), id_,
                  (unsigned long long)msgNum_++, f.size(), (long long)head, fs_, (long long)wall_us);
    std::vector<char> payload(f.size() * 12);
    if (!f.empty()) {
        std::memcpy(payload.data(), f.data(), f.size() * 8);
        std::memcpy(payload.data() + f.size() * 8, u.data(), u.size() * 4);
    }
    zmq_->send("SPIKES", h, payload.data(), payload.size());   // empty payload = heartbeat (head advanced)
}

bool StreamDecoder::maybeSwap(int64_t head) {
    const long long mt = fileMtime(tablesFile());
    if (mt == 0 || mt == tablesMtime_ || double(head - lastPulse_) < 0.6 * fs_) return false;
    streamsort::BankInfo info;
    streamsort::DecoderTables t;
    try { t = streamsort::load_tables(folder_, &info); }
    catch (const std::exception& e) { status_ = std::string("hot-swap failed: ") + e.what(); tablesMtime_ = mt; return false; }
    if (t.C != tables_.C) { status_ = "hot-swap refused: channel count changed"; tablesMtime_ = mt; return false; }
    std::vector<streamsort::Spike> sp;
    dec_->flush(sp);                                       // finalise everything up to head
    publish(sp, head, wallMicros());
    auto nd = std::make_unique<streamsort::Decoder>(t, nThreads_);
    nd->adopt_suppressor(*dec_);
    const int64_t nr = int64_t(ring_.size()) / t.C;
    nd->start_at(head - nr);
    std::vector<streamsort::Spike> warm;
    if (nr) nd->push(ring_.data(), nr, warm);              // already-emitted region: discard
    dec_ = std::move(nd); tables_ = t; info_ = info; tablesMtime_ = mt;
    nEvents_ = dec_->suppressor() ? dec_->suppressor()->events.size() : 0;
    status_ = std::to_string(info_.n_usable) + "/" + std::to_string(info_.n_units) + " units (hot-swapped)";
    if (zmq_) zmq_->send("STATUS", helloHeader());
    return true;
}
