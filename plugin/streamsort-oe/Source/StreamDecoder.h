// One decoding thread per enabled stream: GUI blocks in, spikes / pulses / heartbeats out over ZMQ.
// process() only copies the block (first C channels, converted to raw counts) into a queue.
#pragma once
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "streamsort/bank.hpp"
#include "streamsort/decoder.hpp"
#include "streamsort/pairing.hpp"
#include "ZmqLink.h"

struct Block {
    int64_t s0 = 0;          // stream sample number of the first sample
    int64_t n = 0;
    double t_arrival = 0.0;  // steady clock (s) when process() received it
    int64_t wall_us = 0;
    std::vector<float> rows; // n x C raw counts
};

class StreamDecoder {
public:
    StreamDecoder(std::string streamName, int streamId, double sampleRate, std::string bankFolder,
                  int nThreads, bool ttlRouted, ZmqLink* zmq);
    ~StreamDecoder();
    bool loadOk() const { return loaded_; }
    const std::string& status() const { return status_; }
    int channels() const { return tables_.C; }
    void start();
    void stop();
    void enqueue(Block&& b);                       // called from process(); never blocks on decoding
    void addTtl(int64_t index, double t);           // stim TTL seen by process()
    void setCondition(int id) { pendingCond_.store(id); hasCond_.store(true); }
    uint64_t overruns() const { return overruns_.load(); }
    std::string helloHeader() const;

private:
    void loop();
    void handle(Block& b);
    void publish(const std::vector<streamsort::Spike>& sp, int64_t head, int64_t wall_us);
    bool maybeSwap(int64_t head);
    std::string tablesFile() const;

    std::string name_, folder_;
    int id_;
    double fs_;
    int nThreads_;
    bool ttlRouted_;
    ZmqLink* zmq_;
    streamsort::DecoderTables tables_;
    streamsort::BankInfo info_;
    std::unique_ptr<streamsort::Decoder> dec_;
    std::unique_ptr<streamsort::PulsePairer> pairer_;
    bool loaded_ = false, started_ = false;
    std::string status_;
    int64_t head_ = 0, lastPulse_ = -(int64_t(1) << 40), nextSwapCheck_ = 0, nextHello_ = 0;
    long long tablesMtime_ = 0;
    size_t nEvents_ = 0;
    std::deque<std::pair<int64_t, std::pair<double, int64_t>>> arrivals_;   // (block end, (t_arrival, wall_us))
    std::vector<float> ring_;                                                // last warm rows (hot-swap)
    int warm_ = 450;
    std::mutex m_, ttlM_;
    std::condition_variable cv_;
    std::deque<Block> q_;
    std::deque<std::pair<int64_t, double>> ttlIn_;
    std::atomic<bool> stop_{false}, hasCond_{false};
    std::atomic<int> pendingCond_{-1};
    std::atomic<uint64_t> overruns_{0};
    std::thread th_;
    uint64_t msgNum_ = 0;
};
