// ZMQ publisher (spikes, pulses, stim TTLs, block heartbeats, stream status) and condition subscriber.
// Framing follows the Open Ephys ZMQ Interface: [envelope][JSON header][optional binary payload].
#pragma once
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

class ZmqLink {
public:
    // pubEndpoint e.g. "tcp://*:5557"; condEndpoint e.g. "tcp://127.0.0.1:5558" (worker binds PUB there)
    // context: optional external ZMQ context (tests use inproc endpoints on a shared context)
    ZmqLink(const std::string& pubEndpoint, const std::string& condEndpoint, std::function<void(int)> onCondition, void* context = nullptr);
    ~ZmqLink();
    bool ok() const { return ok_; }
    const std::string& error() const { return err_; }
    // thread-safe; copies the payload
    void send(const std::string& envelope, const std::string& header, const void* payload = nullptr, size_t nbytes = 0);
    uint64_t sent() const { return sent_.load(); }

private:
    struct Msg { std::string env, hdr; std::vector<char> data; };
    void loop();
    void* ctx_ = nullptr;
    bool ownCtx_ = true;
    void* pub_ = nullptr;
    void* sub_ = nullptr;
    bool ok_ = false;
    std::string err_;
    std::function<void(int)> onCondition_;
    std::mutex m_;
    std::condition_variable cv_;
    std::deque<Msg> q_;
    std::atomic<bool> stop_{false};
    std::atomic<uint64_t> sent_{0};
    std::thread th_;
};

// wall clock, microseconds since the Unix epoch (comparable with Python time.time() on the same machine)
int64_t wallMicros();
// steady clock in seconds (pairing clock)
double steadySeconds();
std::string jsonEscape(const std::string& s);
