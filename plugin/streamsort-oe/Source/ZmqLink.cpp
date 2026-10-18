#include "ZmqLink.h"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <zmq.h>

int64_t wallMicros() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

double steadySeconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string jsonEscape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (static_cast<unsigned char>(c) < 0x20) o += ' ';
        else o += c;
    }
    return o;
}

ZmqLink::ZmqLink(const std::string& pubEndpoint, const std::string& condEndpoint, std::function<void(int)> onCondition, void* context)
    : onCondition_(std::move(onCondition)) {
    ownCtx_ = context == nullptr;
    ctx_ = context ? context : zmq_ctx_new();
    pub_ = zmq_socket(ctx_, ZMQ_PUB);
    int hwm = 100000;
    zmq_setsockopt(pub_, ZMQ_SNDHWM, &hwm, sizeof(hwm));
    if (zmq_bind(pub_, pubEndpoint.c_str()) != 0) {
        err_ = std::string("cannot bind ") + pubEndpoint + ": " + zmq_strerror(zmq_errno());
        return;
    }
    if (!condEndpoint.empty()) {
        sub_ = zmq_socket(ctx_, ZMQ_SUB);
        zmq_setsockopt(sub_, ZMQ_SUBSCRIBE, "", 0);
        zmq_connect(sub_, condEndpoint.c_str());     // connect never fails if the worker is not up yet
    }
    ok_ = true;
    th_ = std::thread([this] { loop(); });
}

ZmqLink::~ZmqLink() {
    stop_ = true;
    cv_.notify_all();
    if (th_.joinable()) th_.join();
    int linger = 0;
    if (pub_) { zmq_setsockopt(pub_, ZMQ_LINGER, &linger, sizeof(linger)); zmq_close(pub_); }
    if (sub_) { zmq_setsockopt(sub_, ZMQ_LINGER, &linger, sizeof(linger)); zmq_close(sub_); }
    if (ctx_ && ownCtx_) zmq_ctx_term(ctx_);
}

void ZmqLink::send(const std::string& env, const std::string& hdr, const void* payload, size_t n) {
    if (!ok_) return;
    Msg m{env, hdr, {}};
    if (payload && n) m.data.assign(static_cast<const char*>(payload), static_cast<const char*>(payload) + n);
    { std::lock_guard<std::mutex> l(m_); q_.push_back(std::move(m)); }
    cv_.notify_one();
}

void ZmqLink::loop() {
    // one thread owns both sockets (ZMQ sockets are not thread-safe)
    while (!stop_) {
        std::deque<Msg> batch;
        {
            std::unique_lock<std::mutex> l(m_);
            cv_.wait_for(l, std::chrono::milliseconds(5), [this] { return stop_ || !q_.empty(); });
            batch.swap(q_);
        }
        for (auto& m : batch) {
            zmq_send(pub_, m.env.data(), m.env.size(), ZMQ_SNDMORE);
            zmq_send(pub_, m.hdr.data(), m.hdr.size(), m.data.empty() ? 0 : ZMQ_SNDMORE);
            if (!m.data.empty()) zmq_send(pub_, m.data.data(), m.data.size(), 0);
            ++sent_;
        }
        if (sub_) {
            char buf[512];
            while (true) {
                int n = zmq_recv(sub_, buf, sizeof(buf) - 1, ZMQ_DONTWAIT);
                if (n < 0) break;
                buf[n < int(sizeof(buf)) - 1 ? n : int(sizeof(buf)) - 1] = 0;
                int more = 0; size_t sz = sizeof(more);
                zmq_getsockopt(sub_, ZMQ_RCVMORE, &more, &sz);
                // condition message: header JSON {"type":"condition","id":N}; envelope frames are skipped
                const char* p = std::strstr(buf, "\"id\"");
                if (std::strstr(buf, "\"condition\"") && p) {
                    p = std::strchr(p, ':');
                    if (p && onCondition_) onCondition_(std::atoi(p + 1));
                }
            }
        }
    }
}
