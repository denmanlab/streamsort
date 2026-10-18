// Drives the plugin's own StreamDecoder + ZmqLink (the non-JUCE part of StreamsortProcessor) the way
// process()/handleTTLEvent() do, and records every published message (inproc SUB on a shared context).
// usage: plugin_path_driver <bank_folder> <raw_int16.bin> <nch_file> <buffer> <ttl_samples.txt> <conds.txt> <out.jsonl>
//   ttl_samples.txt: one stim-TTL sample index (relative to the file start) per line
//   conds.txt: "<sample> <condition id>" per line (applied before the block containing <sample>)
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <thread>
#include <zmq.h>
#include "../Source/StreamDecoder.h"
#include "../Source/ZmqLink.h"

int main(int argc, char** argv) {
    if (argc < 8) { std::cerr << "usage: bank raw nch buffer ttl.txt conds.txt out.jsonl\n"; return 2; }
    const int nf = std::stoi(argv[3]), B = std::stoi(argv[4]);
    std::vector<int64_t> ttl; { std::ifstream f(argv[5]); int64_t v; while (f >> v) ttl.push_back(v); }
    std::vector<std::pair<int64_t, int>> conds; { std::ifstream f(argv[6]); int64_t s; int c; while (f >> s >> c) conds.push_back({s, c}); }
    void* ctx = zmq_ctx_new();
    void* sub = zmq_socket(ctx, ZMQ_SUB); zmq_setsockopt(sub, ZMQ_SUBSCRIBE, "", 0);
    int hwm = 0; zmq_setsockopt(sub, ZMQ_RCVHWM, &hwm, sizeof(hwm));
    ZmqLink zmq("inproc://plugin-pub", "", nullptr, ctx);
    zmq_connect(sub, "inproc://plugin-pub");
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    std::atomic<bool> done{false};
    FILE* out = std::fopen(argv[7], "w");
    std::thread rx([&] {
        char buf[1 << 16];
        while (true) {
            std::string env, hdr; std::vector<char> data; int part = 0, more = 1;
            while (more) {
                zmq_msg_t m; zmq_msg_init(&m);
                int rc = zmq_msg_recv(&m, sub, ZMQ_DONTWAIT);
                if (rc < 0) { zmq_msg_close(&m); if (part == 0) { if (done) goto end; std::this_thread::sleep_for(std::chrono::milliseconds(2)); goto next; } continue; }
                const char* d = static_cast<const char*>(zmq_msg_data(&m)); size_t n = zmq_msg_size(&m);
                if (part == 0) env.assign(d, n); else if (part == 1) hdr.assign(d, n); else data.assign(d, d + n);
                size_t sz = sizeof(more); zmq_getsockopt(sub, ZMQ_RCVMORE, &more, &sz); zmq_msg_close(&m); ++part;
            }
            {
                std::string arr;
                if (env == "SPIKES" && !data.empty()) {
                    size_t n = data.size() / 12; arr = ",\"frames\":[";
                    for (size_t i = 0; i < n; ++i) { int64_t f; std::memcpy(&f, &data[i * 8], 8); arr += (i ? "," : "") + std::to_string(f); }
                    arr += "],\"units\":[";
                    for (size_t i = 0; i < n; ++i) { int32_t u; std::memcpy(&u, &data[n * 8 + i * 4], 4); arr += (i ? "," : "") + std::to_string(u); }
                    arr += "]";
                }
                std::fprintf(out, "{\"env\":\"%s\",\"hdr\":%s%s}\n", env.c_str(), hdr.c_str(), arr.c_str());
            }
        next:;
        }
    end:;
        (void)buf;
    });
    StreamDecoder dec("Neuropix-PXI-104.ProbeA-AP", 1, 29970.0757, argv[1], 1, !ttl.empty(), &zmq);
    if (!dec.loadOk()) { std::cerr << "load failed: " << dec.status() << "\n"; return 1; }
    std::cerr << "bank: " << dec.status() << "\n";
    dec.start();
    const int C = dec.channels();
    std::ifstream f(argv[2], std::ios::binary);
    std::vector<int16_t> blk(size_t(B) * nf);
    size_t ti = 0, ci = 0; int64_t s0 = 0, k = 0;
    while (true) {
        f.read(reinterpret_cast<char*>(blk.data()), std::streamsize(blk.size() * 2));
        int64_t n = f.gcount() / (2 * nf);
        if (n <= 0) break;
        const double now = steadySeconds(); const int64_t wall = wallMicros();
        while (ci < conds.size() && conds[ci].first < s0 + n) { dec.setCondition(conds[ci].second); ++ci; }
        while (ti < ttl.size() && ttl[ti] < s0 + n) { dec.addTtl(k, now); char h[200];
            std::snprintf(h, sizeof(h), "{\"type\":\"event\",\"stream\":\"NI\",\"line\":5,\"state\":1,\"sample_num\":%lld,\"ttl_index\":%lld,\"t_wall_us\":%lld}",
                          (long long)ttl[ti], (long long)k, (long long)wall);
            zmq.send("EVENT", h); ++k; ++ti; }
        Block b; b.s0 = s0; b.n = n; b.t_arrival = now; b.wall_us = wall; b.rows.resize(size_t(n) * C);
        for (int64_t i = 0; i < n; ++i) for (int c = 0; c < C; ++c) {
            // GUI path: int16 -> float uV (x * bitVolts) -> plugin rounds back to counts
            float uv = float(blk[size_t(i) * nf + c]) * 0.195f;
            b.rows[size_t(i) * C + c] = std::nearbyint(uv * (1.0f / 0.195f));
        }
        dec.enqueue(std::move(b));
        s0 += n;
        std::this_thread::sleep_for(std::chrono::microseconds(int64_t(1e6 * n / 29970.0757 / 4)));   // 4x real time
    }
    dec.stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    done = true; rx.join(); std::fclose(out);
    std::cerr << "sent " << zmq.sent() << " messages, overruns " << dec.overruns() << "\n";
    zmq_close(sub);
    return 0;
}
