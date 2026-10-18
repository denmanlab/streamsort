#include "streamsort/bank.hpp"
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace streamsort {
namespace {

struct Npy { std::string descr; std::vector<int64_t> shape; std::vector<char> data; };

Npy read_npy(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("missing " + path);
    char magic[6]; f.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error("not an npy file: " + path);
    unsigned char ver[2]; f.read(reinterpret_cast<char*>(ver), 2);
    uint32_t hl = 0;
    if (ver[0] == 1) { uint16_t h; f.read(reinterpret_cast<char*>(&h), 2); hl = h; }
    else { f.read(reinterpret_cast<char*>(&hl), 4); }
    std::string hdr(hl, ' '); f.read(&hdr[0], hl);
    Npy n;
    auto d0 = hdr.find("'descr':"); auto q1 = hdr.find('\'', d0 + 8); auto q2 = hdr.find('\'', q1 + 1);
    n.descr = hdr.substr(q1 + 1, q2 - q1 - 1);
    if (hdr.find("'fortran_order': True") != std::string::npos) throw std::runtime_error("fortran order not supported: " + path);
    auto s0 = hdr.find('(', hdr.find("'shape':")); auto s1 = hdr.find(')', s0);
    std::stringstream ss(hdr.substr(s0 + 1, s1 - s0 - 1)); std::string tok;
    while (std::getline(ss, tok, ',')) { if (tok.find_first_not_of(" ") != std::string::npos) n.shape.push_back(std::stoll(tok)); }
    int64_t count = 1; for (auto v : n.shape) count *= v;
    size_t item = (n.descr == "<f8" || n.descr == "<i8") ? 8 : (n.descr == "<f4" || n.descr == "<i4") ? 4 : (n.descr == "|b1" || n.descr == "|u1") ? 1 : 0;
    if (!item) throw std::runtime_error("unsupported dtype " + n.descr + " in " + path);
    n.data.resize(size_t(count) * item);
    f.read(n.data.data(), std::streamsize(n.data.size()));
    if (!f) throw std::runtime_error("truncated " + path);
    return n;
}

template <class T> std::vector<T> as(const Npy& n) {
    size_t count = 1; for (auto v : n.shape) count *= size_t(v);
    std::vector<T> out(count);
    for (size_t i = 0; i < count; ++i) {
        const char* p = n.data.data();
        if (n.descr == "<f8") { double v; std::memcpy(&v, p + i * 8, 8); out[i] = T(v); }
        else if (n.descr == "<f4") { float v; std::memcpy(&v, p + i * 4, 4); out[i] = T(v); }
        else if (n.descr == "<i8") { int64_t v; std::memcpy(&v, p + i * 8, 8); out[i] = T(v); }
        else if (n.descr == "<i4") { int32_t v; std::memcpy(&v, p + i * 4, 4); out[i] = T(v); }
        else { out[i] = T(uint8_t(p[i])); }
    }
    return out;
}

}  // namespace

DecoderTables load_tables(const std::string& folder, BankInfo* info) {
    std::map<std::string, std::string> kv;
    {
        std::ifstream f(folder + "/decoder_tables.txt");
        if (!f) throw std::runtime_error("no decoder_tables.txt in " + folder + " (run `streamsort bank --ks-folder ...`)");
        std::string line;
        while (std::getline(f, line)) {
            auto e = line.find('=');
            if (line.empty() || line[0] == '#' || e == std::string::npos) continue;
            kv[line.substr(0, e)] = line.substr(e + 1);
        }
    }
    auto I = [&](const char* k) { auto it = kv.find(k); if (it == kv.end()) throw std::runtime_error(std::string("missing scalar ") + k); return std::stoll(it->second); };
    auto D = [&](const char* k) { auto it = kv.find(k); if (it == kv.end()) throw std::runtime_error(std::string("missing scalar ") + k); return std::stod(it->second); };
    auto A = [&](const char* k) { return read_npy(folder + "/" + k + ".npy"); };
    DecoderTables t;
    t.C = int(I("C")); t.bit_volts = D("bit_volts"); t.K = int(I("K"));
    t.dedup = int(I("dedup")); t.before_need = int(I("before_need")); t.after_need = int(I("after_need"));
    t.lag = int(I("lag")); t.bpre = int(I("bpre")); t.bpost = int(I("bpost"));
    t.clip_counts = int(I("clip_counts")); t.clip_min_channels = int(I("clip_min_channels"));
    t.T = int(I("T")); t.Ks = int(I("Ks")); t.n_before = int(I("n_before"));
    t.amp_lo = D("amp_lo"); t.amp_hi = D("amp_hi"); t.min_energy = float(D("min_energy"));
    t.sos = as<double>(A("sos")); t.n_sections = int(t.sos.size() / 6); t.zi_unit = as<double>(A("zi_unit"));
    t.w_ptr = as<int64_t>(A("w_ptr")); t.w_row = as<int64_t>(A("w_row")); t.w_val = as<float>(A("w_val"));
    t.thr = as<float>(A("thr")); t.nbr = as<int64_t>(A("nbr"));
    t.cand_ptr = as<int64_t>(A("cand_ptr")); t.cand_units = as<int64_t>(A("cand_units")); t.tro_cand = as<int64_t>(A("tro_cand"));
    t.Tcat = as<float>(A("Tcat")); t.ttcat = as<float>(A("ttcat")); t.sset = as<int64_t>(A("sset")); t.shifts = as<int64_t>(A("shifts"));
    t.unit_ids = as<int64_t>(A("unit_ids"));
    t.has_suppressor = I("has_suppressor") != 0;
    if (t.has_suppressor) {
        auto& s = t.sup; s.C = t.C;
        s.clip = int(I("sup_clip")); s.kmin = int(I("sup_kmin")); s.L = int(I("sup_L")); s.pre = int(I("sup_pre"));
        s.post = int(I("sup_post")); s.maxspan = int(I("sup_maxspan")); s.nb = int(I("sup_nb"));
        s.n_history = int(I("sup_n_history")); s.min_history = int(I("sup_min_history"));
        s.subtract = I("sup_subtract") != 0; s.bridge = I("sup_bridge") != 0;
        s.taper = as<float>(A("sup_taper"));
    }
    if (int64_t(t.w_ptr.size()) != t.C + 1 || int64_t(t.thr.size()) != t.C || int64_t(t.cand_ptr.size()) != t.C + 1)
        throw std::runtime_error("decoder tables in " + folder + " are inconsistent with C=" + std::to_string(t.C));
    if (info) {
        info->scalars = kv;
        info->sample_rate = kv.count("sample_rate") ? std::stod(kv["sample_rate"]) : 0.0;
        info->n_units = kv.count("n_units") ? std::stoi(kv["n_units"]) : 0;
        info->n_usable = kv.count("n_usable") ? std::stoi(kv["n_usable"]) : 0;
        info->fingerprint = kv.count("fingerprint") ? kv["fingerprint"] : "";
    }
    return t;
}

}  // namespace streamsort
