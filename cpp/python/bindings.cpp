// Python binding of the streamsort C++ core: module `streamsort_core`.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "streamsort/decoder.hpp"
#include "streamsort/pairing.hpp"
#include "streamsort/bank.hpp"

namespace py = pybind11;
using namespace streamsort;

template <class T> static std::vector<T> vec(const py::dict& d, const char* k) {
    auto a = py::array_t<T, py::array::c_style | py::array::forcecast>(d[k]);
    return std::vector<T>(a.data(), a.data() + a.size());
}

static DecoderTables tables_from_dict(const py::dict& d) {
    DecoderTables t;
    t.C = d["C"].cast<int>(); t.bit_volts = d["bit_volts"].cast<double>();
    t.sos = vec<double>(d, "sos"); t.n_sections = int(t.sos.size() / 6); t.zi_unit = vec<double>(d, "zi_unit");
    t.w_ptr = vec<int64_t>(d, "w_ptr"); t.w_row = vec<int64_t>(d, "w_row"); t.w_val = vec<float>(d, "w_val");
    t.thr = vec<float>(d, "thr"); t.nbr = vec<int64_t>(d, "nbr"); t.K = d["K"].cast<int>();
    for (auto k : {"dedup", "before_need", "after_need", "lag", "bpre", "bpost", "clip_counts", "clip_min_channels", "T", "Ks", "n_before"}) {
        int v = d[k].cast<int>();
        std::string s(k);
        if (s == "dedup") t.dedup = v; else if (s == "before_need") t.before_need = v; else if (s == "after_need") t.after_need = v;
        else if (s == "lag") t.lag = v; else if (s == "bpre") t.bpre = v; else if (s == "bpost") t.bpost = v;
        else if (s == "clip_counts") t.clip_counts = v; else if (s == "clip_min_channels") t.clip_min_channels = v;
        else if (s == "T") t.T = v; else if (s == "Ks") t.Ks = v; else t.n_before = v;
    }
    t.cand_ptr = vec<int64_t>(d, "cand_ptr"); t.cand_units = vec<int64_t>(d, "cand_units"); t.tro_cand = vec<int64_t>(d, "tro_cand");
    t.Tcat = vec<float>(d, "Tcat"); t.ttcat = vec<float>(d, "ttcat"); t.sset = vec<int64_t>(d, "sset"); t.shifts = vec<int64_t>(d, "shifts");
    t.amp_lo = d["amp_lo"].cast<double>(); t.amp_hi = d["amp_hi"].cast<double>(); t.min_energy = d["min_energy"].cast<float>();
    t.unit_ids = vec<int64_t>(d, "unit_ids");
    t.has_suppressor = d["has_suppressor"].cast<bool>();
    if (t.has_suppressor) {
        auto& s = t.sup;
        s.C = t.C;
        s.clip = d["sup_clip"].cast<int>(); s.kmin = d["sup_kmin"].cast<int>(); s.L = d["sup_L"].cast<int>();
        s.pre = d["sup_pre"].cast<int>(); s.post = d["sup_post"].cast<int>(); s.maxspan = d["sup_maxspan"].cast<int>();
        s.nb = d["sup_nb"].cast<int>(); s.n_history = d["sup_n_history"].cast<int>(); s.min_history = d["sup_min_history"].cast<int>();
        s.taper = vec<float>(d, "sup_taper"); s.subtract = d["sup_subtract"].cast<bool>(); s.bridge = d["sup_bridge"].cast<bool>();
    }
    return t;
}

static py::dict spikes_to_dict(const std::vector<Spike>& v) {
    size_t n = v.size();
    py::array_t<int64_t> f(n), u(n), pc(n);
    py::array_t<float> sc(n), am(n);
    auto F = f.mutable_data(); auto U = u.mutable_data(); auto P = pc.mutable_data();
    auto S = sc.mutable_data(); auto A = am.mutable_data();
    for (size_t i = 0; i < n; ++i) { F[i] = v[i].frame; U[i] = v[i].unit; S[i] = v[i].score; A[i] = v[i].amp; P[i] = v[i].peak_channel; }
    py::dict d; d["frame"] = f; d["unit"] = u; d["score"] = sc; d["amp"] = am; d["peak_channel"] = pc;
    return d;
}

class PyDecoder {
public:
    explicit PyDecoder(const py::dict& d, int n_threads) : dec_(tables_from_dict(d), n_threads) {}
    void start_at(int64_t s) { dec_.start_at(s); }
    py::dict push(py::array raw) {
        auto a = py::array_t<float, py::array::c_style | py::array::forcecast>(raw);
        if (a.ndim() != 2 || a.shape(1) != dec_.tables().C) throw std::invalid_argument("expected (n, C) buffer");
        std::vector<Spike> out;
        { py::gil_scoped_release rel; dec_.push(a.data(), a.shape(0), out); }
        return spikes_to_dict(out);
    }
    py::dict flush() { std::vector<Spike> out; { py::gil_scoped_release rel; dec_.flush(out); } return spikes_to_dict(out); }
    void set_condition(int k) { dec_.set_condition(k); }
    py::list events() {
        py::list l;
        if (auto* s = dec_.suppressor())
            for (auto& e : s->events) l.append(py::make_tuple(e.onset, e.span, e.cond, e.n_hist, e.subtracted));
        return l;
    }
    int64_t n_blanked() const { return dec_.n_blanked; }
    std::vector<double> timing() const { return std::vector<double>(dec_.timing, dec_.timing + 6); }
private:
    Decoder dec_;
};

PYBIND11_MODULE(streamsort_core, m) {
    m.doc() = "streamsort C++ decoder core";
    py::class_<PulsePairer>(m, "PulsePairer")
        .def(py::init<bool, double>(), py::arg("ttl_routed"), py::arg("tol_s") = 0.05)
        .def("add_ttl", &PulsePairer::add_ttl)
        .def("add_artifact", &PulsePairer::add_artifact)
        .def("pair", [](PulsePairer& p, double now) {
            py::list l; for (auto& q : p.pair(now)) l.append(py::make_tuple(q.onset_sample, q.t, q.ttl_index)); return l; });
    m.def("load_tables_check", [](const std::string& folder) {
        BankInfo info; auto t = load_tables(folder, &info);
        return py::make_tuple(t.C, info.n_units, info.n_usable, info.sample_rate, info.fingerprint); });
    py::class_<PyDecoder>(m, "Decoder")
        .def(py::init<const py::dict&, int>(), py::arg("tables"), py::arg("n_threads") = 1)
        .def("start_at", &PyDecoder::start_at)
        .def("push", &PyDecoder::push)
        .def("flush", &PyDecoder::flush)
        .def("set_condition", &PyDecoder::set_condition)
        .def("events", &PyDecoder::events)
        .def_property_readonly("n_blanked", &PyDecoder::n_blanked)
        .def("timing", &PyDecoder::timing);
}
