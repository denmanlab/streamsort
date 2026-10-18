// streamsort_core_test <tables_folder> <raw_int16.bin> <n_channels_in_file> <buffer> <out.csv>
// Decodes a raw int16 file (first C channels) through the C++-only path the plugin uses
// (table folder loader + Decoder) and writes frame,unit per spike.
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>
#include "streamsort/bank.hpp"

int main(int argc, char** argv) {
    if (argc < 6) { std::cerr << "usage: tables raw.bin nch_file buffer out.csv\n"; return 2; }
    try {
        streamsort::BankInfo info;
        auto t = streamsort::load_tables(argv[1], &info);
        streamsort::Decoder dec(t);
        dec.start_at(0);
        const int nf = std::stoi(argv[3]), buf = std::stoi(argv[4]), C = t.C;
        std::ifstream f(argv[2], std::ios::binary);
        std::vector<int16_t> blk(size_t(buf) * nf); std::vector<float> x(size_t(buf) * C);
        std::vector<streamsort::Spike> out;
        while (true) {
            f.read(reinterpret_cast<char*>(blk.data()), std::streamsize(blk.size() * 2));
            int64_t n = f.gcount() / (2 * nf);
            if (n <= 0) break;
            for (int64_t i = 0; i < n; ++i) for (int c = 0; c < C; ++c) x[i * C + c] = float(blk[i * nf + c]);
            dec.push(x.data(), n, out);
        }
        dec.flush(out);
        FILE* o = std::fopen(argv[5], "w");
        std::fprintf(o, "frame,unit\n");
        size_t n_assigned = 0;
        for (auto& s : out) if (s.unit >= 0) { std::fprintf(o, "%lld,%lld\n", (long long)s.frame, (long long)s.unit); ++n_assigned; }
        std::fclose(o);
        std::cout << "units " << info.n_usable << "/" << info.n_units << " spikes " << n_assigned
                  << " events " << (dec.suppressor() ? dec.suppressor()->events.size() : 0) << "\n";
    } catch (const std::exception& e) { std::cerr << "error: " << e.what() << "\n"; return 1; }
    return 0;
}
