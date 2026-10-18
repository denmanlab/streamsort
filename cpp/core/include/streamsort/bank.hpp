// Load decoder tables exported by `streamsort bank` (folder of .npy arrays + decoder_tables.txt scalars).
#pragma once
#include <map>
#include <string>
#include "decoder.hpp"

namespace streamsort {

struct BankInfo {
    std::map<std::string, std::string> scalars;   // decoder_tables.txt
    double sample_rate = 0.0;
    int n_units = 0, n_usable = 0;
    std::string fingerprint;
};

// Throws std::runtime_error with a readable message if the folder is incomplete.
DecoderTables load_tables(const std::string& folder, BankInfo* info = nullptr);

}  // namespace streamsort
