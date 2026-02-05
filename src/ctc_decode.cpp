#include "ctc_decode.h"

namespace ctc_decode {

std::string GiaiMaCTC(const std::vector<int64_t>& indices, const std::string& alphabet, bool collapse_repeats) {
    const int64_t blank = static_cast<int64_t>(alphabet.size()) - 1; // '_' là blank

    std::string out;
    out.reserve(indices.size());

    int64_t prev = -1;
    for (int64_t idx : indices) {
        if (idx == blank) {
            prev = idx;
            continue;
        }

        if (collapse_repeats && idx == prev) {
            continue;
        }

        if (idx < 0 || idx >= static_cast<int64_t>(alphabet.size())) {
            prev = idx;
            continue;
        }

        out.push_back(alphabet[static_cast<size_t>(idx)]);
        prev = idx;
    }

    return out;
}

} // namespace ctc_decode
