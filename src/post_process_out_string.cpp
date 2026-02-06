#include "post_process_out_string.h"

namespace post_process_out_string {

std::string PostProcessOutString(const std::vector<int64_t>& indices,
                                const std::string& alphabet,
                                int64_t blank_index) {
    if (indices.empty() || alphabet.empty()) {
        return std::string();
    }

    int64_t end = static_cast<int64_t>(indices.size());

    // Strip trailing blanks only (blank/pad is expected at the end and may repeat).
    if (blank_index >= 0 && blank_index < static_cast<int64_t>(alphabet.size())) {
        while (end > 0 && indices[static_cast<size_t>(end - 1)] == blank_index) {
            --end;
        }
    }

    std::string out;
    out.reserve(static_cast<size_t>(end));

    for (int64_t i = 0; i < end; ++i) {
        const int64_t idx = indices[static_cast<size_t>(i)];
        if (idx < 0 || idx >= static_cast<int64_t>(alphabet.size())) {
            continue;
        }
        out.push_back(alphabet[static_cast<size_t>(idx)]);
    }

    return out;
}

} // namespace post_process_out_string
