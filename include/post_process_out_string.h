#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace post_process_out_string {

// Simple post-processing:
// - Map indices to characters by `alphabet`
// - Strip trailing blank/pad tokens only (expected to be '_' at the end)
std::string PostProcessOutString(const std::vector<int64_t>& indices,
								const std::string& alphabet,
								int64_t blank_index);

} // namespace post_process_out_string
