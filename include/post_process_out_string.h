#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace post_process_out_string {

// Decode by mapping indices to characters and only stripping blank at the end.
//
// NOT CTC decoding:
// - Does not collapse repeats
// - Does not remove blanks in the middle
// - Only strips trailing blank/pad tokens (which may repeat)
std::string PostProcessOutString(const std::vector<int64_t>& indices,
								const std::string& alphabet,
								int64_t blank_index);

} // namespace post_process_out_string
