#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ctc_decode {

// Giải mã theo kiểu CTC:
// - Bỏ ký tự blank (mặc định là ký tự cuối cùng trong alphabet, ví dụ '_')
// - Nếu collapse_repeats=true: gộp các ký tự lặp liên tiếp
std::string GiaiMaCTC(const std::vector<int64_t>& indices, const std::string& alphabet, bool collapse_repeats);

} // namespace ctc_decode
