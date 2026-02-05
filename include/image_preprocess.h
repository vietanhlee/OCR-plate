#pragma once

#include <filesystem>

#include <opencv2/core.hpp>

namespace image_preprocess {

// Đọc ảnh từ disk, chuyển sang RGB, resize về (W,H), trả về cv::Mat dạng HWC, RGB, uint8.
cv::Mat DocVaTienXuLyAnh_RGB_U8_HWC(const std::filesystem::path& image_path, int target_w, int target_h);

} // namespace image_preprocess
