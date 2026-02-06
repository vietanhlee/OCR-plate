#pragma once

#include <string>

namespace app_config {

// Model hiện có trong workspace: model/model_ocr_plate.onnx
inline constexpr const char* kModelPath = "../model/model_ocr_plate.onnx";

// Input shape : (1, 64, 128, 3) kiểu uint8, layout NHWC
inline constexpr int kInputH = 64;
inline constexpr int kInputW = 128;
inline constexpr int kInputC = 3;

// Ký tự cuối '_' là blank cho CTC.
inline const std::string kAlphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_";

// Ảnh mặc định để chạy nếu không truyền --image
inline constexpr const char* kDefaultImagePath = "../img/51V4579.jpg";

} // namespace app_config
