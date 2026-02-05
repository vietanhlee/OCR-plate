#pragma once

#include <string>

namespace app_config {

// ================== Cấu hình cố định ==================
// Bạn nói đã “fix cứng” sẵn nên mình để cố định tại đây.
// Model hiện có trong workspace: model.onnx
inline constexpr const char* kModelPath = "model.onnx";

// Input shape giống infer.py: (1, 64, 128, 3) kiểu uint8, layout NHWC
inline constexpr int kInputH = 64;
inline constexpr int kInputW = 128;
inline constexpr int kInputC = 3;

// Alphabet giống infer.py. Ký tự cuối '_' là blank cho CTC.
inline const std::string kAlphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_";

// Ảnh mặc định để chạy nhanh nếu bạn không truyền --image.
// (Nếu không tồn tại thì chương trình sẽ báo lỗi.)
inline constexpr const char* kDefaultImagePath = "51V4579.jpg";

} // namespace app_config
