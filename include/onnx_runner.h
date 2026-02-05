#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace onnx_runner {

// Chạy model ONNX với input ảnh uint8 NHWC (1,H,W,3).
// Trả về chuỗi indices (argmax theo classes) theo chiều thời gian T.
std::vector<int64_t> ChayModelVaLayArgMax(
    Ort::Env& env,
    const std::string& model_path,
    const uint8_t* nhwc_u8,
    int h,
    int w,
    int c);

} // namespace onnx_runner
