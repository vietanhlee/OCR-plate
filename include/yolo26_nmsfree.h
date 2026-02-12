#pragma once

#include <cstdint>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

namespace yolo26_nmsfree {

struct Detection {
	float x1 = 0.0f;
	float y1 = 0.0f;
	float x2 = 0.0f;
	float y2 = 0.0f;
	float score = 0.0f;
	int cls = -1;
};

struct LetterboxInfo {
	int orig_w = 0;
	int orig_h = 0;
	int in_w = 0;
	int in_h = 0;
	float scale = 1.0f;
	int pad_x = 0;
	int pad_y = 0;
};

struct InputSpec {
	ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
	bool nchw = true; // true: NCHW, false: NHWC
	int64_t n = -1; // batch; -1 nghĩa là dynamic/không rõ
	int64_t c = 3;
	int64_t h = 640;
	int64_t w = 640;
};

// Chạy YOLOv26 end-to-end (NMS-free) để detect.
// Output kỳ vọng: (N, 300, 6) hoặc (300, 6), mỗi dòng: [x1,y1,x2,y2,score,cls]
// Tọa độ có thể là theo pixel của input hoặc normalized [0,1].
std::vector<std::vector<Detection>> RunBatch(
	Ort::Session& session,
	const std::vector<cv::Mat>& bgr_images,
	float conf_threshold);

} // namespace yolo26_nmsfree
