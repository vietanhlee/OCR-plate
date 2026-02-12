#include "yolo26_nmsfree.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <opencv2/imgproc.hpp>

namespace yolo26_nmsfree {
namespace {

cv::Mat LetterboxToSizeRGB(const cv::Mat& bgr, int target_w, int target_h, LetterboxInfo& info);

template <typename T>
void FillTensorFromRGB_NCHW(const std::vector<cv::Mat>& rgbs_u8, int h, int w, std::vector<T>& out, bool scale_01);

template <typename T>
void FillTensorFromRGB_NHWC(const std::vector<cv::Mat>& rgbs_u8, int h, int w, std::vector<T>& out, bool scale_01);

std::vector<std::vector<Detection>> ParseOutput(Ort::Value& out0, const std::vector<LetterboxInfo>& infos, float conf_threshold);

InputSpec GetInputSpec(Ort::Session& session) {
	Ort::AllocatorWithDefaultOptions allocator;
	auto type_info = session.GetInputTypeInfo(0);
	auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
	const auto elem_type = tensor_info.GetElementType();
	const auto shape = tensor_info.GetShape();
	if (shape.size() != 4) {
		throw std::runtime_error("YOLO input rank khong hop le (can 4), rank=" + std::to_string(shape.size()));
	}

	InputSpec spec;
	spec.type = elem_type;
	spec.n = shape[0];

	// Heuristic nhận dạng layout: NCHW nếu shape[1]==3, hoặc NHWC nếu shape[3]==3.
	const bool dim1_is_c = (shape[1] == 3);
	const bool dim3_is_c = (shape[3] == 3);
	if (dim1_is_c && !dim3_is_c) {
		spec.nchw = true;
		spec.c = 3;
		spec.h = shape[2];
		spec.w = shape[3];
	} else if (dim3_is_c && !dim1_is_c) {
		spec.nchw = false;
		spec.c = 3;
		spec.h = shape[1];
		spec.w = shape[2];
	} else {
		// Fallback: mặc định NCHW.
		spec.nchw = true;
		spec.c = (shape[1] > 0 ? shape[1] : 3);
		spec.h = shape[2];
		spec.w = shape[3];
	}

	if (spec.c != 3) {
		throw std::runtime_error("YOLO input khong ho tro so kenh != 3");
	}
	if (spec.h <= 0 || spec.w <= 0) {
		// Shape động: default 640x640
		spec.h = 640;
		spec.w = 640;
	}
	return spec;
}

std::vector<std::vector<Detection>> RunBatchNoSplit(
	Ort::Session& session,
	const std::vector<cv::Mat>& bgr_images,
	float conf_threshold) {
	if (bgr_images.empty()) {
		return {};
	}

	const InputSpec spec = GetInputSpec(session);
	const int in_h = static_cast<int>(spec.h);
	const int in_w = static_cast<int>(spec.w);

	std::vector<cv::Mat> rgbs;
	rgbs.reserve(bgr_images.size());
	std::vector<LetterboxInfo> infos;
	infos.reserve(bgr_images.size());
	for (const auto& bgr : bgr_images) {
		LetterboxInfo info;
		rgbs.push_back(LetterboxToSizeRGB(bgr, in_w, in_h, info));
		infos.push_back(info);
	}

	Ort::AllocatorWithDefaultOptions allocator;
	auto input_name_alloc = session.GetInputNameAllocated(0, allocator);
	const char* input_name = input_name_alloc.get();

	const size_t output_count = session.GetOutputCount();
	if (output_count == 0) {
		throw std::runtime_error("YOLO model khong co output");
	}
	std::vector<Ort::AllocatedStringPtr> output_name_alloc;
	std::vector<const char*> output_names;
	output_name_alloc.reserve(output_count);
	output_names.reserve(output_count);
	for (size_t i = 0; i < output_count; ++i) {
		output_name_alloc.push_back(session.GetOutputNameAllocated(i, allocator));
		output_names.push_back(output_name_alloc.back().get());
	}

	Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	const int64_t batch = static_cast<int64_t>(rgbs.size());

	std::vector<int64_t> input_shape;
	if (spec.nchw) {
		input_shape = {batch, 3, spec.h, spec.w};
	} else {
		input_shape = {batch, spec.h, spec.w, 3};
	}

	std::vector<float> input_f32;
	std::vector<uint8_t> input_u8;
	Ort::Value input_tensor{nullptr};
	if (spec.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
		if (spec.nchw) {
			FillTensorFromRGB_NCHW(rgbs, in_h, in_w, input_f32, /*scale_01=*/true);
		} else {
			FillTensorFromRGB_NHWC(rgbs, in_h, in_w, input_f32, /*scale_01=*/true);
		}
		input_tensor = Ort::Value::CreateTensor<float>(
			mem_info,
			input_f32.data(),
			input_f32.size(),
			input_shape.data(),
			input_shape.size());
	} else if (spec.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
		if (spec.nchw) {
			FillTensorFromRGB_NCHW(rgbs, in_h, in_w, input_u8, /*scale_01=*/false);
		} else {
			const size_t per = static_cast<size_t>(in_h) * static_cast<size_t>(in_w) * 3ull;
			input_u8.resize(static_cast<size_t>(batch) * per);
			for (size_t i = 0; i < rgbs.size(); ++i) {
				std::memcpy(input_u8.data() + i * per, rgbs[i].data, per);
			}
		}
		input_tensor = Ort::Value::CreateTensor<uint8_t>(
			mem_info,
			input_u8.data(),
			input_u8.size(),
			input_shape.data(),
			input_shape.size());
	} else {
		throw std::runtime_error("YOLO input type chua ho tro (chi ho tro float32/uint8)");
	}

	const std::vector<const char*> input_names = {input_name};
	auto outputs = session.Run(
		Ort::RunOptions{nullptr},
		input_names.data(),
		&input_tensor,
		1,
		output_names.data(),
		output_names.size());

	Ort::Value& out0 = outputs.at(0);
	return ParseOutput(out0, infos, conf_threshold);
}

cv::Mat LetterboxToSizeRGB(const cv::Mat& bgr, int target_w, int target_h, LetterboxInfo& info) {
	if (bgr.empty()) {
		throw std::runtime_error("Anh rong");
	}
	info.orig_w = bgr.cols;
	info.orig_h = bgr.rows;
	info.in_w = target_w;
	info.in_h = target_h;

	const float r = std::min(static_cast<float>(target_w) / static_cast<float>(info.orig_w),
						static_cast<float>(target_h) / static_cast<float>(info.orig_h));
	info.scale = r;

	const int new_w = static_cast<int>(std::round(info.orig_w * r));
	const int new_h = static_cast<int>(std::round(info.orig_h * r));
	info.pad_x = (target_w - new_w) / 2;
	info.pad_y = (target_h - new_h) / 2;

	cv::Mat resized;
	cv::resize(bgr, resized, cv::Size(new_w, new_h), 0.0, 0.0, cv::INTER_LINEAR);

	cv::Mat padded(target_h, target_w, CV_8UC3, cv::Scalar(114, 114, 114));
	resized.copyTo(padded(cv::Rect(info.pad_x, info.pad_y, new_w, new_h)));

	cv::Mat rgb;
	cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
	if (!rgb.isContinuous()) {
		rgb = rgb.clone();
	}
	return rgb;
}

Detection ParseRow(const float* row, int in_w, int in_h) {
	float a = row[0];
	float b = row[1];
	float c = row[2];
	float d = row[3];

	// Nếu tọa độ normalized thì scale lên theo pixel input trước.
	const float max_xy = std::max({std::abs(a), std::abs(b), std::abs(c), std::abs(d)});
	if (max_xy <= 1.5f) {
		a *= static_cast<float>(in_w);
		c *= static_cast<float>(in_w);
		b *= static_cast<float>(in_h);
		d *= static_cast<float>(in_h);
	}

	float x1 = a;
	float y1 = b;
	float x2 = c;
	float y2 = d;
	// Nếu không phải xyxy thì hiểu là cxcywh.
	if (x2 < x1 || y2 < y1) {
		const float cx = a;
		const float cy = b;
		const float w = c;
		const float h = d;
		x1 = cx - w / 2.0f;
		y1 = cy - h / 2.0f;
		x2 = cx + w / 2.0f;
		y2 = cy + h / 2.0f;
	}

	Detection det;
	det.x1 = x1;
	det.y1 = y1;
	det.x2 = x2;
	det.y2 = y2;
	det.score = row[4];
	det.cls = static_cast<int>(std::lround(row[5]));
	return det;
}

Detection MapBackToOriginal(const Detection& in, const LetterboxInfo& info) {
	Detection out = in;
	out.x1 = (out.x1 - static_cast<float>(info.pad_x)) / info.scale;
	out.y1 = (out.y1 - static_cast<float>(info.pad_y)) / info.scale;
	out.x2 = (out.x2 - static_cast<float>(info.pad_x)) / info.scale;
	out.y2 = (out.y2 - static_cast<float>(info.pad_y)) / info.scale;

	out.x1 = std::clamp(out.x1, 0.0f, static_cast<float>(info.orig_w - 1));
	out.y1 = std::clamp(out.y1, 0.0f, static_cast<float>(info.orig_h - 1));
	out.x2 = std::clamp(out.x2, 0.0f, static_cast<float>(info.orig_w - 1));
	out.y2 = std::clamp(out.y2, 0.0f, static_cast<float>(info.orig_h - 1));
	return out;
}

float IoU(const Detection& a, const Detection& b) {
	const float xx1 = std::max(a.x1, b.x1);
	const float yy1 = std::max(a.y1, b.y1);
	const float xx2 = std::min(a.x2, b.x2);
	const float yy2 = std::min(a.y2, b.y2);
	const float w = std::max(0.0f, xx2 - xx1);
	const float h = std::max(0.0f, yy2 - yy1);
	const float inter = w * h;
	const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
	const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
	const float uni = area_a + area_b - inter;
	if (uni <= 0.0f) {
		return 0.0f;
	}
	return inter / uni;
}

std::vector<Detection> DedupNearIdenticalDetections(std::vector<Detection> dets) {
	if (dets.size() <= 1) {
		return dets;
	}

	std::sort(dets.begin(), dets.end(), [](const Detection& lhs, const Detection& rhs) {
		return lhs.score > rhs.score;
	});

	std::vector<Detection> kept;
	kept.reserve(dets.size());
	for (const auto& d : dets) {
		bool duplicated = false;
		for (const auto& k : kept) {
			const float iou = IoU(d, k);
			if (iou >= 0.98f) {
				duplicated = true;
				break;
			}
		}
		if (!duplicated) {
			kept.push_back(d);
		}
	}
	return kept;
}

template <typename T>
void FillTensorFromRGB_NCHW(const std::vector<cv::Mat>& rgbs_u8, int h, int w, std::vector<T>& out, bool scale_01) {
	const size_t n = rgbs_u8.size();
	out.resize(n * 3ull * static_cast<size_t>(h) * static_cast<size_t>(w));
	for (size_t i = 0; i < n; ++i) {
		const uint8_t* p = rgbs_u8[i].ptr<uint8_t>(0);
		for (int yy = 0; yy < h; ++yy) {
			for (int xx = 0; xx < w; ++xx) {
				const size_t idx_hwc = (static_cast<size_t>(yy) * static_cast<size_t>(w) + static_cast<size_t>(xx)) * 3ull;
				const uint8_t r = p[idx_hwc + 0];
				const uint8_t g = p[idx_hwc + 1];
				const uint8_t b = p[idx_hwc + 2];
				const size_t base = i * 3ull * static_cast<size_t>(h) * static_cast<size_t>(w);
				const size_t plane = static_cast<size_t>(h) * static_cast<size_t>(w);
				const size_t pos = static_cast<size_t>(yy) * static_cast<size_t>(w) + static_cast<size_t>(xx);
				if constexpr (std::is_same_v<T, float>) {
					const float k = scale_01 ? (1.0f / 255.0f) : 1.0f;
					out[base + 0 * plane + pos] = static_cast<float>(r) * k;
					out[base + 1 * plane + pos] = static_cast<float>(g) * k;
					out[base + 2 * plane + pos] = static_cast<float>(b) * k;
				} else {
					out[base + 0 * plane + pos] = static_cast<T>(r);
					out[base + 1 * plane + pos] = static_cast<T>(g);
					out[base + 2 * plane + pos] = static_cast<T>(b);
				}
			}
		}
	}
}

template <typename T>
void FillTensorFromRGB_NHWC(const std::vector<cv::Mat>& rgbs_u8, int h, int w, std::vector<T>& out, bool scale_01) {
	const size_t n = rgbs_u8.size();
	out.resize(n * static_cast<size_t>(h) * static_cast<size_t>(w) * 3ull);
	for (size_t i = 0; i < n; ++i) {
		const uint8_t* p = rgbs_u8[i].ptr<uint8_t>(0);
		const size_t base = i * static_cast<size_t>(h) * static_cast<size_t>(w) * 3ull;
		if constexpr (std::is_same_v<T, float>) {
			const float k = scale_01 ? (1.0f / 255.0f) : 1.0f;
			for (size_t j = 0; j < static_cast<size_t>(h) * static_cast<size_t>(w) * 3ull; ++j) {
				out[base + j] = static_cast<float>(p[j]) * k;
			}
		} else {
			for (size_t j = 0; j < static_cast<size_t>(h) * static_cast<size_t>(w) * 3ull; ++j) {
				out[base + j] = static_cast<T>(p[j]);
			}
		}
	}
}

std::vector<std::vector<Detection>> ParseOutput(Ort::Value& out0, const std::vector<LetterboxInfo>& infos, float conf_threshold) {
	if (!out0.IsTensor()) {
		throw std::runtime_error("YOLO output[0] khong phai tensor");
	}
	auto type_info = out0.GetTensorTypeAndShapeInfo();
	const auto shape = type_info.GetShape();
	const auto elem_type = type_info.GetElementType();

	int64_t batch = 1;
	int64_t rows = 0;
	int64_t cols = 0;
	if (shape.size() == 3) {
		batch = shape[0];
		rows = shape[1];
		cols = shape[2];
	} else if (shape.size() == 2) {
		batch = 1;
		rows = shape[0];
		cols = shape[1];
	} else {
		throw std::runtime_error("YOLO output shape khong hop le (can rank 2/3), rank=" + std::to_string(shape.size()));
	}
	if (cols != 6) {
		throw std::runtime_error("YOLO output cols != 6, cols=" + std::to_string(cols));
	}
	if (rows <= 0) {
		throw std::runtime_error("YOLO output rows khong hop le");
	}
	if (static_cast<size_t>(batch) != infos.size()) {
		throw std::runtime_error("YOLO output batch != so anh input");
	}
	if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
		throw std::runtime_error("YOLO output type chua ho tro (chi ho tro float32)");
	}

	const float* data = out0.GetTensorData<float>();
	std::vector<std::vector<Detection>> all;
	all.resize(static_cast<size_t>(batch));
	for (int64_t n = 0; n < batch; ++n) {
		std::vector<Detection> dets;
		dets.reserve(static_cast<size_t>(rows));
		const float* base = (shape.size() == 3) ? (data + n * rows * cols) : data;
		for (int64_t r = 0; r < rows; ++r) {
			const float* row = base + r * cols;
			const float score = row[4];
			if (!(score >= conf_threshold)) {
				continue;
			}
			Detection in_box = ParseRow(row, infos[static_cast<size_t>(n)].in_w, infos[static_cast<size_t>(n)].in_h);
			Detection mapped = MapBackToOriginal(in_box, infos[static_cast<size_t>(n)]);
			mapped.score = score;
			mapped.cls = static_cast<int>(std::lround(row[5]));
			if (mapped.x2 <= mapped.x1 || mapped.y2 <= mapped.y1) {
				continue;
			}
			dets.push_back(mapped);
		}
		all[static_cast<size_t>(n)] = DedupNearIdenticalDetections(std::move(dets));
	}
	return all;
}

} // namespace

std::vector<std::vector<Detection>> RunBatch(
	Ort::Session& session,
	const std::vector<cv::Mat>& bgr_images,
	float conf_threshold) {
	if (bgr_images.empty()) {
		return {};
	}

	const InputSpec spec = GetInputSpec(session);
	const int64_t fixed_batch = (spec.n > 0) ? spec.n : -1;
	if (fixed_batch > 0 && static_cast<int64_t>(bgr_images.size()) != fixed_batch) {
		// Chia batch thành các chunk theo fixed_batch (thường gặp: 1)
		std::vector<std::vector<Detection>> all;
		all.reserve(bgr_images.size());
		for (size_t i = 0; i < bgr_images.size(); ) {
			const size_t take = static_cast<size_t>(fixed_batch);
			const size_t end = std::min(bgr_images.size(), i + take);
			if (end - i != take) {
				// Nếu model fix batch (vd: 1) thì chỉ chạy được chunk đủ size.
				// Fallback: chạy từng ảnh đơn.
				for (; i < bgr_images.size(); ++i) {
					auto one = RunBatchNoSplit(session, {bgr_images[i]}, conf_threshold);
					all.push_back(std::move(one.at(0)));
				}
				break;
			}
			std::vector<cv::Mat> chunk;
			chunk.reserve(take);
			for (size_t j = i; j < end; ++j) {
				chunk.push_back(bgr_images[j]);
			}
			auto out = RunBatchNoSplit(session, chunk, conf_threshold);
			for (auto& v : out) {
				all.push_back(std::move(v));
			}
			i = end;
		}
		return all;
	}

	return RunBatchNoSplit(session, bgr_images, conf_threshold);
}

} // namespace yolo26_nmsfree
