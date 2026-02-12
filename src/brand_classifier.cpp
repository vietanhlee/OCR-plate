#include "brand_classifier.h"

#include <cmath>
#include <stdexcept>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace brand_classifier {
namespace {

void PrepareInputNchwFloat(const cv::Mat& bgr, int input_h, int input_w, std::vector<float>& out) {
	if (bgr.empty()) {
		throw std::runtime_error("Anh dau vao phan loai rong");
	}
	if (bgr.channels() != 3) {
		throw std::runtime_error("Anh dau vao phan loai can 3 kenh");
	}

	cv::Mat rgb;
	cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
	cv::Mat resized;
	cv::resize(rgb, resized, cv::Size(input_w, input_h), 0.0, 0.0, cv::INTER_LINEAR);
	if (resized.type() != CV_8UC3) {
		resized.convertTo(resized, CV_8UC3);
	}
	if (!resized.isContinuous()) {
		resized = resized.clone();
	}

	out.resize(3ull * static_cast<size_t>(input_h) * static_cast<size_t>(input_w));
	const uint8_t* p = resized.ptr<uint8_t>(0);
	const size_t plane = static_cast<size_t>(input_h) * static_cast<size_t>(input_w);
	for (int y = 0; y < input_h; ++y) {
		for (int x = 0; x < input_w; ++x) {
			const size_t idx_hwc = (static_cast<size_t>(y) * static_cast<size_t>(input_w) + static_cast<size_t>(x)) * 3ull;
			const size_t pos = static_cast<size_t>(y) * static_cast<size_t>(input_w) + static_cast<size_t>(x);
			out[0 * plane + pos] = static_cast<float>(p[idx_hwc + 0]) / 255.0f;
			out[1 * plane + pos] = static_cast<float>(p[idx_hwc + 1]) / 255.0f;
			out[2 * plane + pos] = static_cast<float>(p[idx_hwc + 2]) / 255.0f;
		}
	}
}

BrandResult DecodeTop1(const float* data, int64_t class_count) {
	if (class_count <= 0) {
		throw std::runtime_error("So class output khong hop le");
	}
	int best_idx = 0;
	float best_val = data[0];
	bool all_prob = (data[0] >= 0.0f && data[0] <= 1.0f);
	double sum_row = static_cast<double>(data[0]);
	for (int64_t i = 1; i < class_count; ++i) {
		const float v = data[i];
		if (v > best_val) {
			best_val = v;
			best_idx = static_cast<int>(i);
		}
		all_prob = all_prob && (v >= 0.0f && v <= 1.0f);
		sum_row += static_cast<double>(v);
	}

	float conf = 0.0f;
	const bool looks_like_prob = all_prob && (std::abs(sum_row - 1.0) <= 1e-2);
	if (looks_like_prob) {
		conf = best_val;
	} else {
		double sum_exp = 0.0;
		for (int64_t i = 0; i < class_count; ++i) {
			sum_exp += std::exp(static_cast<double>(data[i] - best_val));
		}
		conf = (sum_exp > 0.0) ? static_cast<float>(1.0 / sum_exp) : 0.0f;
	}

	BrandResult r;
	r.class_id = best_idx;
	r.conf = conf;
	return r;
}

} // namespace

BrandResult ClassifySingle(
	Ort::Session& session,
	const cv::Mat& bgr_image,
	int input_h,
	int input_w) {
	std::vector<float> input;
	PrepareInputNchwFloat(bgr_image, input_h, input_w, input);

	Ort::AllocatorWithDefaultOptions allocator;
	auto input_name_alloc = session.GetInputNameAllocated(0, allocator);
	const char* input_name = input_name_alloc.get();
	auto output_name_alloc = session.GetOutputNameAllocated(0, allocator);
	const char* output_name = output_name_alloc.get();

	const std::vector<int64_t> input_shape = {1, 3, input_h, input_w};
	Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
		mem_info,
		input.data(),
		input.size(),
		input_shape.data(),
		input_shape.size());

	const std::vector<const char*> input_names = {input_name};
	auto outputs = session.Run(
		Ort::RunOptions{nullptr},
		input_names.data(),
		&input_tensor,
		1,
		&output_name,
		1);

	Ort::Value& out0 = outputs.at(0);
	if (!out0.IsTensor()) {
		throw std::runtime_error("Output brand khong phai tensor");
	}
	const auto out_info = out0.GetTensorTypeAndShapeInfo();
	const auto out_type = out_info.GetElementType();
	if (out_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
		throw std::runtime_error("Output brand can float32");
	}
	const auto out_shape = out_info.GetShape();
	const float* data = out0.GetTensorData<float>();
	if (out_shape.size() == 2) {
		if (out_shape[0] != 1) {
			throw std::runtime_error("Output brand rank2 can batch=1 khi infer don");
		}
		return DecodeTop1(data, out_shape[1]);
	}
	if (out_shape.size() == 1) {
		return DecodeTop1(data, out_shape[0]);
	}
	throw std::runtime_error("Output brand rank khong hop le (can 1 hoac 2)");
}

} // namespace brand_classifier
