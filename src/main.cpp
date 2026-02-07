#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <onnxruntime_cxx_api.h>

#include "app_config.h"
#include "post_process_out_string.h"
#include "image_preprocess.h"
#include "onnx_runner.h"

#include "utils/cli_args.h"
#include "utils/ocr_report.h"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
	try {
		const fs::path model_path = app_config::kModelPath;
		fs::path image_path;
		try {
			const auto opt = cli_args::Parse(argc, argv);
			if (opt.show_help) {
				cli_args::PrintUsage(argv[0], std::cout);
				return 0;
			}
			image_path = opt.image_path;
		} catch (const std::exception& e) {
			std::cerr << e.what() << "\n";
			cli_args::PrintUsage(argv[0], std::cout);
			return 2;
		}

		if (image_path.empty()) {
			image_path = app_config::kDefaultImagePath;
			std::cout << "(note) Khong truyen --image, dung anh mac dinh: " << image_path.string() << "\n";
		}

		if (!fs::exists(model_path)) {
			throw std::runtime_error("Khong tim thay model: " + model_path.string());
		}
		if (!fs::exists(image_path)) {
			throw std::runtime_error("Khong tim thay anh: " + image_path.string());
		}

		// 1) Tiền xử lý ảnh
		cv::Mat resized_rgb = image_preprocess::DocVaTienXuLyAnh_RGB_U8_HWC(
			image_path,
			app_config::kInputW,
			app_config::kInputH);

		// 2) Chạy ONNX Runtime -> lấy chuỗi argmax theo thời gian
		Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "main");
		auto r = onnx_runner::ChayModelVaLayArgMaxVaConf(
			env,
			model_path.string(),
			reinterpret_cast<const uint8_t*>(resized_rgb.data),
			app_config::kInputH,
			app_config::kInputW,
			app_config::kInputC);
		const auto& indices = r.indices;
		const auto& conf = r.conf;

		// 3) Hậu xử lý -> biển số (xóa '_' ở cuối)
		const int64_t blank_index = static_cast<int64_t>(app_config::kAlphabet.size()) - 1;
		const std::string decoded = post_process_out_string::PostProcessOutString(
			indices,
			app_config::kAlphabet,
			blank_index);
		ocr_report::PrintResult(std::cout, decoded, indices, conf, app_config::kAlphabet, 9);
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "Loi: " << e.what() << "\n";
		return 1;
	}
}