#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <onnxruntime_cxx_api.h>

#include "app_config.h"
#include "ctc_decode.h"
#include "image_preprocess.h"
#include "onnx_runner.h"

namespace fs = std::filesystem;

static void PrintUsage(const char* argv0) {
	std::cout
		<< "Cach dung:\n"
		<< "  " << argv0 << " --image <duong_dan_anh.jpg> [--no_collapse]\n\n"
		<< "Ghi chu:\n"
		<< "  - Model duoc fix cung: " << app_config::kModelPath << "\n"
		<< "  - Tien xu ly: doc anh -> RGB -> resize (" << app_config::kInputW << "x" << app_config::kInputH << ") -> uint8 NHWC\n"
		<< "  - Giai ma: CTC (bo blank '_', mac dinh co collapse ky tu lap; tat bang --no_collapse)\n";
}

int main(int argc, char** argv) {
	try {
		const fs::path model_path = app_config::kModelPath;
		fs::path image_path;
		bool collapse_repeats = true;

		for (int i = 1; i < argc; ++i) {
			std::string a = argv[i];
			if ((a == "--image" || a == "-i") && i + 1 < argc) {
				image_path = argv[++i];
			} else if (a == "--no_collapse") {
				collapse_repeats = false;
			} else if (a == "--help" || a == "-h") {
				PrintUsage(argv[0]);
				return 0;
			} else {
				std::cerr << "Tham so khong hop le: " << a << "\n";
				PrintUsage(argv[0]);
				return 2;
			}
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
		auto indices = onnx_runner::ChayModelVaLayArgMax(
			env,
			model_path.string(),
			reinterpret_cast<const uint8_t*>(resized_rgb.data),
			app_config::kInputH,
			app_config::kInputW,
			app_config::kInputC);

		// 3) Giải mã CTC -> biển số
		const std::string decoded = ctc_decode::GiaiMaCTC(indices, app_config::kAlphabet, collapse_repeats);
		std::cout << "Bien so: " << decoded << "\n";
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "Loi: " << e.what() << "\n";
		return 1;
	}
}
