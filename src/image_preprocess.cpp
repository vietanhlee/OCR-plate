#include "image_preprocess.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <stdexcept>
#include <string>

namespace image_preprocess {

cv::Mat DocVaTienXuLyAnh_RGB_U8_HWC(const std::filesystem::path& image_path, int target_w, int target_h) {
    cv::Mat bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    if (bgr.empty()) {
        throw std::runtime_error("Không đọc được ảnh: " + image_path.string());
    }

    // OpenCV imread trả về BGR, mình đổi sang RGB cho giống infer.py
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(target_w, target_h), 0.0, 0.0, cv::INTER_LINEAR);

    if (resized.type() != CV_8UC3) {
        resized.convertTo(resized, CV_8UC3);
    }

    // Đảm bảo dữ liệu liên tục để đưa thẳng vào tensor
    if (!resized.isContinuous()) {
        resized = resized.clone();
    }

    return resized; // HWC, RGB, uint8
}

} // namespace image_preprocess
