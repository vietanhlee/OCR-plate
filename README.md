# OCR Plate Pipeline (ONNX Runtime C++ + OpenCV)

Pipeline hiện tại chạy theo luồng:

1. Detect phương tiện (YOLO26 NMS-free)
2. Crop phương tiện
3. Phân loại hãng xe trên crop phương tiện (multi-thread)
4. Detect biển số trên crop phương tiện (YOLO26 NMS-free, batch)
5. Crop biển số
6. OCR biển số (batch)
7. Vẽ bbox + nhãn lên ảnh kết quả

## 1) Yêu cầu môi trường

- Linux (khuyến nghị Ubuntu/Debian)
- CMake >= 3.10
- GCC/G++ hỗ trợ C++17+
- OpenCV dev

Cài OpenCV:

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev
```

ONNX Runtime đã được vendor sẵn trong `third_party/onnxruntime`.

## 2) Build

```bash
rm -rf build
mkdir -p build
cd build
cmake -S .. -B .
cmake --build . -j"$(nproc)"
```

Binary sau build: `out/build/bin/main`

## 3) Chạy

### Chạy 1 ảnh

```bash
cd build
../out/build/bin/main --image ../img/test1.jpg
```

### Chạy cả thư mục ảnh

```bash
cd build
../out/build/bin/main --folder ../img
```

Ghi chú:

- Chỉ dùng **một** trong hai tham số `--image` hoặc `--folder`.
- Nếu không truyền tham số nào, chương trình dùng ảnh mặc định trong `include/app_config.h`.

## 4) Model đang dùng

Khai báo tại `include/app_config.h`:

- `model/vehicle_detection.onnx`
- `model/plate_detection.onnx`
- `model/brand_car_classification.onnx`
- `model/model_ocr_plate.onnx`

## 5) Ngưỡng và input chính

Trong `include/app_config.h`:

- `kVehicleConfThresh = 0.3`
- `kPlateConfThresh = 0.5`
- `kOcrConfAvgThresh = 0.6`
- OCR input: `64x128` (RGB, uint8, NHWC)
- Brand input: `224x224` (theo code preprocess brand classifier)

## 6) Ý nghĩa hiển thị trên ảnh

- BBox phương tiện:
	- Xanh dương: phương tiện có ít nhất một biển số hợp lệ
	- Đỏ: phương tiện không có biển số hợp lệ
	- Nhãn: `vehicle_class`, `brand_id` (dạng `b<id>`), và `vehicle_conf`
- BBox biển số:
	- Hiển thị `text`, `plate_conf` (YOLO detect), `ocr_conf_avg`
	- Màu bbox biển số:
		- Xanh lá nếu `ocr_conf_avg >= 0.6`
		- Đỏ nếu `ocr_conf_avg < 0.6`

## 7) Output

Ảnh kết quả được ghi vào thư mục:

- `out/build/img_out`

Tên file output theo mẫu:

- `<ten_anh>_annotated.jpg`

Ngoài ảnh, chương trình cũng in log chi tiết (vehicle/brand/plate/OCR conf) ra stdout.