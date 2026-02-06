# OCR plate - ONNX Runtime (C++) trên Linux

Repo này chạy infer biển số bằng ONNX Runtime C++ (CPU) và OpenCV để đọc/resize ảnh.

## 1) Yêu cầu

- Linux (Ubuntu/Debian khuyến nghị)
- CMake >= 3.10
- GCC/G++ hỗ trợ C++17
- OpenCV (dev)

Cài OpenCV (Ubuntu/Debian):

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev
```

ONNX Runtime đã có sẵn trong thư mục `onnxruntime/` (đã gồm `include/` và `lib/`).

## 2) Build

Từ thư mục dự án:

```bash
rm -rf build
mkdir -p build
cd build
cmake -S .. -B . 
cmake --build . -j"$(nproc)"
```

Sau khi build xong sẽ có file `out/build/bin/main`.

## 3) Chạy

### Cách 1: Truyền ảnh đầu vào

```bash
../out/build/bin/main --image ../img/51F9846.jpg
```

```bash
../out/build/bin/main --image ../img/51F13251.jpg
```

```bash
../out/build/bin/main --image ../img/51F20754.jpg
```

```bash
../out/build/bin/main --image ../img/51H58092.jpg
```

```bash
../out/build/bin/main --image ../img/51V4579.jpg
```
### Cách 2: Không truyền `--image`

Chương trình sẽ dùng ảnh mặc định (được fix trong code):

- ../img/51V4579.jpg`

Chạy:

```bash
../out/build/bin/main
```

### Tuỳ chọn decode

Mặc định decode CTC sẽ **gộp ký tự lặp liên tiếp** (collapse repeats).
Nếu bạn muốn tắt gộp ký tự lặp:

```bash
../out/build/bin/main --image 51V4579.jpg --no_collapse
```

## 4) Cấu hình cố định trong code

Các cấu hình đang được fix cứng trong file:

- Model: `model/model_ocr_plate.onnx`
- Input: `(1, 64, 128, 3)` kiểu `uint8` layout `NHWC`


Bạn có thể đổi nhanh trong: `include/app_config.h`.

