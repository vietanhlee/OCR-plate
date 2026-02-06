## Build & run OCR plate (C++/CMake) with OpenCV + bundled ONNX Runtime
## Usage:
##   docker build -t ocr-plate .
##   docker run --rm -v "$PWD/img:/app/img" ocr-plate --image /app/img/51V4579.jpg

FROM ubuntu:22.04 AS build

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Copy only what we need to build. (Keeps rebuilds fast when changing code.)
COPY CMakeLists.txt ./
COPY include ./include
COPY src ./src
COPY model ./model
COPY third_party ./third_party

RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build -j"$(nproc)"


FROM ubuntu:22.04 AS runtime

ARG DEBIAN_FRONTEND=noninteractive

# Runtime deps for OpenCV + ONNX Runtime
# Avoid hardcoding Ubuntu versioned OpenCV runtime packages (names vary across distros).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-dev \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# App binary
COPY --from=build /src/out/build/bin/main /app/main

# Model + ONNX Runtime shared library
COPY --from=build /src/model /app/model
COPY --from=build /src/third_party/onnxruntime/lib /app/third_party/onnxruntime/lib

# Optional sample images (can be overridden with -v $PWD/img:/app/img)
COPY img /app/img

ENV LD_LIBRARY_PATH=/app/third_party/onnxruntime/lib

ENTRYPOINT ["/app/main"]
