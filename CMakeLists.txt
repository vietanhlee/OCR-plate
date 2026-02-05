cmake_minimum_required(VERSION 3.10)
project(onnx_infer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/onnxruntime")

find_package(OpenCV REQUIRED)

add_executable(onnx_infer
	main.cpp
	src/image_preprocess.cpp
	src/ctc_decode.cpp
	src/onnx_runner.cpp
)

target_include_directories(onnx_infer PRIVATE
	"${CMAKE_SOURCE_DIR}/include"
	"${ONNXRUNTIME_DIR}/include"
	"${OpenCV_INCLUDE_DIRS}"
)

target_link_libraries(onnx_infer PRIVATE
	"${ONNXRUNTIME_DIR}/lib/libonnxruntime.so"
	${OpenCV_LIBS}
	pthread
	dl
)

set_target_properties(onnx_infer PROPERTIES
	BUILD_RPATH "${ONNXRUNTIME_DIR}/lib"
	INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib"
)
