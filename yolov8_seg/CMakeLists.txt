cmake_minimum_required(VERSION 3.18)

# 设置 CUDA 编译器路径
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.6/bin/nvcc)  # Jetson 上通常在这里

project(yolov8_seg LANGUAGES CXX CUDA)

# -------------------- 编译标准 --------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
add_definitions(-pthread)

# -------------------- OpenCV --------------------
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# -------------------- CUDA --------------------
# Jetson 上 CUDA 通常安装在标准位置
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

# -------------------- TensorRT --------------------
# Jetson 上的 TensorRT 路径
include_directories(/usr/include/aarch64-linux-gnu)
link_directories(/usr/lib/aarch64-linux-gnu)

# -------------------- 项目头文件 --------------------
include_directories(${PROJECT_SOURCE_DIR}/include)

# -------------------- 源文件 --------------------
file(GLOB_RECURSE CPP_FILES 
    src/*.cpp
)

file(GLOB_RECURSE CUDA_FILES 
    src/*.cu
)

# -------------------- 可执行文件 --------------------
add_executable(yolov8_seg ${CPP_FILES} ${CUDA_FILES})

# -------------------- CUDA 文件设置 --------------------
set_property(SOURCE ${CUDA_FILES} PROPERTY LANGUAGE CUDA)

# 添加 CUDA 编译选项
target_compile_options(yolov8_seg PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# -------------------- Jetson 特定设置 --------------------
# Jetson NX 使用 NVIDIA Volta 架构，计算能力 7.2
set_target_properties(yolov8_seg PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "72"  # Jetson NX 的计算能力
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# -------------------- 链接库 --------------------
target_link_libraries(yolov8_seg
    PRIVATE
    nvinfer
    nvonnxparser
    nvinfer_plugin
    cudart
    ${OpenCV_LIBS}
    pthread
)

# -------------------- 添加 CUDA 编译定义 --------------------
target_compile_definitions(yolov8_seg PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:__CUDACC__>
)

# -------------------- 性能优化设置 --------------------
# 针对 Jetson 的优化选项
target_compile_options(yolov8_seg PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--gpu-architecture=compute_72>
    $<$<COMPILE_LANGUAGE:CUDA>:--gpu-code=sm_72>
    $<$<COMPILE_LANGUAGE:CUDA>:--ftz=true>
    $<$<COMPILE_LANGUAGE:CUDA>:--prec-div=false>
    $<$<COMPILE_LANGUAGE:CUDA>:--prec-sqrt=false>
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=true>
)

# -------------------- 内存优化 --------------------
# 减少 Jetson 上的内存使用
target_compile_options(yolov8_seg PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--maxrregcount=32>
)

# -------------------- 确保 CUDA 文件正确编译 --------------------
set_source_files_properties(${CUDA_FILES} PROPERTIES
    CUDA_PTX_COMPILATION OFF
    CUDA_SEPARABLE_COMPILATION ON
)

# -------------------- 添加性能分析支持 --------------------
option(ENABLE_CUDA_PROFILING "Enable CUDA profiling" OFF)
if(ENABLE_CUDA_PROFILING)
    target_compile_definitions(yolov8_seg PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:ENABLE_PROFILING>
    )
endif()