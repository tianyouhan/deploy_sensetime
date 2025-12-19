cmake_minimum_required(VERSION 3.21)
project(multiscaleDeformableAttnPlugin LANGUAGES C CXX CUDA)

# ---------------------------------------
# 基础配置
# ---------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------------------------------
# 命令行可传变量（兼容 TensorRT 顶层调用）
# ---------------------------------------
macro(set_ifndef VAR VALUE)
    if(NOT DEFINED ${VAR})
        set(${VAR} ${VALUE})
    endif()
endmacro()

set_ifndef(GPU_ARCHS "87")
set_ifndef(CUDA_ARCHITECTURES ${GPU_ARCHS})
set_ifndef(CUDA_VERSION "12.8")
set_ifndef(TRT_PLATFORM_ID "aarch64")
set_ifndef(TRT_LIB_DIR "/usr/lib/${TRT_PLATFORM_ID}-linux-gnu")
set_ifndef(TRT_OUT_DIR "${CMAKE_BINARY_DIR}/out")

message(STATUS "GPU_ARCHS=${GPU_ARCHS}")
message(STATUS "CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}")
message(STATUS "CUDA_VERSION=${CUDA_VERSION}")
message(STATUS "TRT_LIB_DIR=${TRT_LIB_DIR}")
message(STATUS "TRT_OUT_DIR=${TRT_OUT_DIR}")

# ---------------------------------------
# CUDA 设置
# ---------------------------------------
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
find_package(CUDAToolkit REQUIRED)

# ---------------------------------------
# GPU 架构 SASS/PTX 生成
# ---------------------------------------
set(GENCODES "")
set(BERT_GENCODES "")

separate_arguments(GPU_ARCHS)  # 支持命令行传逗号分隔多个 arch
foreach(arch ${GPU_ARCHS})
    if(${arch} GREATER_EQUAL 75)
        set(BERT_GENCODES "${BERT_GENCODES} -gencode arch=compute_${arch},code=sm_${arch}")
    endif()
    set(GENCODES "${GENCODES} -gencode arch=compute_${arch},code=sm_${arch}")
endforeach()

# 为最新架构生成 PTX
list(GET GPU_ARCHS -1 LATEST_SM)
set(GENCODES "${GENCODES} -gencode arch=compute_${LATEST_SM},code=compute_${LATEST_SM}")
if(${LATEST_SM} GREATER_EQUAL 75)
    set(BERT_GENCODES "${BERT_GENCODES} -gencode arch=compute_${LATEST_SM},code=compute_${LATEST_SM}")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${GENCODES} -O3 --use_fast_math --expt-relaxed-constexpr -Xcompiler -Wno-deprecated-declarations")

# ---------------------------------------
# TensorRT 路径
# ---------------------------------------
set(TENSORRT_INCLUDE_DIRS
    /mnt/data/hantianyou/Compile/TensorRT/include
    /mnt/data/hantianyou/Compile/TensorRT/plugin
)
include_directories(
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIRS}
    /usr/include
)
link_directories(
    ${TRT_LIB_DIR}
    /usr/lib
)

# ---------------------------------------
# 源文件
# ---------------------------------------
set(SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/multiscaleDeformableAttn.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/multiscaleDeformableAttnPlugin.cpp
)

# ---------------------------------------
# 输出路径
# ---------------------------------------
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${TRT_OUT_DIR})

# ---------------------------------------
# 编译动态库
# ---------------------------------------
add_library(multiscaleDeformableAttnPlugin SHARED ${SRCS})

target_compile_definitions(multiscaleDeformableAttnPlugin PRIVATE -DNDEBUG)

target_link_libraries(multiscaleDeformableAttnPlugin
    PRIVATE nvinfer cudart nvinfer_plugin
)

set_target_properties(multiscaleDeformableAttnPlugin PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES}
    CUDA_SEPARABLE_COMPILATION ON
)
