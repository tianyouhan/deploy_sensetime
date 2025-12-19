cmake_minimum_required(VERSION 3.21)

set(BT_CROSS_ROOT "/usr/local/thor/")
set(BT_CROSS_TOOLCHAIN "${BT_CROSS_ROOT}/aarch64--glibc--bleeding-edge-2024.02-1")
set(BT_SYSROOT "/")
set(SENSEAUTO_DEV_INSTALLED_ROOT /)
set(FIND_SENSEAUTO_PACKAGE_PATH /opt)

set(CUDA_ROOT_DIR /usr/local/thor/cuda-12.8/thor/targets/aarch64-linux)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/thor/cuda-12.8)
set(CUDA_TOOLKIT_TARGET_DIR ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux)
set(CUDAToolkit_ROOT ${CUDA_TOOLKIT_ROOT_DIR})

set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)
set(CMAKE_CUDA_HOST_COMPILER "/usr/local/thor/aarch64--glibc--bleeding-edge-2024.02-1/bin/aarch64-buildroot-linux-gnu-g++")
set(CMAKE_CXX_COMPILER "/usr/local/thor/aarch64--glibc--bleeding-edge-2024.02-1/bin/aarch64-buildroot-linux-gnu-g++")

set(LINUX_GNU /usr/local/thor/aarch64-linux-gnu)
set(LINUX_DRIVE /usr/local/thor/drive-linux)

set(TARGET_OS linux)
set(PLATFORM_NAME aarch64)

project(multiscaleDeformableAttnPlugin)
enable_language(C CXX CUDA)

set(SRCS ${CMAKE_CURRENT_SOURCE_DIR}/multiscaleDeformableAttn.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/multiscaleDeformableAttnPlugin.cpp
)

include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
include_directories(/usr/local/thor/aarch64-linux-gnu/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../plugin)

add_library(multiscaleDeformableAttnPlugin SHARED
    ${SRCS}
    )
set_target_properties(multiscaleDeformableAttnPlugin PROPERTIES
    CUDA_ARCHITECTURES "101"
)
