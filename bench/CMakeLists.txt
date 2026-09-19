# Builds sweep_convs.cpp, the parameterized harness used to measure the
# published Im2win Tensor Core kernel at shapes other than its own benchmark
# layers. This is what produced the 37.0 TFLOPS kernel-only figure and the
# generality audit in Section 4.
#
# It links against the Im2win source tree, which is not vendored here. Place
# this file and sweep_convs.cpp in a build directory inside a clone of
# TensorConv/im2win, so that ../src/common and ../src/tensorCore resolve:
#
#   git clone https://github.com/TensorConv/im2win.git
#   mkdir im2win/sweepbuild
#   cp bench/CMakeLists.txt bench/sweep_convs.cpp im2win/sweepbuild/
#   cd im2win/sweepbuild
#   export CUDA_SOURCE_NAME=<the .cu for the kernel variant under test>
#   cmake -G Ninja . && ninja
#   ./sweep B Cin H W Cout K stride
#
# CMAKE_CUDA_ARCHITECTURES is 86 for the RTX 3090 these measurements used.
# Change it to match your device or the kernel will not launch.
cmake_minimum_required(VERSION 3.10)
project(sweep LANGUAGES CUDA CXX)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src/common)
find_package(CUDAToolkit REQUIRED)
set(CUDA_SOURCE_FILE "../src/tensorCore/$ENV{CUDA_SOURCE_NAME}")
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS -O3)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx2 -mfma -fopenmp")
set(CMAKE_CUDA_ARCHITECTURES "86")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_executable(sweep ./sweep_convs.cpp)
target_compile_features(sweep PRIVATE cxx_std_17)
enable_language(CUDA)
add_library(codeCUDA2 STATIC ${CUDA_SOURCE_FILE})
set_target_properties(codeCUDA2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(sweep PRIVATE CUDA::cublas CUDA::cudart codeCUDA2)
