cmake_minimum_required(VERSION 3.18)

project(max_search_comparison LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)
find_package(CUDAToolkit 12.0 REQUIRED)

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(benchmarks)