cmake_minimum_required(VERSION 3.10)

set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")

set(CUDA_ROOT "/usr/local/cuda")
set(CMAKE_CUDA_COMPILER "${CUDA_ROOT}/bin/nvcc")

project(tccl LANGUAGES CXX CUDA VERSION 0.1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_compile_options(-O3 -std=c++14 -g -Wall -Wno-unused-result)
# Resolve "nvcc fatal : Unknown option '-Wall'"
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-forward-unknown-to-host-compiler>)

# Setup CUDA
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
include_directories(${CUDA_ROOT}/include)
link_directories(${CUDA_ROOT}/lib64)
set(CMAKE_CUDA_ARCHITECTURES 70 80)
# disable warning related to CUDA_ROOT
cmake_policy(SET CMP0074 NEW)
find_package(CUDA)

include_directories("deps/spdlog-1.11.0/include")
include_directories("deps/json-3.11.2/include")
include_directories("deps/tinyxml2-9.0.0")
include_directories("/opt/mellanox/sharp/include")
include_directories("/home/n1/heehoon/tccl-exp/nccl-2.18.3-1/build/include")
link_directories("/opt/mellanox/sharp/lib")
link_directories("/home/n1/heehoon/tccl-exp/nccl-2.18.3-1/build/lib")

# print all include directories
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
message(STATUS "INCLUDE_DIRECTORIES='${dirs}'")
# print all link directories
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY LINK_DIRECTORIES)
message(STATUS "LINK_DIRECTORIES='${dirs}'")

# tinyxml2
add_subdirectory(deps/tinyxml2-9.0.0)

add_subdirectory(src)
add_subdirectory(tests)