cmake_minimum_required(VERSION 3.16)
project(Mandelbrot C CXX CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 17)
find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
find_package(Microsoft.GSL CONFIG)
add_executable(Mandelbrot main.cpp kernel.cuh kernel.cu misc/pfc_timing.h misc/pfc_threading.h misc/pfc_random.h)
target_link_libraries(Mandelbrot ${CUDA_LIBRARIES})

set_target_properties(
        Mandelbrot
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)