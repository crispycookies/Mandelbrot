#ifndef CUDA_
#define CUDA_

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <complex>

#include "bitmap/pfc_types.h"

__constant__ int const g_colors = 128;
__constant__ int const g_infinity = {1225000000};

__device__
int iterate(const std::complex<float> & c) noexcept;
cudaError_t call_iteration_kernel(pfc::pixel_t * gpu_ptr, std::complex<float> & left, std::complex<float>  & right, const std::complex<float>  & zPoint,  int height, int width, float factor, cudaStream_t * streams, int count);

#endif