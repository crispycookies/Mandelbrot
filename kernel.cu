#include "kernel.cuh"
#include "math.h"

__device__ auto global_thread_idx_x() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__host__ __device__ float norm(cuFloatComplex & z)
{
    auto x = z.x * z.x;
    auto y = z.y * z.y;
    return sqrt(x+y);
}


__host__ __device__
int iterate(const cuFloatComplex & c) noexcept {
    auto i {0};
    cuFloatComplex z = {0};
    do{
        z = cuCmulf(z,z);
        z = cuCaddf(z,c);
    }while((i++ < g_colors) && ((int)norm(z)<g_infinity));
    return i;
}


__global__ void iterate_GPU(pfc::pixel_t * gpu_ptr, float x_fin, float x_start, float y_fin, float y_start, int height, int width) {
    size_t const current_idx = global_thread_idx_x();

    float dx = (x_fin - x_start)/(float)(width - 1);
    float dy = (y_fin - y_start)/(float)(height - 1);

    int x = (int)current_idx % width;
    int y = (int)current_idx / width;

    cuComplex c;
    c.x = {x_start + ((float)x)*dx};
    c.y = {y_fin - (float)y*dy};

    if (current_idx < height * width) {
        gpu_ptr[current_idx] = {pfc::byte_t(iterate(c)),0,0};
    }
}


cudaError_t call_iteration_kernel(pfc::pixel_t * gpu_ptr, std::complex<float> & left, std::complex<float>  & right, const std::complex<float>  & zoomPoint, int height, int width, float factor){

    auto const size{ static_cast <int> (height*width) };

    auto const  tib = 512;

    auto x_start = left.real();
    auto y_start = left.imag();
    auto x_fin = right.real();
    auto y_fin = right.imag();

    x_fin -= (x_fin - zoomPoint.real()) * (1-factor);
    y_fin -= (y_fin - zoomPoint.imag()) * (1-factor);
    x_start -= (x_start - zoomPoint.real()) * (1-factor);
    y_start -= (y_start - zoomPoint.imag()) * (1-factor);

    iterate_GPU <<<((size+tib-1)/tib),tib >>> (gpu_ptr,  x_fin, x_start, y_fin, y_start, height, width);

    left = {x_start, y_start};
    right = {x_fin, y_fin};

    cudaDeviceSynchronize();
    return cudaGetLastError();
}