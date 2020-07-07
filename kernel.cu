#include "kernel.cuh"
#include "math.h"

__constant__ const int height = 4608;
__constant__ const int width = 8192;

__device__ inline auto global_thread_idx_x() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ constexpr inline float norm(cuFloatComplex & z)
{
    auto x = z.x * z.x;
    auto y = z.y * z.y;
    return x+y;
}


__device__
int inline iterate(const cuFloatComplex & c) noexcept {
    auto i {0};
    cuFloatComplex z = {0};
#pragma unroll
    do{
        z = cuCmulf(z,z);
        z = cuCaddf(z,c);
    }while((i++ < g_colors) && ((int)norm(z)<g_infinity));
    return i;
}


__global__ void iterate_GPU(pfc::pixel_t * gpu_ptr,  float xleft, float yright, float  dx, float dy, float inv_width) {
    size_t const current_idx = (global_thread_idx_x());

    int x = (int)(current_idx) % width;
    int y = (int)((current_idx) * inv_width);

    cuComplex c;
    c.x = {xleft + ((float)x)*dx};
    c.y = {yright - (float)y*dy};

    if (current_idx < height*width) {
        gpu_ptr[current_idx].blue = pfc::byte_t(iterate(c));
    }
}


cudaError_t call_iteration_kernel(pfc::pixel_t * gpu_ptr, std::complex<float> left, std::complex<float>  right, const std::complex<float>  & zPoint, float factor, cudaStream_t * streams, int count){
    auto const size{ static_cast <int> (height*width) };

    auto const  tib = 128;

    auto xleft = left.real();
    auto yleft = left.imag();
    auto xright = right.real();
    auto yright = right.imag();

#pragma unroll
    for(int i = 1; i <= count; i++){

        xright -= (xright - zPoint.real()) * (1-factor);
        yright -= (yright - zPoint.imag()) * (1-factor);
        xleft -= (xleft - zPoint.real()) * (1-factor);
        yleft -= (yleft - zPoint.imag()) * (1-factor);
    }


    float dx = (xright - xleft)/(float)(width - 1);
    float dy = (yright - yleft)/(float)(height - 1);

    iterate_GPU <<<((size+tib-1)/(tib)),tib ,0, *streams>>> (gpu_ptr,  xleft, yright, dx, dy, (1./width));


    return cudaGetLastError();
}