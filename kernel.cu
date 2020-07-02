#include <iostream>
#include "kernel.cuh"
#include "math.h"

__device__ auto global_thread_idx_x() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ float norm(cuFloatComplex & z)
{
    auto x = z.x * z.x;
    auto y = z.y * z.y;
    return sqrt(x+y);
}


__device__
int iterate(const cuFloatComplex & c) noexcept {
    auto i {0};
    cuFloatComplex z = {0};
    do{
        z = cuCmulf(z,z);
        z = cuCaddf(z,c);
    }while((i++ < g_colors) && ((int)norm(z)<g_infinity));



    return i;
}


__global__ void iterate_GPU(pfc::pixel_t * gpu_ptr, float xright, float xleft, float yright, float yleft, const float x_zp, const float y_zp, int height, int width, const size_t count, float factor) {
    size_t const current_idx = global_thread_idx_x();
    size_t const iteration_idx = current_idx/(height*width);

    for(size_t i = 0; i < iteration_idx; i++){
        xright -= (xright - x_zp) * (1-factor);
        yright -= (yright - y_zp) * (1-factor);
        xleft -= (xleft - x_zp) * (1-factor);
        yleft -= (yleft - y_zp) * (1-factor);
    }

    float dx = (xright - xleft)/(float)(width - 1);
    float dy = (yright - yleft)/(float)(height - 1);

    int x = (int)current_idx % width;
    int y = (int)current_idx / width - (int)(iteration_idx*width);

    cuComplex c;
    c.x = {xleft + ((float)x)*dx};
    c.y = {yright - (float)y*dy};

    if (current_idx < (height * width * count)) {
        gpu_ptr[current_idx] = {pfc::byte_t(iterate(c)),0,0};
    }
    //printf("%f ,%f, %f, %f\n", xright, yright, xleft, yleft);

}


cudaError_t call_iteration_kernel(pfc::pixel_t * gpu_ptr, std::complex<float> & left, std::complex<float>  & right, const std::complex<float>  & zPoint, int height, int width, float factor, const size_t count){

    auto const size{ static_cast <int> (height*width) * count};

    auto const  tib =
            1024;

    auto xleft = left.real();
    auto yleft = left.imag();
    auto xright = right.real();
    auto yright = right.imag();
/*
    xright -= (xright - zPoint.real()) * (1-factor);
    yright -= (yright - zPoint.imag()) * (1-factor);
    xleft -= (xleft - zPoint.real()) * (1-factor);
    yleft -= (yleft - zPoint.imag()) * (1-factor);
*/
    iterate_GPU <<<((size+tib-1)/tib),tib >>> (gpu_ptr,  xright, xleft, yright, yleft, zPoint.real(), zPoint.imag(),height, width,count, factor);

    for(size_t i = 0; i < count; i++){
        xright -= (xright - zPoint.real()) * (1-factor);
        yright -= (yright - zPoint.imag()) * (1-factor);
        xleft -= (xleft - zPoint.real()) * (1-factor);
        yleft -= (yleft - zPoint.imag()) * (1-factor);
    }

    left = {xleft, yleft};
    right = {xright, yright};


    std::cout << "LEFT:" <<left << std::endl;
    std::cout << "RIGHT:" <<right << std::endl;

    cudaDeviceSynchronize();
    return cudaGetLastError();
}