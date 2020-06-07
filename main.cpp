#include "bitmap/pfc_bitmap_3.h"


#include <iostream>
#include <complex>
#include <thread>
#include <memory>

#include "misc/pfc_timing.h"
#include "misc/pfc_threading.h"

#include "kernel.cuh"

using namespace std;

#define var auto


inline int iterate(int & i, complex<float> & z, const  complex<float> & c) noexcept {
    //auto i {0};
    //complex<double> z = 0;
    do{
        z *= z;
        z += c;
    }while((i++ < g_colors) && (norm(z)<g_infinity));
    return i;
}
// 28280ms
std::pair<std::vector<std::shared_ptr<pfc::bitmap>>, int> CalculateOnCPU(std::size_t count, complex<float> & left, complex<float> & right, const complex<float> & zoomPoint, const float factor, std::size_t height, std::size_t width, std::size_t additional_threads = 0){
    std::vector<std::shared_ptr<pfc::bitmap>> retval;

    //preallocating
    std::cout << "Pre-Alloc Buffer for Pictures" << std::endl;
    auto pre_alloc = pfc::timed_run([&]() {
        for(int i = 0; i < count; i++){
            retval.emplace_back(std::make_shared<pfc::bitmap>(width, height));
        }
    });
    std::cout << "Allocation took " << std::chrono::duration_cast<std::chrono::milliseconds>(pre_alloc).count() << "ms" << std::endl;
    std::cout << std::endl;

    //calculating
    std::cout << "Calculating Picture(s)["+std::to_string(count)+"]" << std::endl;

    var x_start = left.real();
    var y_start = left.imag();
    var x_fin = right.real();
    var y_fin = right.imag();

    int c = 0;

    auto calculation = pfc::timed_run([&]() {
        for(auto bmp : retval){
            //std::cout << c++ << std::endl;
            x_fin -= (x_fin - zoomPoint.real()) * (1-factor);
            y_fin -= (y_fin - zoomPoint.imag()) * (1-factor);
            x_start -= (x_start - zoomPoint.real()) * (1-factor);
            y_start -= (y_start - zoomPoint.imag()) * (1-factor);
            pfc::parallel_range<size_t>(additional_threads, height, [&](size_t t, size_t begin, size_t end) {


                complex<float> c[16] = {0};
                for (int y = begin; y < end; y++) {
                    //21 s
                    //18 s
                    // Loop Unrolling
                    auto & span {bmp->pixel_span ()};
                    auto * const p_buffer {std::data (span)};
                    for (int x{0}; x < bmp->width(); x+=16) {
                        complex<float> z[16] =  {0};
                        int i[16] = {0};

                        float dx = (x_fin - x_start)/(float)(bmp->width() - 1);
                        float dy = (y_fin - y_start)/(float)(bmp->height() - 1);

                        c[0] = {x_start + ((float)x+0)*dx,y_fin - (float)y*dy};
                        c[1] = {x_start + ((float)x+1)*dx,y_fin - (float)y*dy};
                        c[2] = {x_start + ((float)x+2)*dx,y_fin - (float)y*dy};
                        c[3] = {x_start + ((float)x+3)*dx,y_fin - (float)y*dy};

                        c[4] = {x_start + ((float)x+4)*dx,y_fin - (float)y*dy};
                        c[5] = {x_start + ((float)x+5)*dx,y_fin - (float)y*dy};
                        c[6] = {x_start + ((float)x+6)*dx,y_fin - (float)y*dy};
                        c[7] = {x_start + ((float)x+7)*dx,y_fin - (float)y*dy};

                        c[8] = {x_start + ((float)x+8)*dx,y_fin - (float)y*dy};
                        c[9] = {x_start + ((float)x+9)*dx,y_fin - (float)y*dy};
                        c[10] = {x_start + ((float)x+10)*dx,y_fin - (float)y*dy};
                        c[11] = {x_start + ((float)x+11)*dx,y_fin - (float)y*dy};

                        c[12] = {x_start + ((float)x+12)*dx,y_fin - (float)y*dy};
                        c[13] = {x_start + ((float)x+13)*dx,y_fin - (float)y*dy};
                        c[14] = {x_start + ((float)x+14)*dx,y_fin - (float)y*dy};
                        c[15] = {x_start + ((float)x+15)*dx,y_fin - (float)y*dy};

                        var r0 = pfc::byte_t(iterate(i[0],z[0],c[0]));
                        var r1 = pfc::byte_t(iterate(i[1],z[1],c[1]));
                        var r2 = pfc::byte_t(iterate(i[2],z[2],c[2]));
                        var r3 = pfc::byte_t(iterate(i[3],z[3],c[3]));

                        var r4 = pfc::byte_t(iterate(i[4],z[4],c[4]));
                        var r5 = pfc::byte_t(iterate(i[5],z[5],c[5]));
                        var r6 = pfc::byte_t(iterate(i[6],z[6],c[6]));
                        var r7 = pfc::byte_t(iterate(i[7],z[7],c[7]));

                        var r8 = pfc::byte_t(iterate(i[8],z[8],c[8]));
                        var r9 = pfc::byte_t(iterate(i[9],z[9],c[9]));
                        var r10 = pfc::byte_t(iterate(i[10],z[10],c[10]));
                        var r11 = pfc::byte_t(iterate(i[11],z[11],c[11]));

                        var r12 = pfc::byte_t(iterate(i[12],z[12],c[12]));
                        var r13 = pfc::byte_t(iterate(i[13],z[13],c[13]));
                        var r14 = pfc::byte_t(iterate(i[14],z[14],c[14]));
                        var r15 = pfc::byte_t(iterate(i[15],z[15],c[15]));

                        p_buffer[y * width + x] = {
                                r0, 0, 0
                        };
                        p_buffer[y * width + x+1] = {
                                r1, 0, 0
                        };
                        p_buffer[y * width + x+2] = {
                                r2, 0, 0
                        };
                        p_buffer[y * width + x+3] = {
                                r3, 0, 0
                        };

                        p_buffer[y * width + x+4] = {
                                r4, 0, 0
                        };
                        p_buffer[y * width + x+5] = {
                                r5, 0, 0
                        };
                        p_buffer[y * width + x+6] = {
                                r6, 0, 0
                        };
                        p_buffer[y * width + x+7] = {
                                r7, 0, 0
                        };

                        p_buffer[y * width + x+8] = {
                                r8, 0, 0
                        };
                        p_buffer[y * width + x+9] = {
                                r9, 0, 0
                        };
                        p_buffer[y * width + x+10] = {
                                r10, 0, 0
                        };
                        p_buffer[y * width + x+11] = {
                                r11, 0, 0
                        };

                        p_buffer[y * width + x+12] = {
                                r12, 0, 0
                        };
                        p_buffer[y * width + x+13] = {
                                r13, 0, 0
                        };
                        p_buffer[y * width + x+14] = {
                                r14, 0, 0
                        };
                        p_buffer[y * width + x+15] = {
                                r15, 0, 0
                        };
                    }
                }
            });/*
            left -= (left-zoomPoint)*(1-factor);
            right -= (right-zoomPoint)*(1*factor);
            */
        }
    });
    //copy back
    left = {x_start, y_start};
    right = {x_fin, y_fin};


    std::cout << "CPU Calculation took " << std::chrono::duration_cast<std::chrono::milliseconds>(calculation).count() << "ms\n" << std::endl;

    return {retval, std::chrono::duration_cast<std::chrono::milliseconds>(calculation).count()};
}


void store(const std::string prefix, std::vector<std::shared_ptr<pfc::bitmap>> slides, int & cnt){
    std::cout << "Storing Files... Pls Wait" << std::endl;
    for(const auto & c : slides){
        if(c == nullptr){
            throw std::string("Failure; Empty Picture");
        }
        c->to_file(prefix + std::to_string(cnt) + ".bmp");
        cnt++;
    }
    std::cout << "Finished" << std::endl;
}

void check(cudaError_t const e) {
    if (e != cudaSuccess) {
        throw std::runtime_error{ cudaGetErrorName(e) };
    }
}

void copy_to_gpu(pfc::pixel_t *& cpu, pfc::pixel_t *& gpu, int size) {
    check(cudaMemcpy(gpu, cpu,size * sizeof(pfc::pixel_t), cudaMemcpyHostToDevice));
}

void copy_to_cpu(pfc::pixel_t *& cpu, pfc::pixel_t *& gpu, int size) {
    check(cudaMemcpy(cpu, gpu, size * sizeof(pfc::pixel_t), cudaMemcpyDeviceToHost));
}
void allocate_memory(std::shared_ptr<pfc::bitmap> & cpu_source, std::shared_ptr<pfc::bitmap> & cpu_destination, pfc::pixel_t *& gpu, int width, int height) {
    cpu_source = std::make_shared<pfc::bitmap>(width, height);
    cpu_destination = std::make_shared<pfc::bitmap>(width, height);

    //GPU Malloc
    check(cudaMalloc(&gpu, cpu_source->size()*sizeof(pfc::pixel_t)));
}
void free_memory(pfc::pixel_t *& gpu) {
    check(cudaFree(gpu)); gpu = nullptr;
}


int checked_main(complex<float> & left, complex<float> & right, const complex<float> & zoomPoint, int height, int width, float factor, int count, const std::string & prefix){
    std::shared_ptr<pfc::bitmap> cpu_source = nullptr;
    std::shared_ptr<pfc::bitmap> cpu_destination= nullptr;
    pfc::pixel_t * gpu = nullptr;

    check(cudaSetDevice(0));
    cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop, 0));

    std::cout << "Device:\t" <<prop.name << '\n';
    std::cout << "Compute Capability:\t" << prop.major << '.' << prop.minor << '\n';
    std::cout << "-----------------------------------" << std::endl;

    cudaDeviceSynchronize();
    allocate_memory(cpu_source,cpu_destination,gpu,width,height);

    var & span {cpu_source->pixel_span ()};
    pfc::pixel_t * p_buffer {std::data (span)};

    var & span_dest {cpu_destination->pixel_span ()};
    pfc::pixel_t * p_buffer_dest {std::data (span_dest)};

    //copy_to_gpu(p_buffer, gpu,cpu_source->size());
    int time = 0;

    for(int i = 0; i < count;i++){
        auto timed_run = pfc::timed_run([&]() {
            check(call_iteration_kernel(gpu,left,right,zoomPoint, height, width,factor));
            copy_to_cpu(p_buffer_dest, gpu,cpu_source->size());
        });
        time += std::chrono::duration_cast<std::chrono::milliseconds>(timed_run).count();
        cpu_destination->to_file(prefix+ std::to_string(i)+".bmp");
    }

    free_memory(gpu);
    check(cudaDeviceReset());

    std::cout << "GPU Calculation took " << time << "ms\n" << std::endl;

    return time;
}

int main ()  {

    try{
        //General
        int count = 200;
        int store_cnt = 0;

        complex<float> left = {-2.74529004, -1.01192498};
        complex<float> right = {1.25470996 , 1.23807502};
        complex<float> zoomPoint = {-0.745289981 , 0.113075003};

        std::cout << "Warming Up CPU" << std::endl;
        pfc::warm_up_cpu();
        std::cout << "Finished" << std::endl;
        std::cout << std::endl;

        //GPU
        std::cout << "\033[22;32mGPU Calculation" << std::endl;
        int time_gpu = checked_main(left, right, zoomPoint, 4608,8192,0.95,count, "Mandel_GPU_");
        std::cout << "Finished" << std::endl;

        left = {-2.74529004, -1.01192498};
        right = {1.25470996 , 1.23807502};
        zoomPoint = {-0.745289981 , 0.113075003};

        std::cout << "Warming Up CPU" << std::endl;
        pfc::warm_up_cpu();
        std::cout << "Finished" << std::endl;
        std::cout << std::endl;

        std::cout << "\033[22;31mCPU Calculation" << std::endl;
        auto slides_0_100 = CalculateOnCPU(count/2,left, right, zoomPoint,0.95,4608,8192,1000);
        store("Mandel_CPU_", slides_0_100.first, store_cnt);
        slides_0_100.first.clear();
        auto slides_100_200 = CalculateOnCPU(count/2,left, right, zoomPoint,0.95,4608,8192,1000);
        store("Mandel_CPU_", slides_100_200.first, store_cnt);
        std::cout << "Finished" << std::endl;

        if(slides_100_200.first.at(0) == nullptr){
            throw std::string("Cannot Calculate Statistical Data as at least one Element in Result Vector is invalid or empty");
        }

        var size = slides_100_200.first.at(0)->size() * sizeof(pfc::BGR_4_t) * count/1000000;
        var time_cpu = slides_100_200.second + slides_0_100.second;

        if(time_cpu == 0 /*|| time_gpu == 0*/){
            throw std::string("Invalid Time measured");
        }

        slides_100_200.first.clear();

        std::cout << "\033[01;37m" << std::endl;

        std::cout << "CPU:         " << "R7 3700x @ 4.3 GHz" << std::endl;
        std::cout << "Runtime:     " << time_cpu << "ms (for " << std::to_string(count) << " Bitmaps and " << std::to_string(size) << "MB of Data)" << std::endl;
        std::cout << "throughput:  " << size/((float)time_cpu/1000) << "MB/s" << std::endl;
        std::cout << std::endl;

        cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop, 0));

        std::cout << "GPU:         " << prop.name << std::endl;
        std::cout << "Runtime:     " << time_gpu << "ms (for " << std::to_string(count) << " Bitmaps and " << std::to_string(size) << "MB of Data)" << std::endl;
        std::cout << "throughput:  " << size/((float)time_gpu/1000) << "MB/s" << std::endl;
        std::cout << std::endl;

        std::cout << "Speedup(%):  " << time_cpu*100/time_gpu << std::endl;
        std::cout << std::endl;

    }
    catch (const std::string & exe) {
        std::cerr << "Failed with Message: " << exe << std::endl;
    }
}


