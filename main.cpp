#include "bitmap/pfc_bitmap_3.h"


#include <iostream>
#include <complex>
#include <thread>
#include <memory>

#include "misc/pfc_timing.h"
#include "misc/pfc_threading.h"

#include "kernel.cuh"

using namespace std;

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
std::pair<std::vector<std::shared_ptr<pfc::bitmap>>, int> CalculateOnCPU(std::size_t count, complex<float> & left, complex<float> & right, const complex<float> & zPoint, const float factor, std::size_t height, std::size_t width, std::size_t additional_threads = 0){
    std::vector<std::shared_ptr<pfc::bitmap>> retval;

    //preallocating
    std::cout << "\033[22;31mPre-Alloc Buffer for Pictures" << std::endl;
    auto pre_alloc = pfc::timed_run([&]() {
        for(int i = 0; i < count; i++){
            retval.emplace_back(std::make_shared<pfc::bitmap>(width, height));
        }
    });
    std::cout << "Allocation took " << std::chrono::duration_cast<std::chrono::milliseconds>(pre_alloc).count() << "ms" << std::endl;
    std::cout << std::endl;

    //calculating
    std::cout << "Calculating Picture(s)["+std::to_string(count)+"]" << std::endl;

    auto xleft = left.real();
    auto yleft = left.imag();
    auto xright = right.real();
    auto yright = right.imag();

    int c = 0;

    auto calculation = pfc::timed_run([&]() {
        for(auto bmp : retval){
            //std::cout << c++ << std::endl;
            xright -= (xright - zPoint.real()) * (1-factor);
            yright -= (yright - zPoint.imag()) * (1-factor);
            xleft -= (xleft - zPoint.real()) * (1-factor);
            yleft -= (yleft - zPoint.imag()) * (1-factor);
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

                        float dx = (xright - xleft)/(float)(bmp->width() - 1);
                        float dy = (yright - yleft)/(float)(bmp->height() - 1);

                        c[0] = {xleft + ((float)x+0)*dx,yright - (float)y*dy};
                        c[1] = {xleft + ((float)x+1)*dx,yright - (float)y*dy};
                        c[2] = {xleft + ((float)x+2)*dx,yright - (float)y*dy};
                        c[3] = {xleft + ((float)x+3)*dx,yright - (float)y*dy};

                        c[4] = {xleft + ((float)x+4)*dx,yright - (float)y*dy};
                        c[5] = {xleft + ((float)x+5)*dx,yright - (float)y*dy};
                        c[6] = {xleft + ((float)x+6)*dx,yright - (float)y*dy};
                        c[7] = {xleft + ((float)x+7)*dx,yright - (float)y*dy};

                        c[8] = {xleft + ((float)x+8)*dx,yright - (float)y*dy};
                        c[9] = {xleft + ((float)x+9)*dx,yright - (float)y*dy};
                        c[10] = {xleft + ((float)x+10)*dx,yright - (float)y*dy};
                        c[11] = {xleft + ((float)x+11)*dx,yright - (float)y*dy};

                        c[12] = {xleft + ((float)x+12)*dx,yright - (float)y*dy};
                        c[13] = {xleft + ((float)x+13)*dx,yright - (float)y*dy};
                        c[14] = {xleft + ((float)x+14)*dx,yright - (float)y*dy};
                        c[15] = {xleft + ((float)x+15)*dx,yright - (float)y*dy};

                        auto r0 = pfc::byte_t(iterate(i[0],z[0],c[0]));
                        auto r1 = pfc::byte_t(iterate(i[1],z[1],c[1]));
                        auto r2 = pfc::byte_t(iterate(i[2],z[2],c[2]));
                        auto r3 = pfc::byte_t(iterate(i[3],z[3],c[3]));

                        auto r4 = pfc::byte_t(iterate(i[4],z[4],c[4]));
                        auto r5 = pfc::byte_t(iterate(i[5],z[5],c[5]));
                        auto r6 = pfc::byte_t(iterate(i[6],z[6],c[6]));
                        auto r7 = pfc::byte_t(iterate(i[7],z[7],c[7]));

                        auto r8 = pfc::byte_t(iterate(i[8],z[8],c[8]));
                        auto r9 = pfc::byte_t(iterate(i[9],z[9],c[9]));
                        auto r10 = pfc::byte_t(iterate(i[10],z[10],c[10]));
                        auto r11 = pfc::byte_t(iterate(i[11],z[11],c[11]));

                        auto r12 = pfc::byte_t(iterate(i[12],z[12],c[12]));
                        auto r13 = pfc::byte_t(iterate(i[13],z[13],c[13]));
                        auto r14 = pfc::byte_t(iterate(i[14],z[14],c[14]));
                        auto r15 = pfc::byte_t(iterate(i[15],z[15],c[15]));

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
            left -= (left-zPoint)*(1-factor);
            right -= (right-zPoint)*(1*factor);
            */
        }
    });
    //copy back
    left = {xleft, yleft};
    right = {xright, yright};


    std::cout << "CPU Calculation took " << std::chrono::duration_cast<std::chrono::milliseconds>(calculation).count() << "ms\n" << std::endl;

    return {retval, std::chrono::duration_cast<std::chrono::milliseconds>(calculation).count()};
}


void store(const std::string prefix, std::vector<std::shared_ptr<pfc::bitmap>> slides, int & cnt){
    std::cout << "\033[01;37mStoring Files... Pls Wait" << std::endl;
    for(const auto & c : slides){
        if(c == nullptr){
            throw std::string("Failure; Empty Picture");
        }
        c->to_file(prefix + std::to_string(cnt) + ".bmp");
        cnt++;
    }
    std::cout << "Finished" << std::endl << std::endl;
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
void allocate_memory(std::vector<std::shared_ptr<pfc::bitmap>> & cpu_destination, pfc::pixel_t *& gpu, int width, int height, const std::size_t count) {
    for(int i = 0; i < count; i++){
        cpu_destination.emplace_back(std::make_shared<pfc::bitmap>(width, height));
    }
    //GPU Malloc
    check(cudaMalloc(&gpu, cpu_destination.at(0)->size()*sizeof(pfc::pixel_t)*count));
}
void free_memory(pfc::pixel_t *& gpu) {
    check(cudaFree(gpu)); gpu = nullptr;
}

void copy_to_cpu_vector(std::vector<std::shared_ptr<pfc::bitmap>> & cpu_destination, pfc::pixel_t * gpu){
    int size = cpu_destination.at(0)->size();
    for(int i = 0; i < cpu_destination.size(); i++){
        auto & span_dest {cpu_destination.at(i)->pixel_span ()};
        pfc::pixel_t * p_buffer_dest {std::data (span_dest)};
        pfc::pixel_t * offset = (pfc::pixel_t*)(size * sizeof(pfc::pixel_t) * i + (long)gpu);

        copy_to_cpu(p_buffer_dest, offset,cpu_destination.at(i)->size());
    }
}

int checked_main(complex<float> & left, complex<float> & right, const complex<float> & zPoint, int height, int width, float factor, int count, const std::string & prefix, const bool save = true, const std::size_t parallel_count = 25){
    std::vector<std::shared_ptr<pfc::bitmap>> cpu_destination;
    pfc::pixel_t * gpu = nullptr;

    check(cudaSetDevice(0));
    cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop, 0));

    std::cout << "Device:\t" <<prop.name << '\n';
    std::cout << "Compute Capability:\t" << prop.major << '.' << prop.minor << '\n';
    std::cout << "-----------------------------------" << std::endl;

    cudaDeviceSynchronize();
    allocate_memory(cpu_destination,gpu,width,height,parallel_count);


    /*
    auto & span_dest {cpu_destination->pixel_span ()};
    pfc::pixel_t * p_buffer_dest {std::data (span_dest)};
    */

    int time = 0;

    int counter = 0;

    for(int i = 0; i < count/parallel_count;i++){
        auto timed_run = pfc::timed_run([&]() {
            check(call_iteration_kernel(gpu,left,right,zPoint, height, width, factor, parallel_count));
            copy_to_cpu_vector(cpu_destination,gpu);
            //copy_to_cpu(p_buffer_dest, gpu,cpu_destination->size());
        });
        time += std::chrono::duration_cast<std::chrono::milliseconds>(timed_run).count();
        if(save) {
            for(auto & c : cpu_destination){
              c->to_file(prefix + std::to_string(counter++) + ".bmp");
            }
            //cpu_destination->to_file(prefix + std::to_string(i) + ".bmp");
        }
    }

    free_memory(gpu);
    check(cudaDeviceReset());

    std::cout << "GPU Calculation took " << time << "ms\n" << std::endl;

    return time;
}

void warm_up(){
    std::cout << "\033[01;37mWarming Up CPU" << std::endl;
    pfc::warm_up_cpu();
    std::cout << "Finished" << std::endl;
    std::cout << std::endl;
}

int calc_cpu(std::size_t count, complex<float> & left, complex<float> & right, const complex<float> & zPoint, const float factor, std::size_t height, std::size_t width, std::size_t additional_threads = 0, const bool save = false){
    int store_cnt = 0;

    auto slides_0_100 = CalculateOnCPU(count/2,left, right, zPoint,factor,height,width,additional_threads);
    if(slides_0_100.first.empty()){
        throw std::string("First Vector in CPU Calculation is Empty");
    }
    if(save){
        std::cout << "Storing First Chunk of Data" << std::endl;
        store("Mandel_CPU_", slides_0_100.first, store_cnt);
        std::cout << "Done" << std::endl;
    }
    slides_0_100.first.clear();

    auto slides_100_200 = CalculateOnCPU(count/2,left, right, zPoint,factor,height,width,additional_threads);
    if(slides_100_200.first.empty()){
        throw std::string("First Vector in CPU Calculation is Empty");
    }
    if(save){
        std::cout << "Storing Second Chunk of Data" << std::endl;
        store("Mandel_CPU_", slides_100_200.first, store_cnt);
        std::cout << "Done" << std::endl;
    }
    std::cout << "LEFT:" <<left << std::endl;
    std::cout << "RIGHT:" <<right << std::endl;
    return slides_100_200.second + slides_0_100.second;
}

int main ()  {

    try{
        //General
        int count = 200;
        bool save = true;

        int height = 4608;
        int width = 8192;

        complex<float> left = {-2.74529004, -1.01192498};
        complex<float> right = {1.25470996 , 1.23807502};
        complex<float> zPoint = {-0.745289981 , 0.113075003};

        warm_up();

        //GPU
        std::cout << "\033[22;32mGPU Calculation" << std::endl;
        int time_gpu = checked_main(left, right, zPoint, height,width,0.95, count, "Mandel_GPU_",save,25);
        std::cout << "Finished" << std::endl;

        left = {-2.74529004, -1.01192498};
        right = {1.25470996 , 1.23807502};
        zPoint = {-0.745289981 , 0.113075003};

        warm_up();
        std::cout << "\033[22;31mCPU Calculation" << std::endl;
        auto time_cpu = 11;//calc_cpu(count,left, right, zPoint,0.95,height,width,1000, save);

        auto size = height*width * sizeof(pfc::BGR_4_t) * count/1000000;


        if(time_cpu == 0 || time_gpu == 0){
            throw std::string("Invalid Time measured");
        }

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
