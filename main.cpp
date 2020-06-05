#include "bitmap/pfc_bitmap_3.h"


#include <iostream>
#include <complex>
#include <thread>
#include <memory>

#include "misc/pfc_timing.h"
#include "misc/pfc_threading.h"

using namespace std;

#define var auto

int g_colors = 128;
int g_infinity = {1225000000};

template<typename complex_t>
inline complex_t square(complex_t & z){
    return z*z;
}

template<typename complex_t>
inline int iterate(int & i, complex_t & z, const complex_t & c) noexcept {
    //auto i {0};
    //complex<double> z = 0;
    do{
        z *= z;
        z += c;
    }while((i++ < g_colors) && (norm(z)<g_infinity));
    return i;
}
// 28280ms
std::vector<std::shared_ptr<pfc::bitmap>> CalculateOnCPU(std::size_t count, float, float, float, float, std::size_t height, std::size_t width, std::size_t additional_threads = 0){
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

    std::cout << "Warming Up CPU" << std::endl;
    pfc::warm_up_cpu();
    std::cout << "Finished" << std::endl;
    std::cout << std::endl;

    //calculating
    std::cout << "Calculating Picture(s)["+std::to_string(count)+"]" << std::endl;



    auto calculation = pfc::timed_run([&]() {
        for(auto bmp : retval){
            pfc::parallel_range<size_t>(additional_threads, height, [&](size_t t, size_t begin, size_t end) {

                complex<double> c[16] = {0};
                for (int y = begin; y < end; y++) {
                    //21 s
                    //18 s
                    // Loop Unrolling
                    auto & span {bmp->pixel_span ()};
                    auto * const p_buffer {std::data (span)};
                    for (int x{0}; x < bmp->width(); x+=16) {
                        complex<double> z[16] =  {0};
                        int i[16] = {0};

                        c[0] = {(float) x / bmp->width() - 1.5, (float) y / bmp->height() - 0.5};
                        c[1] = {(float) (x + 1) / bmp->width() - 1.5, (float) y / bmp->height() - 0.5};

                        c[2] = {(float) (x + 2) / bmp->width() - 1.5, (float) y / bmp->height() - 0.5};
                        c[3] = {(float) (x + 3) / bmp->width() - 1.5, (float) y / bmp->height() - 0.5};
                        c[4] = {(float) (x + 4) / bmp->width() - 1.5, (float) y / bmp->height() - 0.5};
                        c[5] = {(float) (x + 5) / bmp->width() - 1.5, (float) y / bmp->height() - 0.5};
                        c[6] = {(float) (x + 6) / bmp->width() - 1.5, (float) y / bmp->height() - 0.5};
                        c[7] = {(float) (x + 7) / bmp->width() - 1.5, (float) y / bmp->height() - 0.5};
                        c[8] = {(float) (x + 8) / bmp->width() - 1.5, (float) y / bmp->height() - 0.5};
                        c[9] = {(float) (x + 9) / bmp->width() - 1.5, (float) y / bmp->height() - 0.5};

                        c[10] = {(float) (x + 10) / bmp->width() - 1.5, (float) y / bmp->height() - 0.5};
                        c[11] = {(float) (x + 11) / bmp->width() - 1.5, (float) y / bmp->height() - 0.5};
                        c[12] = {(float) (x + 12) / bmp->width() - 1.5, (float) y / bmp->height() - 0.5};
                        c[13] = {(float) (x + 13) / bmp->width() - 1.5, (float) y / bmp->height() - 0.5};
                        c[14] = {(float) (x + 14) / bmp->width() - 1.5, (float) y / bmp->height() - 0.5};
                        c[15] = {(float) (x + 15) / bmp->width() - 1.5, (float) y / bmp->height() - 0.5};

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
            });
        }
    });
    std::cout << "CPU Calculation took " << std::chrono::duration_cast<std::chrono::milliseconds>(calculation).count() << "ms" << std::endl;

    return retval;
}


int main ()  {


    std::vector<std::shared_ptr<pfc::bitmap>> slides = CalculateOnCPU(10,0,0,0,0,4608,8192,800);
    //std::vector<std::shared_ptr<pfc::bitmap>> slides2 =  CalculateOnCPU(100,0,0,0,0,4608,8192,400);



    /*int cnt = 0;
    for(const auto & c : slides){
        if(c == nullptr){
            throw std::string("Failure; Empty Picture");
        }
        c->to_file("T_mandelbrot_" + std::to_string(cnt) + ".bmp");
        cnt++;
    }*/
}


