#include "bitmap/pfc_bitmap_3.h"


#include <iostream>
#include <complex>
#include <thread>
#include <memory>

#include "misc/pfc_timing.h"
#include "misc/pfc_threading.h"

using namespace std;

#define var auto

int g_colors = 255;
int g_infinity = {1225000000};

template<typename complex_t>
inline complex_t square(complex_t & z){
    return z*z;
}

template<typename complex_t>
inline int iterate(const complex_t & c) noexcept {
    auto i {0};
    complex<double> z = 0;
    do{
        z = square(z);
        z += c;
    }while((i++ < g_colors) && (norm(z)<g_infinity));
    return i;
}
// 28280ms
std::vector<std::shared_ptr<pfc::bitmap>> CalculateOnCPU(std::size_t count, float, float, float, float, std::size_t height, std::size_t width, std::size_t additional_threads = 0){
    std::vector<std::shared_ptr<pfc::bitmap>> retval;

    std::cout << "Warming Up CPU" << std::endl;
    pfc::warm_up_cpu();
    std::cout << "Finished" << std::endl;
    std::cout << std::endl;

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



    auto calculation = pfc::timed_run([&]() {
        for(auto bmp : retval){
            pfc::parallel_range<size_t>(additional_threads, height, [&](size_t t, size_t begin, size_t end) {
                complex<double> c0 = 0;
                complex<double> c1 = 0;
                complex<double> c2 = 0;
                complex<double> c3 = 0;

                complex<double> c4 = 0;
                complex<double> c5 = 0;
                complex<double> c6 = 0;
                complex<double> c7 = 0;

                complex<double> c8 = 0;
                complex<double> c9 = 0;
                complex<double> c10 = 0;
                complex<double> c11 = 0;

                complex<double> c12 = 0;
                complex<double> c13 = 0;
                complex<double> c14 = 0;
                complex<double> c15 = 0;


                complex<double> c[16] = {0};

                for (int y = begin; y < end; y++) {
                    //21 s
                    //18 s
                    // Loop Unrolling
                    for (int x{0}; x < bmp->width(); x+=16) {
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



                        var r0 = pfc::byte_t(iterate(c[0]));
                        var r1 = pfc::byte_t(iterate(c[1]));
                        var r2 = pfc::byte_t(iterate(c[2]));
                        var r3 = pfc::byte_t(iterate(c[3]));
                        var r4 = pfc::byte_t(iterate(c[4]));
                        var r5 = pfc::byte_t(iterate(c[5]));
                        var r6 = pfc::byte_t(iterate(c[6]));
                        var r7 = pfc::byte_t(iterate(c[7]));
                        var r8 = pfc::byte_t(iterate(c[8]));
                        var r9 = pfc::byte_t(iterate(c[9]));
                        var r10 = pfc::byte_t(iterate(c[10]));
                        var r11 = pfc::byte_t(iterate(c[11]));
                        var r12 = pfc::byte_t(iterate(c[12]));
                        var r13 = pfc::byte_t(iterate(c[13]));
                        var r14 = pfc::byte_t(iterate(c[14]));
                        var r15 = pfc::byte_t(iterate(c[15]));



                        bmp->at(x, y) = {
                                r0, 0, 0
                        };

                        bmp->at(x + 1, y) = {
                                r1, 0, 0
                        };

                        bmp->at(x + 2, y) = {
                                r2, 0, 0
                        };

                        bmp->at(x + 3, y) = {
                                r3, 0, 0
                        };

                        bmp->at(x + 4, y) = {
                                r4, 0, 0
                        };

                        bmp->at(x + 5, y) = {
                                r5, 0, 0
                        };

                        bmp->at(x + 6, y) = {
                                r6, 0, 0
                        };

                        bmp->at(x + 7, y) = {
                                r7, 0, 0
                        };
                        bmp->at(x + 8, y) = {
                                r8, 0, 0
                        };

                        bmp->at(x + 9, y) = {
                                r9, 0, 0
                        };
                        bmp->at(x + 10, y) = {
                                r10, 0, 0
                        };
                        bmp->at(x + 11, y) = {
                                r11, 0, 0
                        };
                        bmp->at(x + 12, y) = {
                                r12, 0, 0
                        };
                        bmp->at(x + 13, y) = {
                                r13, 0, 0
                        };
                        bmp->at(x + 14, y) = {
                                r14, 0, 0
                        };
                        bmp->at(x + 15, y) = {
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


    std::vector<std::shared_ptr<pfc::bitmap>> slides = CalculateOnCPU(10,0,0,0,0,4608,8192,std::thread::hardware_concurrency()*25);



    int cnt = 0;
    for(const auto & c : slides){
        if(c == nullptr){
            throw std::string("Failure; Empty Picture");
        }
        c->to_file("T_mandelbrot_" + std::to_string(cnt) + ".bmp");
        cnt++;
    }
}


