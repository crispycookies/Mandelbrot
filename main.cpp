#include "bitmap/pfc_bitmap_3.h"


#include <iostream>
#include <complex>
#include <thread>
#include <memory>

#include "misc/pfc_timing.h"
#include "misc/pfc_threading.h"

using namespace std;

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
            retval.push_back(std::make_shared<pfc::bitmap>(width, height));
        }
    });
    std::cout << "Allocation took " << std::chrono::duration_cast<std::chrono::milliseconds>(pre_alloc).count() << "ms" << std::endl;
    std::cout << std::endl;


    //calculating
    std::cout << "Calculating Picture(s)["+std::to_string(count)+"]" << std::endl;



    auto calculation = pfc::timed_run([&]() {
        for(auto bmp : retval){
            pfc::parallel_range<size_t>(std::thread::hardware_concurrency()+additional_threads, height, [&](size_t t, size_t begin, size_t end) {
                complex<double> c0 = 0;
                complex<double> c1 = 0;
                complex<double> c2 = 0;
                complex<double> c3 = 0;
                complex<double> c4 = 0;
                complex<double> c5 = 0;
                complex<double> c6 = 0;
                complex<double> c7 = 0;


                for (int y = begin; y < end; y++) {
                    //21 s
                    //18 s
                    // Loop Unrolling
                    for (int x{0}; x < bmp->width(); x+=8) {
                        c0 = {(float)x/ bmp->width()-1.5, (float)y/bmp->height()-0.5};
                        c1 = {(float)(x+1)/ bmp->width()-1.5, (float)y/bmp->height()-0.5};
                        c2 = {(float)(x+2)/ bmp->width()-1.5, (float)y/bmp->height()-0.5};
                        c3 = {(float)(x+3)/ bmp->width()-1.5, (float)y/bmp->height()-0.5};
                        c4 = {(float)(x+4)/ bmp->width()-1.5, (float)y/bmp->height()-0.5};
                        c5 = {(float)(x+5)/ bmp->width()-1.5, (float)y/bmp->height()-0.5};
                        c6 = {(float)(x+6)/ bmp->width()-1.5, (float)y/bmp->height()-0.5};
                        c7 = {(float)(x+7)/ bmp->width()-1.5, (float)y/bmp->height()-0.5};


                        bmp->at(x, y) = {
                                pfc::byte_t(iterate( c0)), 0, 0
                        };

                        bmp->at(x+1, y) = {
                                pfc::byte_t(iterate( c1)), 0, 0
                        };

                        bmp->at(x+2, y) = {
                                pfc::byte_t(iterate( c2)), 0, 0
                        };

                        bmp->at(x+3, y) = {
                                pfc::byte_t(iterate( c3)), 0, 0
                        };

                        bmp->at(x+4, y) = {
                                pfc::byte_t(iterate( c4)), 0, 0
                        };

                        bmp->at(x+5, y) = {
                                pfc::byte_t(iterate( c5)), 0, 0
                        };

                        bmp->at(x+6, y) = {
                                pfc::byte_t(iterate( c6)), 0, 0
                        };

                        bmp->at(x+7, y) = {
                                pfc::byte_t(iterate( c7)), 0, 0
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


    std::vector<std::shared_ptr<pfc::bitmap>> slides = CalculateOnCPU(100,0,0,0,0,4000,4000,std::thread::hardware_concurrency());



    int cnt = 0;
    for(const auto & c : slides){
        if(c == nullptr){
            throw std::string("Failure; Empty Picture");
        }
        c->to_file("T_mandelbrot_" + std::to_string(cnt) + ".bmp");
        cnt++;
    }
}


