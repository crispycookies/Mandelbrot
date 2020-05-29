#include "bitmap/pfc_bitmap_3.h"


#include <iostream>
#include <fstream>
#include <complex>

using namespace std;

float width = 5000;
float height = 5000;

int g_colors = 128;
int g_infinity = {1225000000};

template<typename complex_t>
inline complex_t square(complex_t z){
    return z*z;
}

template<typename complex_t>
inline int iterate(const complex_t & c) noexcept {
    auto i {0};
    complex_t z = 0;
    do{
        z= square<complex_t>(z);
        z+=c;
    }while((i++ < g_colors) && (norm(z)<g_infinity));
    return i;
}

int main ()  {
    pfc::bitmap bmp {(int)width, (int)height};

    for (int y {0}; y < bmp.height(); ++y) {
        for (int x {0}; x < bmp.width(); ++x) {
            complex<double> c{ (float)x/ bmp.width()-1.5, (float)y/bmp.height()-0.5};
            bmp.at(x,y) = {
                    pfc::byte_t (iterate(c)), 0, 0
            };

        }
    }
    bmp.to_file("test.bmp");
}