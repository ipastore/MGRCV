#pragma once
#include "bmp.h"
#include <random>
#include <thread>

class RGBf {
    float _r, _g, _b;
public:
    RGBf(float r, float g, float b) : _r(r),_g(g),_b(b) {}
    RGBf(float v) : _r(v), _g(v), _b(v) {}
    RGBf(const RGB& rgb) : _r(rgb.r()), _g(rgb.g()), _b(rgb.b()) {}
    operator RGB() const { return RGB(std::max(0.0f,std::min(255.0f,_r)),
         std::max(0.0f,std::min(255.0f,_g)),
         std::max(0.0f,std::min(255.0f,_b))); }

    RGBf operator+(const RGBf& that) const {
        return RGBf(this->_r+that._r, this->_g+that._g, this->_b+that._b);
    }

    RGBf operator*(float f) const {
        return RGBf(this->_r*f, this->_g*f, this->_b*f);
    }

    RGBf operator/(float f) const {
        return RGBf(this->_r/f, this->_g/f, this->_b/f);
    }
};


class ColorFunctionReal {
public:
    virtual RGBf color_at(float x, float y) const = 0;
};

void fill(Image& image, const ColorFunctionReal& f, int spp) {
    std::random_device rd;
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);    
    float dx = 2.0/float(image.width()); float dy = 2.0/float(image.height());
    for (int j = 0; j<image.height(); ++j)
        for (int i = 0;i<image.width(); ++i) {
            RGBf pixel(0,0,0);
            for (int s = 0; s<spp; ++s)
                pixel = pixel + f.color_at(-1.0f+(float(i)+dis(gen))*dx,
                                           -1.0f+(float(j)+dis(gen))*dy); 
            image(i,j) = pixel/spp;
        }
}

void fill_task(Image& image, const ColorFunctionReal& f, int spp, int line_begin, int line_end) {
    std::random_device rd;
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);    
    float dx = 2.0/float(image.width()); float dy = 2.0/float(image.height());
    for (int j = line_begin; j<line_end; ++j)
        for (int i = 0;i<image.width(); ++i) {
            RGBf pixel(0,0,0);
            for (int s = 0; s<spp; ++s)
                pixel = pixel + f.color_at(-1.0f+(float(i)+dis(gen))*dx,
                                           -1.0f+(float(j)+dis(gen))*dy); 
            image(i,j) = pixel/spp;
        }
}

void fill_parallel(Image& image, const ColorFunctionReal& f, int spp) {
    std::size_t n = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    for (std::size_t i = 0; i < n; ++i) {
        threads.push_back(std::thread([&image,&f,spp,i,n] () {
            fill_task(image, f, spp, (image.height()*i)/n, (image.height()*(i+1))/n);
        }));
    }

    for (auto& th : threads) th.join(); 
}


