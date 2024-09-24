#include "bmp.h"
#include "color_function_real.h"
#include <cmath>
#include <complex>

class Julia : public ColorFunctionReal {
    RGBf dark,middle,bright;
    std::complex<float> domain;
    unsigned int range;
public:
    Julia(const RGBf& d, const RGBf& m, const RGBf& b, 
	    std::complex<float> c, unsigned int r) :
           dark(d),middle(m),bright(b),domain(c),range(r) {}       
    RGBf color_at(float x, float y) const override {
        std::complex<float> z0(x,y);
        std::complex<float> result;
        unsigned int exit = range;
        for (unsigned int i = 0; i<range; ++i) {
            z0 = z0*z0 + domain;
            if (std::abs(z0) >= 4) { exit=i; break; }        
        }
        if (exit < (range/2)) 
			return dark*(float(range/2 - exit)/float(range/2)) 
		        + middle*(float(exit)/float(range/2));
        else 
			return middle*(float(range - exit)/float(range/2)) 
		        + bright*(float(exit - range/2)/float(range/2));
    }
};

int main() {
  Image image(1920,1080);
  fill_parallel(image,
    Julia(RGBf(0,0,0),RGBf(255,0,0),RGBf(255,255,0),std::complex<float>(0,0.8),50),
    8);
  image.save("julia.bmp");
  fill_parallel(image,
    Julia(RGBf(0,0,92),RGBf(0,0,255),RGBf(255,0,255),std::complex<float>(0.13,-0.73),25),
    8);
  image.save("julia02.bmp");
}