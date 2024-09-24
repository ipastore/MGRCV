#include "bmp.h"
#include "color_function_real.h"
#include <cmath>

class Rhombus : public ColorFunctionReal {
    RGBf bright, dark;
    float vertical_axis, horizontal_axis;
public:
    Rhombus(const RGBf& b, const RGBf& d, float v, float h) :
        bright(b), dark(d), vertical_axis(v),horizontal_axis(h) {}    
    RGBf color_at(float x, float y) const override {
        if ((std::abs(x)/horizontal_axis + std::abs(y)/vertical_axis) < 1.0f)
            return dark;
        else
            return bright;
    }
};

int main() {
    Image image(256,256);
    fill(image,Rhombus(RGBf(200,200,200),RGBf(0,0,0),0.8,0.5),2);
    image.save("rhombus-aa02.bmp");
    fill(image,Rhombus(RGBf(200,200,200),RGBf(0,0,0),0.8,0.5),4);
    image.save("rhombus-aa04.bmp");
    fill(image,Rhombus(RGBf(200,200,200),RGBf(0,0,0),0.8,0.5),8);
    image.save("rhombus-aa08.bmp");
    fill(image,Rhombus(RGBf(200,200,200),RGBf(0,0,0),0.8,0.5),16);
    image.save("rhombus-aa16.bmp");
    fill(image,Rhombus(RGBf(200,200,200),RGBf(0,0,0),0.8,0.5),32);
    image.save("rhombus-aa32.bmp");
    fill_parallel(image,Rhombus(RGBf(200,200,200),RGBf(0,0,0),0.8,0.5),64);
    image.save("rhombus-aa64.bmp");
}