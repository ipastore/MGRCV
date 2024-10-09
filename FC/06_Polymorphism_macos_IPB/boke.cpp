#include "boke.h"

    std::string Boke::name() const {
        return "Boke";
    }


// Counter of calls to Boke edit method
int Boke::counter = 0;

// After 10 calls to the edit method, the color state is toggled
void Boke::edit(unsigned char& r, unsigned char& g, unsigned char& b) {
    if (counter % 1000 < 700) {
        // Set color to blue
        r = 0;
        g = 0;
        b = 255;
    } else {
        // Set color to yellow
        r = 255;
        g = 255;
        b = 0;
    }
    counter++;
}




