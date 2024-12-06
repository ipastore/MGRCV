#define cimg_use_jpeg
#include <iostream>
#include "CImg.h"

using namespace cimg_library;

int main() {
    // Load image
    CImg<unsigned char> img("../blacklotus.jpg"); // Load image file
    std::cout << "Image width: " << img.width()
              << ", Image height: " << img.height()
              << ", Number of slices: " << img.depth()
              << ", Number of channels: " << img.spectrum() << std::endl;

    // Dump image data into an array
    int i = 100; 
    int j = 150; 
    std::cout << "Pixel at (" << i << ", " << j << "):" << std::endl;
    std::cout << "Red: " << std::hex << (int) img(i, j, 0, 0) << std::endl; // Red channel
    std::cout << "Green: " << std::hex << (int) img(i, j, 0, 1) << std::endl; // Green channel
    std::cout << "Blue: " << std::hex << (int) img(i, j, 0, 2) << std::endl; // Blue channel

    // Modify the image: Add a crossing blue line through the middle
    unsigned char blue[] = {0, 0, 255}; 
    int mid_height = img.height() / 2;  
    int mid_width = img.width() / 2;    

    img.draw_line(0, mid_height, img.width() - 1, mid_height, blue);

    img.display("Image with Blue Crossing Lines");

    return 0;
}