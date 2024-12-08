#define cimg_use_jpeg
#include <iostream>
#include "CImg.h"

using namespace cimg_library;

void create_test_image(const char* filename, int width, int height) {
    // Create a blank RGB image with black background
    CImg<unsigned char> img(width, height, 1, 3, 0);

    // Define colors
    unsigned char red[] = {255, 0, 0};      // Red for vertical line
    unsigned char green[] = {0, 255, 0};   // Green for horizontal line

    // Add a vertical red line in the center
    img.draw_line(width / 2, 0, width / 2, height - 1, red);

    // Add a horizontal green line in the center
    img.draw_line(0, height / 2, width - 1, height / 2, green);

    // Save the image
    img.save(filename);

    std::cout << "Saved image: " << filename << " (" << width << "x" << height << ")" << std::endl;
}

int main() {
    create_test_image("../images/input/3x3.jpg", 3, 3);
    create_test_image("../images/input/9x9.jpg", 9, 9);
    create_test_image("../images/input/20x20.jpg", 20, 20);
    create_test_image("../images/input/256x256.jpg", 256, 256);
    create_test_image("../images/input/256x512.jpg", 256, 512);
    create_test_image("../images/input/512x512.jpg", 512, 512);
    create_test_image("../images/input/720x1280.jpg", 720, 1280);
    create_test_image("../images/input/1024x1024.jpg", 1024, 1024);
    create_test_image("../images/input/2048x4096.jpg", 2048, 4096);

    return 0;
}