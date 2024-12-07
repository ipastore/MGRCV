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
    // Create and save 3x3 RGB test image
    create_test_image("test_image_3x3.jpg", 3, 3);

    // Create and save 9x9 RGB test image
    create_test_image("test_image_9x9.jpg", 9, 9);

    // Create and save 20x20 RGB test image
    create_test_image("test_image_20x20.jpg", 20, 20);

    return 0;
}