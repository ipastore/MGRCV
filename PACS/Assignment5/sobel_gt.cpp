#include <math.h>
#include <stdio.h>
#define cimg_use_jpeg
#include <iostream>
#include "CImg.h"
#include <fstream>


using namespace cimg_library;

// Sobel kernels
const int kernelX[3][3] = {{-1, 0, 1}, 
                           {-2, 0, 2}, 
                           {-1, 0, 1}};

const int kernelY[3][3] = {{ 1,  2,  1}, 
                           { 0,  0,  0}, 
                           {-1, -2, -1}};

// Sobel filter function
void sobelFilter(const CImg<unsigned char>& grayscale, CImg<unsigned char>& output, const std::string& logFilePath) {
    int width = grayscale.width();
    int height = grayscale.height();

    // Initialize output image
    output.fill(0);

        // Open log file
    std::ofstream logFile(logFilePath);
    if (!logFile.is_open()) {
        std::cerr << "Failed to open log file: " << logFilePath << std::endl;
        std::ofstream newLogFile(logFilePath);
        if (!newLogFile.is_open()) {
            std::cerr << "Failed to create log file: " << logFilePath << std::endl;
            return;
        }
        logFile.swap(newLogFile);
    }

    // Log header
    logFile << "x,y,gx,gy,gradient,gradient_after_clamp\n";

    // Apply Sobel filter
    cimg_forXYC(grayscale, x, y, c) {
        if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
            output(x, y, c) = 0; // Set edge pixels to 0
            
            // log
            logFile << x << "," << y << ",0,0,0,0\n";
            
            continue;
        }

        float magX = 0.0;
        float magY = 0.0;

        // Convolve with Sobel kernels
        for (int a = -1; a <= 1; a++) {
            for (int b = -1; b <= 1; b++) {
                int xn = x + b;
                int yn = y + a;
                float pixel = grayscale(xn, yn, c);
                magX += pixel * kernelX[a + 1][b + 1];
                magY += pixel * kernelY[a + 1][b + 1];
            }
        }

        
        // Compute gradient magnitude
        float magnitude = sqrt(magX * magX + magY * magY);

        // Clamp to [0, 255] and set pixel
        float clamped_magnitude = std::min(255.0f, std::max(0.0f, magnitude));
        output(x, y, c) = static_cast<unsigned char>(clamped_magnitude);

        // Log values
        logFile << x << "," << y << "," << magX << "," << magY << "," << magnitude << "," << clamped_magnitude << "\n";
    }

    // Close log file
    logFile.close();
    std::cout << "Gradient data logged to " << logFilePath << std::endl;
    
}

int main() {

    std::string imgName = "20x20.jpg";
    // std::string imgName = "9x9.jpg";
    // std::string imgName = "3x3.jpg";
    // std::string imgName = "montblanc.jpg";
    // std::string imgName = "blacklotus.jpg";
 
    
    std::string imgPath = "../images/input/" + imgName;
    std::string outputImgPath = "../images/gt/" + imgName;
    std::string grayPath = "../images/gray/gt" + imgName;
    std::string logFilePath = "../log/gt/" + imgName + ".csv"; 


    // Image loading
    CImg<unsigned char> img(imgPath.c_str());

    // Convert image to grayscale
    CImg<unsigned char> gray_image = img.RGBtoYCbCr().channel(0);

    // Display grey image
    // gray_image.display("Gray Image");
    gray_image.normalize(0,255).save(grayPath.c_str());

    // Create Output image
    CImg<unsigned char> output_image(gray_image.width(), gray_image.height(), 1, 1,0);

    // Apply Sobel filter
    sobelFilter(gray_image, output_image, logFilePath);

    output_image.display("Sobel Filter Output");

    output_image.save(outputImgPath.c_str());

    printf("Output image saved\n");
    
    return 0;
}