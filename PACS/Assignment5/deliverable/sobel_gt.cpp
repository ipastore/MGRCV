#include <chrono>
#include <math.h>
#include <stdio.h>
#define cimg_use_jpeg
#include <iostream>
#include "CImg.h"
#include <fstream>
#include <string>
#include <vector>

using namespace cimg_library;

// Sobel kernels
const int kernelX[3][3] = {{-1, 0, 1}, 
                           {-2, 0, 2}, 
                           {-1, 0, 1}};

const int kernelY[3][3] = {{ 1,  2,  1}, 
                           { 0,  0,  0}, 
                           {-1, -2, -1}};

// Sobel filter function
void sobelFilter(const CImg<unsigned char>& input, CImg<unsigned char>& output) {
// , const std::string& logFilePath ) {  // LOG FOR DEBUG
    
    int width = input.width();
    int height = input.height();

    // Initialize output image
    output.fill(0);

    // Open log for DEBUG
    // std::ofstream logFile(logFilePath);
    // if (!logFile.is_open()) {
    //     std::cerr << "Failed to open log file: " << logFilePath << std::endl;
    //     std::ofstream newLogFile(logFilePath);
    //     if (!newLogFile.is_open()) {
    //         std::cerr << "Failed to create log file: " << logFilePath << std::endl;
    //         return;
    //     }
    //     logFile.swap(newLogFile);
    // }

    // Log header for DEBUG
    // logFile << "x,y,gx,gy,gradient,gradient_after_clamp\n";

    // Apply Sobel filter
    cimg_forXYC(input, x, y, c) {
        if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
            output(x, y, c) = 0; // Set edge pixels to 0
            
            // // LOG for DEBUG
            // logFile << x << "," << y << ",0,0,0,0\n";
            
            continue;
        }

        float magX = 0.0;
        float magY = 0.0;

        // Convolve with Sobel kernels
        for (int a = -1; a <= 1; a++) {
            for (int b = -1; b <= 1; b++) {
                int xn = x + b;
                int yn = y + a;
                float pixel = input(xn, yn, c);
                magX += pixel * kernelX[a + 1][b + 1];
                magY += pixel * kernelY[a + 1][b + 1];
            }
        }

        
        // Compute gradient magnitude
        float magnitude = sqrt(magX * magX + magY * magY);

        // Clamp to [0, 255] and set pixel
        float clamped_magnitude = std::min(255.0f, std::max(0.0f, magnitude));
        output(x, y, c) = static_cast<unsigned char>(clamped_magnitude);

        // Log for DEBUG
        // logFile << x << "," << y << "," << magX << "," << magY << "," << magnitude << "," << clamped_magnitude << "\n";
    }

    // log for DEBUG
    // logFile.close();
    // std::cout << "Gradient data logged to " << logFilePath << std::endl;
    
}

int main() {

    const int numRuns = 5;

    // List of test image file names
    std::vector<std::string> images_names;
    images_names.push_back("256x256.jpg");
    images_names.push_back("256x512.jpg");
    images_names.push_back("512x512.jpg");
    images_names.push_back("720x1280.jpg");
    images_names.push_back("1024x1024.jpg");
    images_names.push_back("2048x4096.jpg");
  
    std::ofstream resultsFile("../log/results/cpu_sobel_results.csv", std::ios::out);
    resultsFile << "width,height,total_exec, kernel_exec\n";
    
    for (const auto& image : images_names) {
        for (int run = 0; run < numRuns; run++) {
            printf("Running Sobel filter for image: %s, run: %d\n", image.c_str(), run);
            std::string inputImg = "../images/input/" + image;
            std::string image_name = image.substr(image.find_last_of("/\\") + 1);
            std::string outputImg = "../images/gt/" + image_name;
            // std::string logFilePath = "../log/gt/" + image + ".csv";  // LOG FOR DEBUG

            // Start total timer
            auto start_total = std::chrono::high_resolution_clock::now();

            // Image loading
            CImg<unsigned char> input_image(inputImg.c_str());
            // height and width of the image
            size_t width = input_image.width();
            size_t height = input_image.height();

            // Create Output image
            CImg<unsigned char> output_image(width, height, 1, 1,0);


            auto start_sobel = std::chrono::high_resolution_clock::now();
            sobelFilter(input_image, output_image);
            // , logFilePath); // LOG FOR DEBUG
            auto end_sobel = std::chrono::high_resolution_clock::now();

            // // Display and save output image
            // outputImg.display("Sobel Filter Output");
            output_image.save(outputImg.c_str());
            // printf("Output image saved\n");

            // End total timer
            auto end_total = std::chrono::high_resolution_clock::now();
            double totalExecTime = std::chrono::duration<double, std::milli>(end_total - start_total).count();
            
            // Compute execution time in milliseconds
            double sobelExecTime = std::chrono::duration<double, std::milli>(end_sobel - start_sobel).count();

            // Log results
            resultsFile << width << "," << height << "," << totalExecTime<< "," << sobelExecTime << "\n";
            
        }
    }
    resultsFile.close();
    printf("Results logged to file");
    return 0;
}