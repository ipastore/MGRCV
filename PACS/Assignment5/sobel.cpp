#include <OpenCL/opencl.h>
#define CL_TARGET_OPENCL_VERSION 120
#include "CImg.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace cimg_library;

// Function to load kernel source
const char* load_kernel_source(const char* filename) {
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open kernel file: " << filename << std::endl;
        exit(1);
    }
    std::string source((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
    char* kernel_source = new char[source.size() + 1];
    std::copy(source.begin(), source.end(), kernel_source);
    kernel_source[source.size()] = '\0';
    return kernel_source;
}

int main() {
    // OpenCL variables
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    // Image loading
    CImg<unsigned char> input_image("../montblanc.jpg");

    if (input_image.is_empty()) {
    std::cerr << "Failed to load the image. The image is empty." << std::endl;
    return 1;
    }

    std::cout << "Pixel at (10, 10): " 
          << "Red: " << (int)input_image(10, 10, 0, 0) 
          << ", Green: " << (int)input_image(10, 10, 0, 1) 
          << ", Blue: " << (int)input_image(10, 10, 0, 2) 
          << std::endl;

    int width = input_image.width();
    int height = input_image.height();
    CImg<unsigned char> gray_image = input_image.RGBtoYCbCr().channel(0);
    printf("Image width: %d, Image height: %d\n", width, height);
    printf("Number of slices: %d, Number of channels: %d\n", gray_image.depth(), gray_image.spectrum());
    
    // Initialize OpenCL
    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);

    // Create OpenCL image objects
    cl_image_format format = {CL_R, CL_UNORM_INT8};
    cl_mem input_cl_image = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            &format, width, height, 0, gray_image.data(), &err);
    cl_mem output_cl_image = clCreateImage2D(context, CL_MEM_WRITE_ONLY,
                                             &format, width, height, 0, NULL, &err);

    // Load and build the kernel
    const char* kernel_source = load_kernel_source("../sobel.cl");
    program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "sobel_filter", &err);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_cl_image);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_cl_image);
    cl_sampler sampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP, CL_FILTER_NEAREST, &err);
    clSetKernelArg(kernel, 2, sizeof(cl_sampler), &sampler);

    // Execute the kernel
    size_t global_size[] = {width, height};
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, NULL, 0, NULL, NULL);

    // Read output
    unsigned char* output_data = new unsigned char[width * height];
    size_t origin[] = {0, 0, 0};
    size_t region[] = {width, height, 1};
    clEnqueueReadImage(queue, output_cl_image, CL_TRUE, origin, region, 0, 0, output_data, 0, NULL, NULL);

    // Save output
    CImg<unsigned char> output_image(output_data, width, height, 1, 1);
    output_image.save("sobel_output.jpg");

    printf("Output image saved to sobel_output.jpg\n");

    // Cleanup
    delete[] output_data;
    delete[] kernel_source;
    clReleaseMemObject(input_cl_image);
    clReleaseMemObject(output_cl_image);
    clReleaseSampler(sampler);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}