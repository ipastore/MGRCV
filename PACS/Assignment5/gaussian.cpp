#ifdef __APPLE__
  #include <OpenCL/opencl.h>
#else
  #include <CL/cl.h>
#endif
#define CL_TARGET_OPENCL_VERSION 120
#include "CImg.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace cimg_library;

std::string load_kernel_source(const char* filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open kernel file: " + std::string(filepath));
    }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

int main() {
    // Load input image
    CImg<unsigned char> img("../montblanc.jpg");
    CImg<float> gray_image = img.RGBtoYCbCr().channel(0);

    const int width = gray_image.width();
    const int height = gray_image.height();

    // OpenCL setup
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    // 1. Select platform and device
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    // 2. Create context and command queue
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);

    // 3. Load and compile kernel
    std::string kernel_source = load_kernel_source("../gaussian.cl");
    const char* kernel_cstr = kernel_source.c_str();
    program = clCreateProgramWithSource(context, 1, &kernel_cstr, NULL, &err);
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "gaussian_filter", &err);

    // 4. Create input and output images
    cl_image_format format;
    format.image_channel_order = CL_RGBA;
    format.image_channel_data_type = CL_UNORM_INT8;

    cl_mem input_image = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format,
                                         width, height, 0, gray_image.data(), &err);
    cl_mem output_image = clCreateImage2D(context, CL_MEM_WRITE_ONLY, &format, width, height, 0, NULL, &err);

    // 5. Set kernel arguments
    cl_sampler sampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP, CL_FILTER_NEAREST, &err);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_image);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_image);
    clSetKernelArg(kernel, 2, sizeof(cl_sampler), &sampler);

    // 6. Launch the kernel
    size_t global_size[2] = { (size_t)width, (size_t)height };
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, NULL, 0, NULL, NULL);

    // 7. Read the output image
    CImg<float> output_img(width, height, 1, 1);
    size_t origin[3] = { 0, 0, 0 };
    size_t region[3] = { (size_t)width, (size_t)height, 1 };
    clEnqueueReadImage(queue, output_image, CL_TRUE, origin, region, 0, 0, output_img.data(), 0, NULL, NULL);

    // 8. Save and display the output image
    output_img.normalize(0, 255).save("gaussian_output.jpg");
    output_img.display("Gaussian Filter Output");
    printf("Output image saved to gaussian_output.jpg\n");

    // Clean up
    clReleaseMemObject(input_image);
    clReleaseMemObject(output_image);
    clReleaseSampler(sampler);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}