#define CL_TARGET_OPENCL_VERSION 120
#include "CImg.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#ifdef __APPLE__
  #include <OpenCL/opencl.h>
#else
  #include <CL/cl.h>
#endif

using namespace cimg_library;


void cl_error(cl_int code, const char *string){
    if (code != CL_SUCCESS){
        printf("%d - %s\n", code, string);
        exit(-1);
    }
}

#include <stdio.h>

void log_sobel_data(const char *filename, float *debug_data, int width, int height) {
    // Open the file for writing
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    // Write header
    fprintf(file, "x,y,gx,gy,gradient,gradient_after_clamp,\n");
    
    // Loop through each pixel and write its data
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 4; // Assuming each pixel has 4 debug values
            fprintf(file, "%d,%d,%f,%f,%f,%f\n", x, y, debug_data[idx], debug_data[idx + 1], 
            debug_data[idx + 2], debug_data[idx + 3]);
        }
    }

    // Close the file
    fclose(file);
    printf("Data logged to file: %s\n", filename);
}

int main() {
    // OpenCL variables
    int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue command_queue;
    cl_program program;
    cl_kernel kernel;

    std::string imgName = "20x20.jpg";
    // std::string imgName = "9x9.jpg";
    // std::string imgName = "3x3.jpg";
    // std::string imgName = "montblanc.jpg";
    // std::string imgName = "blacklotus.jpg";
 
    
    std::string imgPath = "../images/input/" + imgName;
    std::string outputImgPath = "../images/output/" + imgName;
    std::string grayPath = "../images/gray/" + imgName;


    // Image loading
    CImg<unsigned char> input_image(imgPath.c_str());
    
    const int width = input_image.width();
    const int height = input_image.height();

    // Convert image to grayscale
    CImg<unsigned char> gray_image = input_image.RGBtoYCbCr().channel(0);
    printf("Image width: %d, Image height: %d\n", width, height);
    printf("Number of slices: %d, Number of channels: %d\n", gray_image.depth(), gray_image.spectrum());
    
    // Display grey image
    // gray_image.display("Gray Image");
    gray_image.normalize(0,255).save(grayPath.c_str());

    // Get platform ID
    err = clGetPlatformIDs(1, &platform, NULL);
    cl_error(err, "Error: Failed to Scan for Platforms IDs");
    printf("Platform ID: %p\n", platform);

    // Get device ID
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    cl_error(err, "Error: Failed to get device IDs");
    printf("Device ID: %p\n", device);

    // Create a context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_error(err, "Failed to create a compute context\n");
    printf("Context: %p\n", context);

    // Create a command queue
    command_queue = clCreateCommandQueue(context, device, 0, &err);
    cl_error(err, "Failed to create a command queue\n");
    printf("Command Queue: %p\n", command_queue);


    // Load the kernel file
    FILE *fileHandler = fopen("../sobel.cl", "r");
    fseek(fileHandler, 0, SEEK_END);
    size_t fileSize = ftell(fileHandler);
    rewind(fileHandler);

    // Read kernel source into buffer
    char *sourceCode = (char*) malloc(fileSize + 1);
    sourceCode[fileSize] = '\0';
    fread(sourceCode, sizeof(char), fileSize, fileHandler);
    fclose(fileHandler);

    // Create program from buffer
    program = clCreateProgramWithSource(context, 1, (const char **)&sourceCode, &fileSize, &err);
    cl_error(err, "Failed to create program with source\n");
    free(sourceCode);

    // Build the executable and check errors
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS){
        size_t len;
        char buffer[2048];

        printf("Error: Some error at building process.\n");
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(-1);
    }
    // Create a compute kernel with the program we want to run
    kernel = clCreateKernel(program, "sobel_filter", &err);
    cl_error(err, "Failed to create kernel from the program\n");

    
    //6. Create OpenCL buffer visible to the OpenCl runtime
    // Create OpenCL image objects
    cl_image_format format;
    format.image_channel_order = CL_R;
    format.image_channel_data_type = CL_UNORM_INT8;

    cl_mem input_cl_image = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            &format, width, height, 0, gray_image.data(), &err);
    cl_mem output_cl_image = clCreateImage2D(context, CL_MEM_WRITE_ONLY,
                                             &format, width, height, 0, NULL, &err);

    // Write image into the the memory object
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {width, height, 1};
    // clEnqueueWriteImage(command_queue, input_cl_image, CL_TRUE, origin, region, 0, 0, gray_image.data(), 0, NULL, NULL);
    // cl_error(err, "Failed to enqueue write image command\n");

int num_intermediate_values = 4; // gx, gy and gradient magnitude and gradient after clamping
int debug_size = width * height * num_intermediate_values;

cl_mem debug_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * debug_size, NULL, &err);
cl_error(err, "Failed to create debug buffer");


    // Set the arguments to the kernel
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_cl_image);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_cl_image);
    cl_sampler sampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP, CL_FILTER_NEAREST, &err);
    cl_error(err, "Failed to create sampler\n");
    clSetKernelArg(kernel, 2, sizeof(cl_sampler), &sampler);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &debug_buffer); 
    printf("Arguments set\n");

    // Execute the kernel
    size_t global_size[2] = {width, height};
    // No local size?

    clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_size, NULL, 0, NULL, NULL);
    // Print global, size
    printf("Global size: %d, %d\n", global_size[0], global_size[1]);

    // Read output
    CImg<unsigned char> output_image(width, height, 1, 1);
    clEnqueueReadImage(command_queue, output_cl_image, CL_TRUE, origin, region, 0, 0, output_image.data(), 0, NULL, NULL);

    // Save output
    output_image.normalize(0,250).save(outputImgPath.c_str());
    output_image.display("Sobel Filter Output");
    printf("Output image saved to sobel_output.jpg\n");

    float *debug_data = (float *)malloc(sizeof(float) * debug_size);
    clEnqueueReadBuffer(command_queue, debug_buffer, CL_TRUE, 0, sizeof(float) * debug_size, debug_data, 0, NULL, NULL);

    // Log debug data
    log_sobel_data("sobel_debug_data.txt", debug_data, width, height);

    free(debug_data);

    // Cleanup
    clReleaseMemObject(input_cl_image);
    clReleaseMemObject(output_cl_image);
    clReleaseSampler(sampler);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    return 0;
}