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



void run_sobel_filter(const std::string& image_file, size_t global_height, size_t global_width, size_t local_size, std::ofstream& log_file) {
    // Load the image
    CImg<unsigned char> image(image_file.c_str());
    int width = image.width();
    int height = image.height();

    // Convert image to grayscale
    CImg<float> grayscale = image.RGBtoYCbCr().channel(0);
    std::vector<float> input_data(grayscale.begin(), grayscale.end());

    // Output image data
    std::vector<float> output_data(width * height);

    // OpenCL initialization
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue command_queue;
    std::string image_name = image_file.substr(image_file.find_last_of("/\\") + 1);
    std::string outputImgPath = "../images/output/" + image_name;

    err = clGetPlatformIDs(1, &platform, NULL);
    cl_error(err, "Failed to get platform ID");

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    cl_error(err, "Failed to get device ID");

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_error(err, "Failed to create context");

    command_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    cl_error(err, "Failed to create command queue");

    // Load kernel source
    std::ifstream kernel_file("../sobel_local.cl");
    std::string kernel_code((std::istreambuf_iterator<char>(kernel_file)), std::istreambuf_iterator<char>());
    const char* kernel_source = kernel_code.c_str();

    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    cl_error(err, "Failed to create program");

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        // Print build log on error
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::vector<char> build_log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), NULL);
        std::cerr << "Build log:\n" << build_log.data() << std::endl;
        exit(EXIT_FAILURE);
    }

    

    cl_kernel kernel = clCreateKernel(program, "sobel_filter", &err);
    cl_error(err, "Failed to create kernel");

    // Create OpenCL image objects
    cl_image_format format;
    format.image_channel_order = CL_R;
    format.image_channel_data_type = CL_UNORM_INT8;

    // Input object
    cl_mem input_cl_image = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            &format, width, height, 0, grayscale.data(), &err);
    cl_error(err, "Failed to create input image");
  
    // Output object
    cl_mem output_cl_image = clCreateImage2D(context, CL_MEM_WRITE_ONLY,
                                                &format, width, height, 0, NULL, &err);
    cl_error(err, "Failed to create output image");
    
    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_cl_image);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_cl_image);
    cl_sampler sampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP, CL_FILTER_NEAREST, &err);
    cl_error(err, "Failed to create sampler\n");
    clSetKernelArg(kernel, 2, sizeof(cl_sampler), &sampler);
    
    // Define global and local sizes
    size_t global_work_size[] = {global_width,global_height};
    size_t local_work_size[] = {local_size, local_size};

    // Allocate shared memory for local block (width + 2 halo pixels) x (height + 2 halo pixels)
    size_t local_mem_size = (local_work_size[0] + 2) * (local_work_size[1] + 2) * sizeof(float);
    clSetKernelArg(kernel, 3, local_mem_size, NULL); // Pass NULL for dynamically allocated shared memory

    // Profiling
    cl_event kernel_event;
    auto start_time = std::chrono::high_resolution_clock::now();

    err = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, &kernel_event);
    cl_error(err, "Failed to enqueue kernel");

    clWaitForEvents(1, &kernel_event);

    // Measure kernel execution time
    cl_ulong time_start, time_end;
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    double kernel_exec_time = (time_end - time_start) / 1e6; // ms

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_exec_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    // Read output
    CImg<unsigned char> output_image(width, height, 1, 1);
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {width, height, 1};
    clEnqueueReadImage(command_queue, output_cl_image, CL_TRUE, origin, region, 0, 0, output_image.data(), 0, NULL, NULL);

    // Save output
    output_image.normalize(0,250).save(outputImgPath.c_str());

    // Log results
    double bandwidth = (width * height * sizeof(float)) / (kernel_exec_time * 1e-3); // MB/s
    double throughput = (width * height) / (kernel_exec_time * 1e-3); // Pixels/s
    double memory_footprint = width * height * sizeof(float) * 2 / 1e6; // MB

    log_file << height << "," << width  << "," << local_size << ","
             << total_exec_time << "," << kernel_exec_time << ","
             << bandwidth << "," << throughput << "," << memory_footprint << "\n";

    // Cleanup
    clReleaseMemObject(input_cl_image);
    clReleaseMemObject(output_cl_image);
    clReleaseSampler(sampler);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
}

int main() {
    // List of test image file names
    std::vector<std::string> images_names;
    images_names.push_back("256x256.jpg");
    images_names.push_back("256x512.jpg");
    images_names.push_back("512x512.jpg");
    images_names.push_back("720x1280.jpg");
    images_names.push_back("1024x1024.jpg");
    images_names.push_back("2048x4096.jpg");
  
    // Open CSV file for logging
    std::ofstream log_file("../log/results/dataset.csv", std::ios::out);
    log_file << "height,width,l_size,total_exec,kernel_exec,bandwidth,throughput,memory_footprint\n";

    // Test configurations
    std::vector<size_t> local_sizes;
    local_sizes.push_back(2);
    local_sizes.push_back(4);
    local_sizes.push_back(8);
    local_sizes.push_back(16);


    for (const auto& image : images_names) {
        
        std::string image_path = "../images/input/" + image;

        // Load global image dimensions dynamically
        CImg<unsigned char> input_image(image_path.c_str());
        size_t width = input_image.width();
        size_t height = input_image.height();

        
        for (size_t l_size : local_sizes) {
            for (int i = 0; i < 5; i++) {
                if ((height * width) % l_size == 0) {
                    printf("Running Sobel filter for image: %s, local size: %lu\n", image.c_str(), l_size);
                    run_sobel_filter(image_path, height, width, l_size, log_file);
                } else {
                    printf("Local size %lu is not a divisor of global size %lu\n", l_size, height * width);
                }
            }
        }

    }

    log_file.close();
    return 0;
}