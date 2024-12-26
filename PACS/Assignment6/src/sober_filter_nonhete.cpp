#include <CL/cl.h>
#include <iostream>
#include <vector>
#include "CImg/CImg.h"
#include <fstream>

using namespace cimg_library;

// Function to check OpenCL errors
void cl_error(cl_int err, const char *operation)
{
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error during operation '" << operation << "': " << err << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Function to divide a simulated image into fragments
std::vector<CImg<unsigned char>> divide_image(const CImg<unsigned char> &simulated_image, int num_fragments)
{
    std::vector<CImg<unsigned char>> fragments;
    int fragment_height = simulated_image.height() / num_fragments;

    for (int i = 0; i < num_fragments; i++)
    {
        int y_start = i * fragment_height;
        int y_end = (i == num_fragments - 1) ? simulated_image.height() : (i + 1) * fragment_height;
        fragments.push_back(simulated_image.get_crop(0, y_start, simulated_image.width() - 1, y_end - 1));
    }

    return fragments;
}

int main()
{

    cl_int err;

    //******************************************************//
    //******** STEP 1: Getting the OpenCL PLATFORMS *******//
    //******************************************************//

    cl_uint num_platforms;
    err = clGetPlatformIDs(0, nullptr, &num_platforms);
    cl_error(err, "clGetPlatformIDs (query number of platforms)");

    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    cl_error(err, "clGetPlatformIDs (get platforms)");

    //******************************************************//
    //********* STEP 2: Getting the OpenCL DEVICES *********//
    //******************************************************//

    cl_device_id cpu_device = nullptr;
    cl_device_id gpu_device = nullptr;

    for (size_t i = 0; i < platforms.size(); i++)
    {
        if (!cpu_device)
        {
            cl_uint num_cpus;
            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 1, &cpu_device, &num_cpus);
            if (err != CL_DEVICE_NOT_FOUND)
            {
                cl_error(err, "clGetDeviceIDs (query CPU)");
            }
        }

        if (!gpu_device)
        {
            cl_uint num_gpus;
            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &gpu_device, &num_gpus);
            if (err != CL_DEVICE_NOT_FOUND)
            {
                cl_error(err, "clGetDeviceIDs (query GPU)");
            }
        }

        if (cpu_device && gpu_device)
        {
            break;
        }
    }

    if (!cpu_device || !gpu_device)
    {
        std::cerr << "Error: Could not find both CPU and GPU devices." << std::endl;
        return EXIT_FAILURE;
    }

    // Print selected devices
    char device_name[128];
    if (cpu_device)
    {
        clGetDeviceInfo(cpu_device, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
        std::cout << "Selected CPU: " << device_name << std::endl;
    }

    if (gpu_device)
    {
        clGetDeviceInfo(gpu_device, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
        std::cout << "Selected GPU: " << device_name << std::endl;
    }

    //****************************************************************//
    //********* STEP 3: Creating the CONTEXT for the Devices *********//
    //****************************************************************//

    std::cout << "Creating a context for the  CPU:" << std::endl;
    cl_context cpu_context = clCreateContext(NULL, 1, &cpu_device, NULL, NULL, &err);
    cl_error(err, "clCreateContext (CPU)");

    std::cout << "Creating a context for the  GPU:" << std::endl;
    cl_context gpu_context = clCreateContext(NULL, 1, &gpu_device, NULL, NULL, &err);
    cl_error(err, "clCreateContext (GPU)");

    //****************************************************************//
    //********* STEP 4: Creating the QUEUES for the Devices **********//
    //****************************************************************//

    cl_command_queue cpu_queue = clCreateCommandQueueWithProperties(cpu_context, cpu_device, nullptr, &err);
    cl_error(err, "clCreateCommandQueueWithProperties (CPU)");

    cl_command_queue gpu_queue = clCreateCommandQueueWithProperties(gpu_context, gpu_device, nullptr, &err);
    cl_error(err, "clCreateCommandQueueWithProperties (GPU)");

    //****************************************************************//
    //********* STEP 5: Simulating the large set of IMAGES ***********//
    //****************************************************************//

    // TODO: Como se simula una imagen? Se esta haciendo bien?
    CImg<unsigned char> base_image("./data/blacklotus.jpg");
    const int num_replicas = 50;
    CImg<unsigned char> simulated_image(base_image.width(), base_image.height() * num_replicas, 1, 1, 0);

    for (int i = 0; i < num_replicas; i++)
    {
        simulated_image.draw_image(0, i * base_image.height(), 0, 0, base_image);
    }

    std::cout << "Simulated large image with dimensions: " << simulated_image.width() << "x" << simulated_image.height() << std::endl;

    //**************************************************************//
    //*********** STEP 6: Divide the image into fragments***********//
    //**************************************************************//

    // TODO: Aclarar por qué se divide la imagen en dos fragmentos
    int num_fragments = num_replicas; // Split between CPU and GPU
    auto fragments = divide_image(simulated_image, num_fragments);
    std::cout << "Divided image into " << num_fragments << " fragments." << std::endl;

    //**************************************************************//
    //****** STEP 7: Create OpenCL buffers for each fragment********//
    //**************************************************************//

    // Separate vectors for buffers in CPU and GPU contexts
    std::vector<cl_mem> cpu_input_buffers;
    std::vector<cl_mem> cpu_output_buffers;
    std::vector<cl_mem> gpu_input_buffers;
    std::vector<cl_mem> gpu_output_buffers;

    for (int i = 0; i < num_fragments; i++) {
        size_t fragment_size = fragments[i].size();

        // Allocate buffers in the appropriate context
        if (i % 2 == 0) { // GPU
            cl_mem input_buffer = clCreateBuffer(gpu_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, fragment_size, fragments[i].data(), &err);
            cl_error(err, "clCreateBuffer (GPU input buffer)");
            gpu_input_buffers.push_back(input_buffer);

            cl_mem output_buffer = clCreateBuffer(gpu_context, CL_MEM_WRITE_ONLY, fragment_size, nullptr, &err);
            cl_error(err, "clCreateBuffer (GPU output buffer)");
            gpu_output_buffers.push_back(output_buffer);

            std::cout << "Created GPU buffers for fragment " << i << " (size: " << fragment_size << ")." << std::endl;
        } else { // CPU
            cl_mem input_buffer = clCreateBuffer(cpu_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, fragment_size, fragments[i].data(), &err);
            cl_error(err, "clCreateBuffer (CPU input buffer)");
            cpu_input_buffers.push_back(input_buffer);

            cl_mem output_buffer = clCreateBuffer(cpu_context, CL_MEM_WRITE_ONLY, fragment_size, nullptr, &err);
            cl_error(err, "clCreateBuffer (CPU output buffer)");
            cpu_output_buffers.push_back(output_buffer);

            std::cout << "Created CPU buffers for fragment " << i << " (size: " << fragment_size << ")." << std::endl;
        }
    }

    //*************************************************//
    //****** STEP 8: Load and build the kernel ********//
    //*************************************************//

    std::ifstream kernel_file("./src/sobel_filter.cl");
    std::string kernel_code((std::istreambuf_iterator<char>(kernel_file)), std::istreambuf_iterator<char>());
    const char *kernel_source = kernel_code.c_str();

    // Separate programs for CPU and GPU
    cl_program cpu_program = clCreateProgramWithSource(cpu_context, 1, &kernel_source, nullptr, &err);
    cl_error(err, "clCreateProgramWithSource (CPU)");
    

    cl_program gpu_program = clCreateProgramWithSource(gpu_context, 1, &kernel_source, nullptr, &err);
    cl_error(err, "clCreateProgramWithSource (GPU)");

    // Build programs
    err = clBuildProgram(cpu_program, 1, &cpu_device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(cpu_program, cpu_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> build_log(log_size);
        clGetProgramBuildInfo(cpu_program, cpu_device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);
        std::cerr << "Build log (CPU):\n" << build_log.data() << std::endl;
        cl_error(err, "clBuildProgram (CPU)");
    }

    err = clBuildProgram(gpu_program, 1, &gpu_device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(gpu_program, gpu_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> build_log(log_size);
        clGetProgramBuildInfo(gpu_program, gpu_device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);
        std::cerr << "Build log (GPU):\n" << build_log.data() << std::endl;
        cl_error(err, "clBuildProgram (GPU)");
    }

    // Create kernels
    cl_kernel cpu_kernel = clCreateKernel(cpu_program, "sobel_filter", &err);
    cl_error(err, "clCreateKernel (CPU)");

    cl_kernel gpu_kernel = clCreateKernel(gpu_program, "sobel_filter", &err);
    cl_error(err, "clCreateKernel (GPU)");

    //**************************************************************************//
    //****** STEP 9: Set kernel arguments and enqueue for each fragment ********//
    //**************************************************************************//


    CImg<unsigned char> cpu_output_image;
    CImg<unsigned char> gpu_output_image;

    size_t local_work_size[2] = {16, 16};
    size_t local_mem_size = (local_work_size[0] + 2) * (local_work_size[1] + 2) * sizeof(float);


    // Separate kernels for CPU and GPU
    for (int i = 0; i < num_fragments; i++) {
        if (i % 2 == 0) { // GPU fragment
            clSetKernelArg(gpu_kernel, 0, sizeof(cl_mem), &gpu_input_buffers[i / 2]);
            clSetKernelArg(gpu_kernel, 1, sizeof(cl_mem), &gpu_output_buffers[i / 2]);
            clSetKernelArg(gpu_kernel, 2, sizeof(cl_sampler), nullptr); // Add sampler if required
            clSetKernelArg(gpu_kernel, 3, local_mem_size, nullptr); // Local memory size for halo

            size_t global_work_size[2] =    {static_cast<size_t>(fragments[i].width()),
                                            static_cast<size_t>(fragments[i].height())};

            clEnqueueNDRangeKernel(gpu_queue, gpu_kernel, 2, nullptr, global_work_size, local_work_size, 0, nullptr, nullptr);

            std::cout << "Kernel enqueued for fragment " << i << " on GPU." << std::endl;

        } else { // CPU fragment
            clSetKernelArg(cpu_kernel, 0, sizeof(cl_mem), &cpu_input_buffers[i / 2]);
            clSetKernelArg(cpu_kernel, 1, sizeof(cl_mem), &cpu_output_buffers[i / 2]);
            clSetKernelArg(cpu_kernel, 2, sizeof(cl_sampler), nullptr); // Add sampler if required
            clSetKernelArg(cpu_kernel, 3, local_mem_size, nullptr); // Local memory size for halo

            size_t global_work_size[2] =    {static_cast<size_t>(fragments[i].width()),
                                            static_cast<size_t>(fragments[i].height())};

            clEnqueueNDRangeKernel(cpu_queue, cpu_kernel, 2, nullptr, global_work_size, local_work_size, 0, nullptr, nullptr);

            std::cout << "Kernel enqueued for fragment " << i << " on CPU." << std::endl;
        }

        // Save processed output for one CPU and one GPU fragment
        if (i == 0) { // Save first GPU fragment
            gpu_output_image = CImg<unsigned char>(fragments[i].width(), fragments[i].height(), 1, 1);
            clEnqueueReadBuffer(gpu_queue, gpu_output_buffers[i / 2], CL_TRUE, 0, gpu_output_image.size(), gpu_output_image.data(), 0, nullptr, nullptr);
            gpu_output_image.save("./data/processed_blacklotus_gpu.jpg");
            std::cout << "Saved GPU processed fragment to disk." << std::endl;
        } else if (i == 1) { // Save first CPU fragment
            cpu_output_image = CImg<unsigned char>(fragments[i].width(), fragments[i].height(), 1, 1);
            clEnqueueReadBuffer(cpu_queue, cpu_output_buffers[i / 2], CL_TRUE, 0, cpu_output_image.size(), cpu_output_image.data(), 0, nullptr, nullptr);
            cpu_output_image.save("./data/processed_blacklotus_cpu.jpg");
            std::cout << "Saved CPU processed fragment to disk." << std::endl;
        }
    }

    //*******************************************************//
    //****** STEP 10: Wait for all kernels to finish ********//
    //*******************************************************//

    clFinish(cpu_queue);
    clFinish(gpu_queue);

    //********************************************************//
    //****** STEP 11: Releasing all the resources used ********//
    //********************************************************//

    // Release CPU buffers
    for (auto buffer : cpu_input_buffers) {
        clReleaseMemObject(buffer);
    }
    for (auto buffer : cpu_output_buffers) {
        clReleaseMemObject(buffer);
    }
    // Release GPU buffers
    for (auto buffer : gpu_input_buffers) {
        clReleaseMemObject(buffer);
    }
    for (auto buffer : gpu_output_buffers) {
        clReleaseMemObject(buffer);
    }

    // Release kernels
    clReleaseKernel(cpu_kernel);
    clReleaseKernel(gpu_kernel);
    // Release programs
    clReleaseProgram(cpu_program);
    clReleaseProgram(gpu_program);
    // Release command queues
    clReleaseCommandQueue(cpu_queue);
    clReleaseCommandQueue(gpu_queue);
    // Release contexts
    clReleaseContext(cpu_context);
    clReleaseContext(gpu_context);

    return 0;
}
