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

    cl_device_id gpu_device = nullptr;

    for (size_t i = 0; i < platforms.size(); i++)
    {
        if (!gpu_device)
        {
            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &gpu_device, NULL);
            if (err != CL_DEVICE_NOT_FOUND)
            {
                cl_error(err, "clGetDeviceIDs (query GPU)");
            }
        }

        if (gpu_device)
        {
            break;
        }
    }

    if (!gpu_device)
    {
        std::cerr << "Error: Could not find any GPU device." << std::endl;
        return EXIT_FAILURE;
    }

    // Print selected devices
    char device_name[128];
    clGetDeviceInfo(gpu_device, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
    std::cout << "Selected GPU: " << device_name << std::endl;
    

    //****************************************************************//
    //********* STEP 3: Creating the CONTEXT for the Devices *********//
    //****************************************************************//

    cl_context gpu_context = clCreateContext(NULL, 1, &gpu_device, NULL, NULL, &err);
    cl_error(err, "Failed to create context fot he GPU");

    //****************************************************************//
    //********* STEP 4: Creating the QUEUES for the Devices **********//
    //****************************************************************//

    cl_command_queue gpu_queue = clCreateCommandQueue(gpu_context, gpu_device, CL_QUEUE_PROFILING_ENABLE, &err);
    cl_error(err, "Failed to create command queue");

    //****************************************************************//
    //********* STEP 5: Simulating the large set of IMAGES ***********//
    //****************************************************************//

    // TODO: Como se simula una imagen? Se esta haciendo bien?
    CImg<unsigned char> base_image("../data/blacklotus.jpg");
    const int num_replicas = 5000;
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

    //*************************************************//
    //****** STEP 8: Load and build the kernel ********//
    //*************************************************//

    std::ifstream kernel_file("../src/sobel_filter.cl");
    std::string kernel_code((std::istreambuf_iterator<char>(kernel_file)), std::istreambuf_iterator<char>());
    const char *kernel_source = kernel_code.c_str();

    cl_program gpu_program = clCreateProgramWithSource(gpu_context, 1, &kernel_source, NULL, &err);
    cl_error(err, "clCreateProgramWithSource (GPU)");

    err = clBuildProgram(gpu_program, 1, &gpu_device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(gpu_program, gpu_device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::vector<char> build_log(log_size);
        clGetProgramBuildInfo(gpu_program, gpu_device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), NULL);
        std::cerr << "Build log (GPU):\n" << build_log.data() << std::endl;
        cl_error(err, "clBuildProgram (GPU)");
    }

    // Create kernels
    cl_kernel gpu_kernel = clCreateKernel(gpu_program, "sobel_filter", &err);
    cl_error(err, "Failed to create kernel");

    //**************************************************************************//
    //****** STEP 9: Set kernel arguments and enqueue for each fragment ********//
    //**************************************************************************//

    // Create OpenCL image objects
    cl_image_format format;
    format.image_channel_order = CL_R;
    format.image_channel_data_type = CL_UNORM_INT8;

        // Separate vectors for buffers in CPU and GPU contexts
    std::vector<cl_mem> cpu_input_buffers;
    std::vector<cl_mem> cpu_output_buffers;
    std::vector<cl_mem> gpu_input_buffers;
    std::vector<cl_mem> gpu_output_buffers;

    for (int i = 0; i < num_fragments; i++) {

        size_t fragment_size = fragments[i].size();

        cl_mem input_buffer = clCreateImage2D(gpu_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                &format, fragments[i].width(), fragments[i].height(),
                                                0, fragments[i].data(), &err);
        cl_error(err, "Failed to create input image");
        gpu_input_buffers.push_back(input_buffer);

        cl_mem output_buffer = clCreateImage2D(gpu_context, CL_MEM_WRITE_ONLY,
                                                &format, fragments[i].width(), fragments[i].height(),
                                                0, NULL, &err);
        cl_error(err, "Failed to create (GPU output buffer)");
        gpu_output_buffers.push_back(output_buffer);

        std::cout << "Created GPU buffers for fragment " << i << " (size: " << fragment_size << ")." << std::endl;

    }

    size_t local_size = 8;
    cl_sampler sampler = clCreateSampler(gpu_context, CL_FALSE, CL_ADDRESS_CLAMP, CL_FILTER_NEAREST, &err);
    cl_error(err, "Failed to create sampler\n");

    double gpu_computation_time = 0.0, gpu_communication_time = 0.0;

    // Separate kernels for CPU and GPU
    for (int i = 0; i < num_fragments; i++) {
        
        clSetKernelArg(gpu_kernel, 0, sizeof(cl_mem), &gpu_input_buffers[i]);
        clSetKernelArg(gpu_kernel, 1, sizeof(cl_mem), &gpu_output_buffers[i]);
        clSetKernelArg(gpu_kernel, 2, sizeof(cl_sampler), &sampler);


        size_t global_work_size[2] = {((fragments[i].width() + local_size - 1) / local_size) * local_size,
                                    ((fragments[i].height() + local_size - 1) / local_size) * local_size};

        // size_t global_work_size[2] =    {static_cast<size_t>(fragments[i].width()),
        //                                 static_cast<size_t>(fragments[i].height())};
        size_t local_work_size[2] = {local_size, local_size};
        size_t local_mem_size = (local_work_size[0] + 2) * (local_work_size[1] + 2) * sizeof(float);
        clSetKernelArg(gpu_kernel, 3, local_mem_size, nullptr);

        // Enqueue the kernel
        cl_event kernel_event;
        err = clEnqueueNDRangeKernel(gpu_queue, gpu_kernel, 2, nullptr, global_work_size, local_work_size, 0, nullptr, &kernel_event);
        cl_error(err, "Failed to enqueue kernel");
        clWaitForEvents(1, &kernel_event);

        cl_ulong start_time, end_time;
        clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, nullptr);
        clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, nullptr);
        gpu_computation_time += (end_time - start_time) / 1e6; // Convert to ms
        clReleaseEvent(kernel_event);

        CImg<unsigned char> gpu_output_image(fragments[i].width(), fragments[i].height(), 1, 1);
        size_t origin[3] = {0, 0, 0};
        size_t region[3] = {static_cast<size_t>(fragments[i].width()), static_cast<size_t>(fragments[i].height()), 1};
        cl_event read_event;
        clEnqueueReadImage(gpu_queue, gpu_output_buffers[i], CL_TRUE, origin, region, 0, 0, gpu_output_image.data(), 0, NULL, &read_event);
        clWaitForEvents(1, &read_event);
        clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, nullptr);
        clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, nullptr);
        gpu_communication_time += (end_time - start_time) / 1e6; // Convert to ms
        clReleaseEvent(read_event);

        if (i == 54) { 
            gpu_output_image.normalize(0,250).save("../data/test_processed_blacklotus_gpu.jpg");
            std::cout << "Saved GPU processed fragment to disk." << std::endl;
        }
    }

    //*******************************************************//
    //****** STEP 10: Wait for all kernels to finish ********//
    //*******************************************************//

    clFinish(gpu_queue);

    // Print computation and communication times
    std::cout << "Number of total images processed: " << num_replicas << " Images" << std::endl;
    std::cout << "GPU Computation Time: " << gpu_computation_time << " ms" << std::endl;
    std::cout << "GPU Communication Time: " << gpu_communication_time << " ms" << std::endl;


    //********************************************************//
    //****** STEP 11: Releasing all the resources used ********//
    //********************************************************//


    // Release GPU buffers
    for (auto buffer : gpu_input_buffers) {
        clReleaseMemObject(buffer);
    }
    for (auto buffer : gpu_output_buffers) {
        clReleaseMemObject(buffer);
    }

    clReleaseSampler(sampler);
    // Release kernels
    clReleaseKernel(gpu_kernel);
    // Release programs
    clReleaseProgram(gpu_program);
    // Release command queues
    clReleaseCommandQueue(gpu_queue);
    // Release contexts
    clReleaseContext(gpu_context);

    return 0;
}
