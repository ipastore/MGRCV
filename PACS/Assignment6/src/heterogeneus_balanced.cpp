#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <thread>
#include "CImg/CImg.h"
#include <fstream>
#include <chrono>

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


void process_device(cl_command_queue queue, 
                    cl_context context, 
                    cl_kernel kernel, 
                    const std::vector<CImg<unsigned char>> &fragments, 
                    std::vector<CImg<unsigned char>> &output_fragments,
                    double &computation_time, double &communication_time,
                    double &total_time,
                    double &buffer_creation_time,
                    double &kernel_setup_time,
                    double &resource_release_time) {

    cl_int err;
    size_t local_size = 8;
    cl_image_format format;
    format.image_channel_order = CL_R;
    format.image_channel_data_type = CL_UNORM_INT8;

    cl_sampler sampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP, CL_FILTER_NEAREST, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Failed to create sample: " << err << std::endl;
        exit(EXIT_FAILURE);
    }

    computation_time = 0.0;
    communication_time = 0.0;
    total_time = 0.0;
    buffer_creation_time = 0.0;
    kernel_setup_time = 0.0;
    resource_release_time = 0.0;

    auto start_process = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < fragments.size(); ++i) {

        // Crete input and output buffers
        auto start = std::chrono::high_resolution_clock::now();
        size_t fragment_size = fragments[i].size();

        cl_mem input_buffer = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                &format, fragments[i].width(), fragments[i].height(),
                                                0, (void*)fragments[i].data(), &err);
        cl_error(err, "Failed to create input image");

        cl_mem output_buffer = clCreateImage2D(context, CL_MEM_WRITE_ONLY,
                                                &format, fragments[i].width(), fragments[i].height(),
                                                0, NULL, &err);
        cl_error(err, "Failed to create (GPU output buffer)");
        auto end = std::chrono::high_resolution_clock::now();
        buffer_creation_time += std::chrono::duration<double, std::milli>(end - start).count();


        // Set kernel arguments
        start = std::chrono::high_resolution_clock::now();
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);
        clSetKernelArg(kernel, 2, sizeof(cl_sampler), &sampler);


        // Set global and local work sizes
        size_t global_work_size[2] = {((fragments[i].width() + local_size - 1) / local_size) * local_size,
                                      ((fragments[i].height() + local_size - 1) / local_size) * local_size};
        size_t local_work_size[2] = {local_size, local_size};
        size_t local_mem_size = (local_work_size[0] + 2) * (local_work_size[1] + 2) * sizeof(float);
        clSetKernelArg(kernel, 3, local_mem_size, nullptr);
        end = std::chrono::high_resolution_clock::now();
        kernel_setup_time += std::chrono::duration<double, std::milli>(end - start).count();


        // Enqueue the kernel
        cl_event kernel_event;
        err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global_work_size, local_work_size, 0, nullptr, &kernel_event);
        cl_error(err, "Failed to enqueue kernel");
        clWaitForEvents(1, &kernel_event);

        cl_ulong start_time, end_time;
        clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, nullptr);
        clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, nullptr);
        computation_time += (end_time - start_time) / 1e6; // Convert to ms
        clReleaseEvent(kernel_event);

        // Read back the output buffer
        CImg<unsigned char> output_fragment(fragments[i].width(), fragments[i].height(), 1, 1);
        size_t origin[3] = {0, 0, 0};
        size_t region[3] = {static_cast<size_t>(fragments[i].width()), static_cast<size_t>(fragments[i].height()), 1};
        cl_event read_event;
        clEnqueueReadImage(queue, output_buffer, CL_TRUE, origin, region, 0, 0, output_fragment.data(), 0, nullptr, &read_event);
        clWaitForEvents(1, &read_event);

        clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, nullptr);
        clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, nullptr);
        communication_time += (end_time - start_time) / 1e6; // Convert to ms
        clReleaseEvent(read_event);

        output_fragments.push_back(output_fragment);

        // Cleanup for each fragment
        start = std::chrono::high_resolution_clock::now();
        clReleaseMemObject(input_buffer);
        clReleaseMemObject(output_buffer);
        end = std::chrono::high_resolution_clock::now();
        resource_release_time += std::chrono::duration<double, std::milli>(end - start).count();
    }

    // Wait for all commands to finish
    clFinish(queue);
    clReleaseSampler(sampler);

    auto end_process = std::chrono::high_resolution_clock::now();
    total_time = std::chrono::duration<double, std::milli>(end_process - start_process).count();
}


void save_processed_images(int gpu_fragment_index, int cpu_fragment_index,
                           const std::vector<cl_mem> &cpu_output_buffers, cl_command_queue cpu_queue,
                           const std::vector<cl_mem> &gpu_output_buffers, cl_command_queue gpu_queue,
                           const std::vector<CImg<unsigned char>> &fragments, 
                           const std::string &gpu_output_path, const std::string &cpu_output_path) {
    cl_int err;

    // --- Guardar fragmento procesado por la GPU ---
    size_t gpu_width = fragments[gpu_fragment_index].width();
    size_t gpu_height = fragments[gpu_fragment_index].height();

    CImg<unsigned char> gpu_image(gpu_width, gpu_height, 1, 1);
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {gpu_width, gpu_height, 1};

    clEnqueueReadImage(gpu_queue, gpu_output_buffers[gpu_fragment_index / 2], CL_TRUE, origin, region, 0, 0, gpu_image.data(), 0, nullptr, nullptr);
    gpu_image.save(gpu_output_path.c_str());
    std::cout << "Saved GPU processed fragment to: " << gpu_output_path << std::endl;

    // --- Guardar fragmento procesado por la CPU ---
    size_t cpu_width = fragments[cpu_fragment_index].width();
    size_t cpu_height = fragments[cpu_fragment_index].height();

    CImg<unsigned char> cpu_image(cpu_width, cpu_height, 1, 1);

    region[0] = cpu_width;
    region[1] = cpu_height;
    region[2] = 1;

    clEnqueueReadImage(cpu_queue, cpu_output_buffers[cpu_fragment_index / 2], CL_TRUE, origin, region, 0, 0, cpu_image.data(), 0, nullptr, nullptr);
    cpu_image.save(cpu_output_path.c_str());
    std::cout << "Saved CPU processed fragment to: " << cpu_output_path << std::endl;
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
        if (!gpu_device)
        {
            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &gpu_device, NULL);
            if (err != CL_DEVICE_NOT_FOUND)
            {
                cl_error(err, "clGetDeviceIDs (query GPU)");
            }
        }

        if (!cpu_device)
        {
            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 1, &cpu_device, NULL);
            if (err != CL_DEVICE_NOT_FOUND)
            {
                cl_error(err, "clGetDeviceIDs (query CPU)");
            }
        }


        if (gpu_device && cpu_device)
        {
            break;
        }
    }

    if (!gpu_device && !cpu_device)
    {
        std::cerr << "Error: Could not find both: GPU and CPU device." << std::endl;
        return EXIT_FAILURE;
    }

    // Print selected devices
    char device_name[128];
    clGetDeviceInfo(cpu_device, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
    std::cout << "Selected CPU: " << device_name << std::endl;
    clGetDeviceInfo(gpu_device, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
    std::cout << "Selected GPU: " << device_name << std::endl;

    //****************************************************************//
    //********* STEP 3: Simulating the large set of IMAGES ***********//
    //****************************************************************//

    CImg<unsigned char> base_image("../data/blacklotus.jpg");
    const int num_replicas = 5000;
    CImg<unsigned char> simulated_image(base_image.width(), base_image.height() * num_replicas, 1, 1, 0);

    for (int i = 0; i < num_replicas; i++)
    {
        simulated_image.draw_image(0, i * base_image.height(), 0, 0, base_image);
    }

    std::cout << "Simulated large image with dimensions: " << simulated_image.width() << "x" << simulated_image.height() << std::endl;

    //**************************************************************//
    //*********** STEP 4: Divide the image into fragments***********//
    //**************************************************************//

    int total_fragments = num_replicas; // Split between CPU and GPU
    auto fragments = divide_image(simulated_image, total_fragments);
    std::cout << "Divided image into " << total_fragments << " fragments." << std::endl;

    // Split fragments between CPU and GPU
    std::vector<CImg<unsigned char>> cpu_fragments;
    std::vector<CImg<unsigned char>> gpu_fragments;
    // Output fragments for processed results
    std::vector<CImg<unsigned char>> cpu_output_fragments;
    std::vector<CImg<unsigned char>> gpu_output_fragments;

    int gpu_fragments_count = static_cast<int>(0.75 * total_fragments);
    int cpu_fragments_count = total_fragments - gpu_fragments_count;

    for (int i = 0; i < gpu_fragments_count; i++) {
        gpu_fragments.push_back(fragments[i]);
    }
    for (int i = gpu_fragments_count; i < total_fragments; i++) {
        cpu_fragments.push_back(fragments[i]);
    }

    //**************************************************************************//
    //********* STEP 5: Creating the CONTEXT & QUEUES for the Devices *********//
    //*************************************************************************//

    // CPU setup
    cl_context cpu_context = clCreateContext(nullptr, 1, &cpu_device, nullptr, nullptr, &err);
    cl_error(err, "Failed to create CPU context");
    cl_command_queue cpu_queue = clCreateCommandQueue(cpu_context, cpu_device, CL_QUEUE_PROFILING_ENABLE, &err);
    cl_error(err, "Failed to create CPU queue");

    // GPU setup
    cl_context gpu_context = clCreateContext(nullptr, 1, &gpu_device, nullptr, nullptr, &err);
    cl_error(err, "Failed to create GPU context");
    cl_command_queue gpu_queue = clCreateCommandQueue(gpu_context, gpu_device, CL_QUEUE_PROFILING_ENABLE, &err);
    cl_error(err, "Failed to create GPU queue");


    //**************************************************//
    //****** STEP 6: Load and build the kernels ********//
    //**************************************************//

    // Load and compile kernels for CPU and GPU
    std::ifstream kernel_file("../src/sobel_filter.cl");
    std::string kernel_code((std::istreambuf_iterator<char>(kernel_file)), std::istreambuf_iterator<char>());
    const char *kernel_source = kernel_code.c_str();

    // Set up Kernel for the CPU
    cl_program cpu_program = clCreateProgramWithSource(cpu_context, 1, &kernel_source, nullptr, &err);
    cl_error(err, "Failed to create CPU program");
    clBuildProgram(cpu_program, 1, &cpu_device, nullptr, nullptr, nullptr);
    cl_kernel cpu_kernel = clCreateKernel(cpu_program, "sobel_filter", &err);
    cl_error(err, "Failed to create CPU kernel");

    // Set up kernel for the GPU
    cl_program gpu_program = clCreateProgramWithSource(gpu_context, 1, &kernel_source, nullptr, &err);
    cl_error(err, "Failed to create GPU program");
    clBuildProgram(gpu_program, 1, &gpu_device, nullptr, nullptr, nullptr);
    cl_kernel gpu_kernel = clCreateKernel(gpu_program, "sobel_filter", &err);
    cl_error(err, "Failed to create GPU kernel");


    //**************************************************************//
    //*************** STEP 7: Process Fragments *******************//
    //**************************************************************//

    double cpu_computation_time = 0.0, cpu_communication_time = 0.0;
    double gpu_computation_time = 0.0, gpu_communication_time = 0.0;
    double gpu_total_time = 0.0, cpu_total_time = 0.0;
    double gpu_buffer_creation_time = 0.0, cpu_buffer_creation_time = 0.0;
    double gpu_kernel_setup_time = 0.0, cpu_kernel_setup_time = 0.0;
    double gpu_resource_release_time = 0.0, cpu_resource_release_time = 0.0;


    auto start = std::chrono::high_resolution_clock::now();
    std::thread cpu_thread(process_device,
                           cpu_queue, cpu_context, cpu_kernel,
                           std::cref(cpu_fragments), std::ref(cpu_output_fragments),
                           std::ref(cpu_computation_time), std::ref(cpu_communication_time),
                           std::ref(cpu_total_time), std::ref(cpu_buffer_creation_time),
                           std::ref(cpu_kernel_setup_time), std::ref(cpu_resource_release_time));

    std::thread gpu_thread(process_device,
                           gpu_queue, gpu_context, gpu_kernel,
                           std::cref(gpu_fragments), std::ref(gpu_output_fragments),
                           std::ref(gpu_computation_time), std::ref(gpu_communication_time),
                           std::ref(gpu_total_time), std::ref(gpu_buffer_creation_time),
                           std::ref(gpu_kernel_setup_time), std::ref(gpu_resource_release_time));    

    cpu_thread.join();
    gpu_thread.join();


    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;


    //**************************************************************//
    //*************** STEP 8: Results *********************//
    //**************************************************************//


    // Print computation and communication times
    std::cout << "Number of total images processed: " << num_replicas << " Images" << std::endl;
    std::cout << "Number of images processed by the CPU: " << cpu_fragments.size() << " Images" << std::endl;
    std::cout << "Number of images processed by the GPU: " << gpu_fragments.size() << " Images" << std::endl;

    std::cout << "Total execution THREADS calculated in MAIN: " << elapsed.count() << " ms" << std::endl;
    std::cout << "Total time for CPU_process_device: " << cpu_total_time << " ms" << std::endl;
    std::cout << "Total time for GPU_process_device: " << gpu_total_time << " ms" << std::endl;

    std::cout << "CPU Buffer creation time: " << cpu_buffer_creation_time << " ms" << std::endl;
    std::cout << "GPU Buffer creation time: " << gpu_buffer_creation_time << " ms" << std::endl;

    std::cout << "CPU Kernel setup time: " << cpu_kernel_setup_time << " ms" << std::endl;
    std::cout << "GPU Kernel setup time: " << gpu_kernel_setup_time << " ms" << std::endl;

    std::cout << "CPU Resource release time: " << cpu_resource_release_time << " ms" << std::endl;
    std::cout << "GPU Resource release time: " << gpu_resource_release_time << " ms" << std::endl;

    std::cout << "CPU Computation Time: " << cpu_computation_time << " ms" << std::endl;
    std::cout << "CPU Communication Time: " << cpu_communication_time << " ms" << std::endl;

    std::cout << "GPU Computation Time: " << gpu_computation_time << " ms" << std::endl;
    std::cout << "GPU Communication Time: " << gpu_communication_time << " ms" << std::endl;

    // Calculate workload balance
    double total_computation_time = cpu_computation_time + gpu_computation_time;
    double total_communication_time = cpu_communication_time + gpu_communication_time;
    std::cout << "Workload Balance (CPU %): " << (cpu_computation_time / total_computation_time) * 100 << "%" << std::endl;
    std::cout << "Workload Balance (GPU %): " << (gpu_computation_time / total_computation_time) * 100 << "%" << std::endl;

    try {
        gpu_output_fragments[0].normalize(0, 250).save("../data/processed_gpu_fragment.jpg");
        std::cout << "Saved GPU processed fragment to disk." << std::endl;
    } catch (const CImgIOException &e) {
        std::cerr << "Error saving GPU processed fragment: " << e.what() << std::endl;
    }
    try {
        cpu_output_fragments[1].normalize(0, 250).save("../data/processed_cpu_fragment.jpg");
        std::cout << "Saved CPU processed fragment to disk." << std::endl;
    } catch (const CImgIOException &e) {
        std::cerr << "Error saving CPU processed fragment: " << e.what() << std::endl;
    }

    // Cleanup
    clReleaseKernel(cpu_kernel);
    clReleaseProgram(cpu_program);
    clReleaseCommandQueue(cpu_queue);
    clReleaseContext(cpu_context);

    clReleaseKernel(gpu_kernel);
    clReleaseProgram(gpu_program);
    clReleaseCommandQueue(gpu_queue);
    clReleaseContext(gpu_context);

    return 0;
}

