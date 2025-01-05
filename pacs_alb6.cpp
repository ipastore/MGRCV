#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <thread>
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

void process_device(_cl_command_queue*& queue, 
                    _cl_context*& context, 
                    _cl_kernel*& kernel, 
                    const std::vector<_cl_mem*>& input_buffers, 
                    std::vector<_cl_mem*>& output_buffers) {
    cl_int err;
    cl_sampler sampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP, CL_FILTER_NEAREST, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Failed to create sample: " << err << std::endl;
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < input_buffers.size(); ++i) {
        // Set kernel arguments
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffers[i]);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffers[i]);
        clSetKernelArg(kernel, 2, sizeof(cl_sampler), &sampler);

        // Get the dimensions of the current buffer
        size_t width, height;
        size_t origin[3] = {0, 0, 0};
        size_t region[3];

        clGetImageInfo(input_buffers[i], CL_IMAGE_WIDTH, sizeof(size_t), &width, nullptr);
        clGetImageInfo(input_buffers[i], CL_IMAGE_HEIGHT, sizeof(size_t), &height, nullptr);
        region[0] = width;
        region[1] = height;
        region[2] = 1;

        // Set global and local work sizes
        size_t global_work_size[2] = {((width + local_size - 1) / local_size) * local_size,
                                      ((height + local_size - 1) / local_size) * local_size};
        size_t local_work_size[2] = {local_size, local_size};
        size_t local_mem_size = (local_work_size[0] + 2) * (local_work_size[1] + 2) * sizeof(float);
        clSetKernelArg(kernel, 3, local_mem_size, nullptr);


        // Enqueue the kernel
        err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global_work_size, local_work_size, 0, nullptr, nullptr);
        if (err != CL_SUCCESS)
        {
            std::cerr << "Failed to enqueue kernel: " << err << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // Wait for all commands to finish
    clFinish(queue);
    clReleaseSampler(sampler);
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
    const int num_replicas = 50;
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
    //*************** STEP 7: Distribute Fragments ****************//
    //**************************************************************//


    // Create OpenCL image objects
    cl_image_format format;
    format.image_channel_order = CL_R;
    format.image_channel_data_type = CL_UNORM_INT8;

    // Separate vectors for buffers in CPU and GPU contexts
    std::vector<cl_mem> cpu_input_buffers;
    std::vector<cl_mem> cpu_output_buffers;
    std::vector<cl_mem> gpu_input_buffers;
    std::vector<cl_mem> gpu_output_buffers;

    for (int i = 0; i < total_fragments; i++) {

        size_t fragment_size = fragments[i].size();

        if (i % 2 == 0) { // GPU fragment
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
        } else {
            cl_mem input_buffer = clCreateImage2D(cpu_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                    &format, fragments[i].width(), fragments[i].height(),
                                                    0, fragments[i].data(), &err);
            cl_error(err, "Failed to create input image");
            cpu_input_buffers.push_back(input_buffer);

            cl_mem output_buffer = clCreateImage2D(cpu_context, CL_MEM_WRITE_ONLY,
                                                    &format, fragments[i].width(), fragments[i].height(),
                                                    0, NULL, &err);
            cl_error(err, "Failed to create (GPU output buffer)");
            cpu_output_buffers.push_back(output_buffer);

            std::cout << "Created CPU buffers for fragment " << i << " (size: " << fragment_size << ")." << std::endl;
        }

    }

    //**************************************************************//
    //*************** STEP 7: Process Fragments *******************//
    //**************************************************************//

    std::thread cpu_thread(process_device,
                        &cpu_queue, &cpu_context, &cpu_kernel,
                        std::cref(cpu_input_buffers), std::ref(cpu_output_buffers));

    std::thread gpu_thread(process_device,
                        &gpu_queue, &gpu_context, &gpu_kernel,
                        std::cref(gpu_input_buffers), std::ref(gpu_output_buffers));

    cpu_thread.join(); 
    gpu_thread.join();


    //**************************************************************//
    //*************** STEP 8: Combine Results *********************//
    //**************************************************************//

    int gpu_fragment_to_save = 2; // Índice de fragmento procesado por la GPU (par)
    int cpu_fragment_to_save = 3; // Índice de fragmento procesado por la CPU (impar)

    save_processed_images(gpu_fragment_to_save, cpu_fragment_to_save, 
                        cpu_output_buffers, cpu_queue, 
                        gpu_output_buffers, gpu_queue, 
                        fragments, 
                        "../data/processed_gpu_fragment.jpg", 
                      "../data/processed_cpu_fragment.jpg");


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

