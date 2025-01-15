//Write an OpenCL program that calculates the dot product of to integer arrays. Additionally to the kernel
// code, in the host side of the program, just focus on the buffer management, command-queue management
// and kernel launch.
// a. Please, analytically model the execution time of this work assuming the computational device has the
// following characteristics: 8 compute units, each compute unit has 128 parallel cores, each core has
// two floating-point arithmetic units and, frequency of the computational device is 1.5GHz. Assumption
// 1: just floating point instructions contribute to the execution time. Assumption 2: each FPU can
// process a floating point instrution per cycle.

#include <iostream>
#include <vector>
#include <CL/cl.h>
#include <fstream>
#include <string>
#include <chrono>

void cl_error(cl_int code, const char *string){
    if (code != CL_SUCCESS){
        printf("%d - %s\n", code, string);
        exit(-1);
    }
}

int main() {
    // Load kernel source
    std::ifstream kernel_file("dot_product_integer_arrays.cl");
    std::string kernel_code((std::istreambuf_iterator<char>(kernel_file)), std::istreambuf_iterator<char>());
    const char* kernel_source = kernel_code.c_str();
    
    // Create OpenCL context
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue command_queue;
    cl_program program;
    cl_kernel kernel;
    cl_int err;
    err = clGetPlatformIDs(1, &platform, NULL);
    cl_error(err, "Error: Failed to Scan for Platforms IDs");

    // Get device ID
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    cl_error(err, "Error: Failed to get device IDs");

    // Create a context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_error(err, "Failed to create a compute context\n");

    // Create a command queue
    command_queue = clCreateCommandQueue(context, device, 0, &err);
    cl_error(err, "Failed to create a command queue\n");

    // Create program from source
    program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    cl_error(err, "Failed to create program");

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
    kernel = clCreateKernel(program, "dot_product", &err);
    cl_error(err, "Failed to create kernel from the program\n");

    // Create input data
    std::vector<int> a = {1, 2, 3, 4, 5};
    std::vector<int> b = {6, 7, 8, 9, 10};
    int n = a.size
    // Create buffers
    cl_mem buffer_a = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * n, NULL, &err);
    cl_mem buffer_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * n, NULL, &err);
    cl_mem buffer_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int), NULL, &err);

    // Write data to the buffers
    clEnqueueWriteBuffer(command_queue, buffer_a, CL_TRUE, 0, sizeof(int) * n, a.data(), 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue, buffer_b, CL_TRUE, 0, sizeof(int) * n, b.data(), 0, NULL, NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_a);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_b);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_c);
    clSetKernelArg(kernel, 3, sizeof(int), &n);

    // Execute kernel
    size_t global_size = n;
    cl_event event;
    auto start = std::chrono::high_resolution_clock::now();
    clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, &event);
    clWaitForEvents(1, &event);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    printf("Execution time: %f\n", duration.count());

    // Read the result
    int c;
    clEnqueueReadBuffer(command_queue, buffer_c, CL_TRUE, 0, sizeof(int), &c, 0, NULL, NULL);
    printf("Dot product: %d\n", c);

    // Release resources
    clReleaseMemObject(buffer_a);
    clReleaseMemObject(buffer_b);
    clReleaseMemObject(buffer_c);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    return 0;

}

// Write the kernel code here
__kernel void dot_product(__global int* a, __global int* b, __global int* c, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    *c = sum;
}

//Write another kernel for the dot product of two integer arrays. 
//The kernel should be able to handle arrays of any size.




// 3) Analytical Execution‐Time Model

// We have a device with:
// 	•	8 compute units (CUs), each with 128 parallel cores,
// 	•	Each core has 2 FPUs,
// 	•	Frequency = 1.5 GHz = 1.5\times10^9 cycles/s,
// 	•	Only floating‐point instructions contribute to execution time,
// 	•	Each FPU can do 1 floating‐point instruction per cycle.

// 3.1) Total Theoretical FP Throughput
// 	1.	Per core, we have 2 FPUs → 2 floating ops/cycle/core,
// 	2.	Each CU has 128 cores → 128 × 2 = 256 FP ops/cycle/CU,
// 	3.	The device has 8 CUs → 8 × 256 = 2048 FP ops/cycle for the entire device,
// 	4.	At 1.5 GHz → 1.5\times10^9 \times 2048 = 3.072\times10^{12} floating ops/second = 3.072 TFLOPS.

// Hence in an ideal scenario, the GPU can achieve up to 3.072 trillion FLOPs per second.

// 3.2) Dot‐Product Operation Count

// For each element A[i]*B[i] plus add to partial sum, that’s typically:
// 	•	1 multiply,
// 	•	1 add (to accumulate).

// So 2 FLOPs per element. For N elements, that’s 2\,N FLOPs total.

// 3.3) Execution Time Estimate

// If we ignore memory overhead and only count the “2 FLOPs/element”:


// \text{Time}
// = \frac{2N}{\text{FLOP/s}}.

// Given the device can do ~3.072\times10^{12} FLOPs/s:


// \text{Time}
// = \frac{2\,N}{3.072\times10^{12}}
// = \frac{2\,N}{3.072\times10^{12}}\ \text{seconds}.


// For example, if N=10^7,


// \text{Time}
// \approx \frac{2\times10^7}{3.072\times10^{12}}
// = 6.51\times10^{-6}\ \text{s}
// = 6.51\,\mu\text{s}

// in an ideal scenario ignoring memory latencies, kernel‐launch overhead, etc.

// (Real performance is usually lower, but this is the straightforward formula under the stated assumptions: only FP instructions matter, each FPU does 1 instruction per cycle, etc.)