// File: vector_add_spr.cl

// We'll process 16 floats at a time (float16) to encourage AVX-512 usage.
// If id*16 >= N, we skip. If there's a remainder, we handle it in a tail case.

__kernel void vector_add_spr(__global const float16* A,
                             __global const float16* B,
                             __global float16*       C,
                             const unsigned int N) 
{
    // each work item handles 16 floats
    int id = get_global_id(0);

    // This 'id' indexes a chunk of size 16 in the array-of-float16
    // total # of float16 elements is N/16 if we assume N multiple of 16
    // but let's do a boundary check in case remainder or so
    // We'll interpret "N" as # of float elements total
    // => the # of float16 elements = N/16
    // For safety:
    int totalChunks = (N + 15) / 16;  // ceiling

    if (id < totalChunks) {
        float16 aVal = A[id];
        float16 bVal = B[id];
        float16 cVal = aVal + bVal;
        C[id] = cVal;
    }
}


#include <CL/cl.h>
#include <vector>
#include <iostream>
#include <cmath>

// Hypothetical function that loads kernel source from "vector_add_spr.cl"
std::string loadKernelSource(const char* filename);

// Utility to round up N to a multiple of 16
size_t roundUp16(size_t N) {
    return (N + 15) & ~((size_t)15);
}

int main()
{
    // Example size
    const size_t N = 1000000; // 1M floats
    // We'll store them in float16 chunks
    size_t chunks = (N + 15)/16;  // # of float16 elements

    // 1) Setup CPU-based OpenCL environment 
    // (Find platform for Intel, find CPU device, create context, queue, etc.)
    // Skipping details - assume we have: cl_context context; cl_command_queue queue; etc.

    // 2) Prepare host arrays 
    // We'll store them as float
    // We'll need to interpret them as float16 at the device side, so 
    // the total array of float is N, but we can do "chunks" of float16 in device memory
    std::vector<float> hostA(N), hostB(N), hostC(N, 0.0f);
    // Initialize A and B
    for (size_t i = 0; i < N; ++i) {
        hostA[i] = (float)i;
        hostB[i] = (float)(N - i);
    }

    // 3) Create device buffers (in terms of float16 count)
    cl_int err;
    size_t bufSize = chunks * sizeof(float)*16; // each chunk is float16 => 16 floats
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 bufSize, hostA.data(), &err);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 bufSize, hostB.data(), &err);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bufSize,
                                 nullptr, &err);

    // 4) Build program from source
    std::string src = loadKernelSource("vector_add_spr.cl");
    const char* srcPtr = src.c_str();
    size_t srcLen = src.size();
    cl_program program = clCreateProgramWithSource(context, 1, &srcPtr, &srcLen, &err);
    err = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);

    // 5) Create kernel
    cl_kernel kernel = clCreateKernel(program, "vector_add_spr", &err);

    // 6) Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    // pass the total # of floats
    unsigned int N_ui = (unsigned int)N;
    err = clSetKernelArg(kernel, 3, sizeof(unsigned int), &N_ui);

    // 7) Enqueue kernel
    // The # of work items = # of float16 chunks
    size_t globalSize = chunks;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr,
                                 &globalSize, nullptr,
                                 0, nullptr, nullptr);

    // 8) Read results
    err = clEnqueueReadBuffer(queue, bufC, CL_TRUE,
                              0, bufSize,
                              hostC.data(), 
                              0, nullptr, nullptr);

    // 9) Verify or use results
    // For i in [0..N), hostC[i] should be hostA[i] + hostB[i]
    // e.g. check a few
    std::cout << "C[100] = " << hostC[100] << std::endl;

    // Cleanup
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    // plus context/queue destructors
    return 0;
}