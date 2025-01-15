__kernel void dot_product(
    // A pointer to the first matrix
    __global const float* A,
    // A pointer to the second matrix
    __global const float* B,
    // A pointer to the result matrix
    __global float* result,
    // The number of rows in the matrix
    const int numRows,
    // The number of columns in the matrix
    const int numCols)
{
    // Get the row index
    int row = get_global_id(0);
    // If the row index is less than the number of rows
    if (row < numRows) {
        // Initialize the dot product to 0
        float dot_product = 0.0f;
        // Iterate over the columns
        for (int col = 0; col < numCols; col++) {
            // Compute the dot product
            dot_product += A[row * numCols + col] * B[row * numCols + col];
        }
        // Store the result in the result matrix
        result[row] = dot_product;
    }
}


// Brief Comments
// 	1.	Goal: Each work‐item (indexed by row = get_global_id(0)) computes the dot product of the row of A and the row of B that correspond to that same row index.
// 	2.	Correctness:
// 	•	The code loops over col from 0 to numCols, accumulating the product of corresponding elements A[...] * B[...].
// 	•	The result is stored in result[row].
// 	•	The if (row < numRows) check ensures we do not access out of bounds.
// 	3.	Usage:
// 	•	We presumably launch the kernel in a 1D ND‐range of at least numRows work‐items.

// Overall, the kernel is simple and correct for computing “per‐row dot products” between the rows of A and B. Each row in A is of length numCols, and likewise for B.

// Potential Observations:
// 	•	It does a naive loop from 0..numCols in each thread, which is perfectly fine for moderate matrix sizes. For large arrays, we might consider more advanced optimizations (e.g., local memory).
// 	•	We rely on one dimension of parallelism here—one work‐item per row. This is usually enough if numRows is large enough to keep the GPU busy.

// OpenCL dot product function
void dot_product_openCL(const float* A, const float* B, float* result, int numRows, int numCols) {
    // Get the platform
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    // Get the device
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    // Create the context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    // Create the command queue
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);
    // Create the program
    const char* source = read_file("dot_product_openCL.cl");
    cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, NULL);
    // Build the program
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    // Create the kernel
    cl_kernel kernel = clCreateKernel(program, "dot_product", NULL);
    // Create the buffers
    cl_mem buffer_A = clCreateBuffer(context, CL_MEM_READ_ONLY, numRows * numCols * sizeof(float), NULL, NULL);
    cl_mem buffer_B = clCreateBuffer(context, CL_MEM_READ_ONLY, numRows * numCols * sizeof(float), NULL, NULL);
    cl_mem buffer_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, numRows * sizeof(float), NULL, NULL);
    // Write the buffers
    clEnqueueWriteBuffer(queue, buffer_A, CL_TRUE, 0, numRows * numCols * sizeof(float), A, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, buffer_B, CL_TRUE, 0, numRows * numCols * sizeof(float), B, 0, NULL, NULL);
    // Set the arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_result);
    clSetKernelArg(kernel, 3, sizeof(int), &numRows);
    clSetKernelArg(kernel, 4, sizeof(int), &numCols);
    // Execute the kernel
    size_t global_size = numRows;

    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    // Read the result
    clEnqueueReadBuffer(queue, buffer_result, CL_TRUE, 0, numRows * sizeof(float), result, 0, NULL, NULL);
    // Release the resources
    clReleaseMemObject(buffer_A);
    clReleaseMemObject(buffer_B);
    clReleaseMemObject(buffer_result);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

// From Build program
void launchDotProductKernel(
    cl_context context,
    cl_command_queue queue,
    cl_program program,
    const std::vector<float>& hostA,
    const std::vector<float>& hostB,
    std::vector<float>& hostResult,
    int numRows,
    int numCols)
{
    cl_int err = 0;

    // 1) Create device buffers
    // size of each array is numRows * numCols for A and B, and numRows for result
    size_t sizeAB = static_cast<size_t>(numRows) * numCols * sizeof(float);
    size_t sizeRes = static_cast<size_t>(numRows) * sizeof(float);

    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeAB, (void*)hostA.data(), &err);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeAB, (void*)hostB.data(), &err);
    cl_mem bufferRes = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                      sizeRes, nullptr, &err);

    // 2) Create the kernel
    cl_kernel kernel = clCreateKernel(program, "dot_product", &err);

    // 3) Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferRes);
    err = clSetKernelArg(kernel, 3, sizeof(int), &numRows);
    err = clSetKernelArg(kernel, 4, sizeof(int), &numCols);

    // 4) Decide global/local size
    // We'll launch numRows work-items in 1D
    size_t globalWorkSize[1] = { static_cast<size_t>(numRows) };

    // 5) Enqueue kernel
    err = clEnqueueNDRangeKernel(
        queue,
        kernel,
        1,           // 1D range
        nullptr,     // offset
        globalWorkSize,
        nullptr,     // localWorkSize (let runtime decide)
        0,
        nullptr,
        nullptr
    );

    // 6) Read the result from device
    err = clEnqueueReadBuffer(queue, bufferRes, CL_TRUE, 0,
                              sizeRes, hostResult.data(), 0, nullptr, nullptr);

    // 7) Cleanup (in real code, check for errors!)
    clReleaseKernel(kernel);
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferRes);
}

// Launch the kernel with local size
#include <CL/cl.h>
#include <vector>
#include <cmath>     // for ceil, if needed
#include <iostream>

// (Assume we already have a valid OpenCL context, queue, and program built.)

void launchMyDotProductWithLocalSize(
    cl_context context,
    cl_command_queue queue,
    cl_program program,
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& out,
    int numRows,
    int numCols)
{
    cl_int err = 0;

    // 1) Create buffers
    size_t sizeAB = static_cast<size_t>(numRows)*numCols*sizeof(float);
    size_t sizeRes= static_cast<size_t>(numRows)*sizeof(float);

    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeAB, (void*)A.data(), &err);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeAB, (void*)B.data(), &err);
    cl_mem bufR = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeRes, nullptr, &err);

    // 2) Create kernel
    cl_kernel kernel = clCreateKernel(program, "my_dot_product_local_size", &err);

    // 3) Set kernel args
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufR);
    err = clSetKernelArg(kernel, 3, sizeof(int), &numRows);
    err = clSetKernelArg(kernel, 4, sizeof(int), &numCols);

    // 4) Decide on local & global size
    //    Let's pick e.g. localSize=64.  Each group has 64 threads (rows).
    //    Then compute globalSize as the next multiple of localSize that covers numRows.
    size_t localSize  = 64;
    size_t remainder  = numRows % localSize;
    size_t globalSize = (remainder == 0) 
                        ? (size_t)numRows
                        : ( (size_t)numRows + (localSize - remainder) );

    // 5) Enqueue kernel with explicit local size
    err = clEnqueueNDRangeKernel(
        queue,
        kernel,
        1,             // 1D kernel
        nullptr,       // global offset
        &globalSize,   // global work size
        &localSize,    // local work size
        0,             // # events in wait list
        nullptr,       // wait list
        nullptr        // event
    );

    // 6) Read results
    err = clEnqueueReadBuffer(queue, bufR, CL_TRUE, 0, sizeRes,
                              out.data(), 0, nullptr, nullptr);

    // 7) Cleanup
    clReleaseKernel(kernel);
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufR);
}

// My version of the kernel
__kernel void dot_product_optimized(
    __global const float* A,
    __global const float* B,
    __global float* result,
    const int numRows,
    const int numCols)
{
    int row = get_global_id(0);
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);

    __local float localA[256];
    __local float localB[256];

    float dot_product = 0.0f;

    for (int i = 0; i < numCols; i += local_size) {
        if (i + local_id < numCols) {
            localA[local_id] = A[row * numCols + i + local_id];
            localB[local_id] = B[row * numCols + i + local_id];
        } else {
            localA[local_id] = 0.0f;
            localB[local_id] = 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int j = 0; j < local_size; j++) {
            dot_product += localA[j] * localB[j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < numRows) {
        result[row] = dot_product;
    }
}

// Brief Comments
// 	1.	Goal: Each work‐item computes the dot product of the row of A and the row of B that correspond to that same row index.
// 	2.	Correctness:
// 	•	We use local memory to cache chunks of A and B, which can reduce global memory accesses.
// 	•	We loop over the columns in chunks of local_size, loading data into local memory.
// 	•	We then compute the dot product using the local memory.
// 	•	We use a barrier to ensure all threads have loaded their data before computing the dot product.
// 	•	We store the result in the result array.
// 	3.	Usage:
// 	•	We launch the kernel with a 1D ND‐range of at least numRows work‐items.
// 	•	We set the local size to 64 in this example.

// Overall, the kernel is an optimized version of the previous dot product kernel. It uses local memory to cache chunks of A and B, reducing global memory accesses.
// The kernel is correct and should provide better performance for large matrices.

// Potential Observations:
// 	•	We use local memory to cache data, which can reduce global memory accesses. This can be beneficial for large matrices.
// 	•	We use a barrier to synchronize threads within a work‐group. This ensures all threads have loaded their data before computing the dot product.
// 	•	We loop over the columns in chunks of local_size, which can improve memory access patterns.

