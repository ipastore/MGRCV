__kernel void flipImage(
    __global const float* in,
    __global float* out,
    const unsigned int width,
    const unsigned int height)
{
    // Each work-item gets (col, row) from the global IDs
    int col = get_global_id(0);
    int row = get_global_id(1);

    // Make sure we’re in range
    if(col >= width || row >= height) {
        return;  // out of bounds
    }

    // The “mirrored” column index
    int sym_col = width - col - 1;

    // Compute linear indices for input & output
    // Let row-major indexing: index = row * width + col
    int inIdx  = row * width + col;
    int outIdx = row * width + sym_col;

    // Copy from input to the mirrored position in output
    out[outIdx] = in[inIdx];
}


//Launck the kernel in OPENCL
void flipImageKernel(cl_mem in, cl_mem out, unsigned int width, unsigned int height, cl_command_queue queue)
{
    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &in);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &out);
    clSetKernelArg(kernel, 2, sizeof(unsigned int), &width);
    clSetKernelArg(kernel, 3, sizeof(unsigned int), &height);

    // Launch the kernel
    size_t globalSize[2] = {width, height};
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL);
}