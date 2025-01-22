// File: game_of_life.cl

__kernel void game_of_life(
    __global const char* oldGrid,  // input grid
    __global char* newGrid,        // output grid
    const int width,
    const int height)
{
    // global IDs for row and column
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Check bounds
    if(x >= width || y >= height) {
        return;
    }

    // Compute the index of this cell
    int idx = y * width + x;

    // Count neighbors
    int neighbors = 0;
    // We'll do boundary checks for each neighbor
    // top row: y-1, middle row: y, bottom row: y+1
    // left col: x-1, center: x, right col: x+1

    for(int dy = -1; dy <= 1; dy++){
        for(int dx_ = -1; dx_ <= 1; dx_++){
            if(dx_ == 0 && dy == 0) {
                // don't count the cell itself
                continue;
            }
            int nx = x + dx_;
            int ny = y + dy;
            // boundary check
            if(nx >= 0 && nx < width && ny >= 0 && ny < height){
                int nIdx = ny * width + nx;
                if(oldGrid[nIdx] == 1) {
                    neighbors++;
                }
            }
        }
    }

    // Current state
    char cellState = oldGrid[idx]; // 1=alive, 0=dead

    // Apply Game of Life rules:
    // 1) A live cell with fewer than 2 live neighbors dies
    // 2) A live cell with 2 or 3 neighbors survives
    // 3) A live cell with more than 3 neighbors dies
    // 4) A dead cell with exactly 3 neighbors becomes alive

    char nextState = cellState; // default

    if(cellState == 1) {
        // alive
        if(neighbors < 2) {
            // underpopulation
            nextState = 0;
        } else if(neighbors > 3) {
            // overpopulation
            nextState = 0;
        } else {
            // stays alive if 2 or 3 neighbors
            nextState = 1;
        }
    } else {
        // dead
        if(neighbors == 3) {
            // reproduction
            nextState = 1;
        } else {
            nextState = 0;
        }
    }

    // Write the result
    newGrid[idx] = nextState;
}

#include <CL/cl.h>
#include <vector>
#include <iostream>

// Hypothetical function that loads the kernel source from "game_of_life.cl"
extern std::string loadKernelSource(const char* filename);

int main()
{
    // -------------------------------------------------------
    // 1) Setup OpenCL environment (platform, device, context)
    // This part is highly implementation-specific, so we’ll skip
    // details. We assume we have a valid context, device, queue
    // in the variables: cl_context context; cl_command_queue queue; etc.
    // -------------------------------------------------------

    // Example sizes
    const int width = 512;
    const int height= 512;
    size_t totalSize = width * height;

    // We'll store the grid as a vector of chars: 1=alive, 0=dead
    std::vector<char> hostOldGrid(totalSize, 0);
    std::vector<char> hostNewGrid(totalSize, 0);

    // Fill hostOldGrid with some initial pattern
    // e.g. random or glider, etc.

    // 2) Create device buffers
    cl_int err = 0;
    cl_mem bufOld = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   totalSize*sizeof(char),
                                   hostOldGrid.data(), &err);
    cl_mem bufNew = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                   totalSize*sizeof(char),
                                   nullptr, &err);

    // 3) Build the program from source
    std::string kernelSrc = loadKernelSource("game_of_life.cl");
    const char* srcPtr = kernelSrc.c_str();
    size_t srcLen = kernelSrc.size();
    cl_program program = clCreateProgramWithSource(context, 1, &srcPtr, &srcLen, &err);
    // build the program
    err = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);

    // 4) Create kernel
    cl_kernel kernel = clCreateKernel(program, "game_of_life", &err);

    // 5) Set kernel args
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufOld);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufNew);
    err = clSetKernelArg(kernel, 2, sizeof(int), &width);
    err = clSetKernelArg(kernel, 3, sizeof(int), &height);

    // 6) Enqueue kernel (2D)
    size_t globalSize[2] = { (size_t)width, (size_t)height };
    err = clEnqueueNDRangeKernel(
              queue,
              kernel,
              2,     // 2D NDRange
              nullptr,
              globalSize,
              nullptr,  // local size
              0,
              nullptr,
              nullptr);

    // 7) Read back new grid if we want to see the result
    err = clEnqueueReadBuffer(queue, bufNew, CL_TRUE, 0,
                              totalSize*sizeof(char),
                              hostNewGrid.data(),
                              0, nullptr, nullptr);

    // 8) Cleanup
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(bufOld);
    clReleaseMemObject(bufNew);
    // plus release context, queue, etc.

    // Now hostNewGrid has the next generation
    // We could show it or do more passes

    std::cout << "Done computing next generation." << std::endl;
    return 0;
}