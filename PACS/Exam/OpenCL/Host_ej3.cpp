int main(int argc, char** argv)
{
    int err;                             // error code returned from api calls
    size_t t_buf = 50;                   // size of str_buffer
    char str_buffer[t_buf];              // auxiliary buffer    
    size_t e_buf;                        // effective size of str_buffer in use
        
    const cl_uint num_platforms_ids = 10;                // max of allocatable platforms
    cl_platform_id platforms_ids[num_platforms_ids];     // array of platforms
    cl_uint n_platforms;                                 // effective number of platforms in use
    const cl_uint num_devices_ids = 10;                  // max of allocatable devices
    cl_device_id devices_ids[num_platforms_ids][num_devices_ids];    // array of devices
    cl_uint n_devices[num_platforms_ids];                // effective number of devices in use for each platform
    
    cl_device_id device_id;                              // compute device id 
    cl_context context;                                  // compute context
    cl_command_queue command_queue;                      // compute command queue


    int a_row = 5;
    int a_col = 4;
    int b_row = 4;
    int b_col = 5;

    A[a_row][a_col];
    B[b_row][b_col];
    if(a_col != b_row) return;
    C[a_col][b_row]; 
    

    // 1. Scan the available platforms
    err = clGetPlatformIDs(num_platforms_ids, platforms_ids, &n_platforms);
    cl_error(err, "Error: Failed to scan for platform IDs");
    printf("Number of available platforms: %d\n\n", n_platforms);

    // 2. Scan for devices in each platform
    for (int i = 0; i < n_platforms; i++ ){
        // Device type all can be GPU or CPU 
        err = clGetDeviceIDs( platforms_ids[i], CL_DEVICE_TYPE_ALL, num_devices_ids, devices_ids[i], &(n_devices[i]));
        cl_error(err, "Error: Failed to Scan for Devices IDs");
        printf("\t[%d]-Platform. Number of available devices: %d\n", i, n_devices[i]);
    }	

    // 3. Create context and command queue
    cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platforms_ids[0], 0};
    context = clCreateContext(properties, 1, devices_ids[0], NULL, NULL, &err);
    cl_error(err, "Failed to create a compute context");

    cl_command_queue_properties proprt[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    command_queue = clCreateCommandQueueWithProperties(context, devices_ids[0][0], proprt, &err);
    cl_error(err, "Failed to create a command queue");

    // 4. Load and build the kernel program
    FILE *fileHandler = fopen("Ejercicio3.cl", "r");
    fseek(fileHandler, 0, SEEK_END);
    size_t fileSize = ftell(fileHandler);
    rewind(fileHandler);

    char * sourceCode = (char*) malloc(fileSize + 1);
    sourceCode[fileSize] = '\0';
    fread(sourceCode, sizeof(char), fileSize, fileHandler);
    fclose(fileHandler);
    
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&sourceCode, &fileSize, &err);
    cl_error(err, "Error: Failed to create program with source");

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];
        printf("Error: Failed to build program\n");
        clGetProgramBuildInfo(program, devices_ids[0][0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(-1);
    }

    cl_kernel kernel = clCreateKernel(program, "Ejercicio3", &err);
    cl_error(err, "Failed to create kernel");

    // 5. Load Input parameters / data
    float *inputA = (float *)malloc(a_row * a_col * sizeof(float));
    float *inputB = (float *)malloc(b_row * b_col * sizeof(float));
    float *output = (float *)malloc(a_row * b_col * sizeof(float));

    // Flatear los datos
    for (int i = 0; i < a_row; i++) {
        for (int j = 0; j < a_col; j++) {
            inputA[i * a_col + j] = A[i][j];
        }
    }
    for (int i = 0; i < b_row; i++) {
        for (int j = 0; j < b_col; j++) {
            inputB[i * b_col + j] = B[i][j];
        }
    }

    // 6. Create OpenCL buffers for each channel
    cl_mem inputABuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, a_row * a_col * sizeof(float), NULL, &err);
    cl_error(err, "Error: Failed to create input buffer for A matrix");

    cl_mem inputBBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, b_row * b_col * sizeof(float), NULL, &err);
    cl_error(err, "Error: Failed to create input buffer for B matrix");

    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, a_row * b_col * sizeof(float), NULL, &err);
    cl_error(err, "Error: Failed to create output buffer for red channel");

    // Transfer data to device
    err = clEnqueueWriteBuffer(command_queue, inputABuffer, CL_TRUE, 0, a_row * a_col * sizeof(float), inputA, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, inputBBuffer, CL_TRUE, 0, b_row * b_col * sizeof(float), inputB, 0, NULL, NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputABuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &inputBBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputBuffer);
    clSetKernelArg(kernel, 3, sizeof(int), &a_row);
    clSetKernelArg(kernel, 4, sizeof(int), &b_col);
    clSetKernelArg(kernel, 4, sizeof(int), &a_col); // a_col == b_row

    // 7. Set kernel arguments and launch for the matrix
    size_t global_size[2] = { (size_t)a_row, (size_t)b_col};

    // Execute kernel
    cl_event kernel_event;

    err = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_size, NULL, 0, NULL, &kernel_event);
    cl_error(err, "Failed to execute kernel\n");
    // waiting for the event to finish
    clWaitForEvents(1, &kernel_event);

    // Read output
    err = clEnqueueReadBuffer(command_queue, outputBuffer, CL_TRUE, 0, a_row * b_col *  sizeof(float), output, 0, NULL, NULL);
    cl_error(err, "Failed to read output image\n");

    // Reconstruir la matriz de salida
    for (int i = 0; i < a_row; i++) {
        for (int j = 0; j < b_col; j++) {
            C[i][j] = output[i * b_col + j];
        }
    }

    // 8. Release OpenCL resources

    free(inputA);
    free(inputB);
    free(output);
    clReleaseMemObject(inputABuffer);
    clReleaseMemObject(inputBBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    return 0;

}