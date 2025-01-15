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
    FILE *fileHandler = fopen("Kernel.cl", "r");
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

    cl_kernel kernel = clCreateKernel(program, "functionNameKernel", &err);
    cl_error(err, "Failed to create kernel");

    // 5. Load Input parameters / data

    // Creating 1D input 

    // 6. Create OpenCL buffers for each channel
    cl_mem buffer_input_vector = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, data_size, input, &err);
    cl_error(err, "Error: Failed to create input buffer for red channel");
    cl_mem buffer_output_vector = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size, NULL, &err);
    cl_error(err, "Error: Failed to create output buffer for red channel");

    // 7. Set kernel arguments and launch for each color channel
    size_t global_work_size[3] = { (size_t)width, (size_t)height,  (size_t)channels};
    size_t local_work_size[3] = {16, 16, channels}; // Test with different sizes

    
    // Adjust global work size for divisibility
    if (global_work_size[0] % local_work_size[0] != 0) {
        global_work_size[0] = ((global_work_size[0] / local_work_size[0]) + 1) * local_work_size[0];
    }
    if (global_work_size[1] % local_work_size[1] != 0) {
        global_work_size[1] = ((global_work_size[1] / local_work_size[1]) + 1) * local_work_size[1];
    }

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_input_vector);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_output_vector);
    clSetKernelArg(kernel, 2, sizeof(int), &width);
    clSetKernelArg(kernel, 3, sizeof(int), &height);
    clSetKernelArg(kernel, 4, sizeof(int), &channels);
    clSetKernelArg(kernel, 5, sizeof(float), &sigma);

    cl_event kernel_event, green_event, blue_event;


    cl_event write_event, read_event;

    // Measure host-to-device transfer
    clEnqueueWriteBuffer(command_queue, buffer_input_vector, CL_TRUE, 0, data_size, input, 0, NULL, &write_event);

    // Waiting for the events, if not done its gonna follow without all the data
    clWaitForEvents(1, &write_event);

    cl_ulong start, end;
    clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
 
    err = clEnqueueNDRangeKernel(command_queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &kernel_event);
    cl_error(err, "Error: Failed to launch kernel");

    // Waiting for the events 
    clWaitForEvents(1, &kernel_event);

    // clFinish(command_queue);    
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);

    clFinish(command_queue);

    // 8. Read the output buffer for each channel
    clEnqueueReadBuffer(command_queue, buffer_output_vector, CL_TRUE, 0, data_size, output, 0, NULL, &read_event);   
    clWaitForEvents(1, &read_event);


    // Measure device-to-host transfer
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);


    // 9. Combine the channels into the output image and display it
    
    // Cleanup
    delete[] input;
    delete[] output;

    clReleaseMemObject(buffer_input_vector);
    clReleaseMemObject(buffer_output_vector);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

}