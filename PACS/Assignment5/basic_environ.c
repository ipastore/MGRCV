////////////////////////////////////////////////////////////////////
//File: basic_environ.c
//
//Description: base file for environment exercises with openCL
//
// 
////////////////////////////////////////////////////////////////////

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#ifdef __APPLE__
  #include <OpenCL/opencl.h>
#else
  #include <CL/cl.h>
#endif

// ABOUT ERRORS
// Remmember to check error codes from every OpenCL API call
// Info about error codes can be found at the reference manual of OpenCL
// At the following url, you can find name for each error code
//  https://gist.github.com/bmount/4a7144ce801e5569a0b6
//  https://streamhpc.com/blog/2013-04-28/opencl-error-codes/
// Following function checks errors, and in such a case, it prints the code, the string and it exits

void cl_error(cl_int code, const char *string){
    if (code != CL_SUCCESS){
        printf("%d - %s\n", code, string);
        exit(-1);
    }
}
////////////////////////////////////////////////////////////////////////////////


/* ATTENTION: While prgramming in OpenCL it is a good idea to keep the reference manuals handy:
 * https://bashbaug.github.io/OpenCL-Docs/pdf/OpenCL_API.pdf
 * https://www.khronos.org/files/opencl-1-2-quick-reference-card.pdf (summary of OpenCL API)
 * https://www.khronos.org/assets/uploads/developers/presentations/opencl20-quick-reference-card.pdf
 */


int main(int argc, char** argv)
{
  int err;                              // error code returned from api calls
  size_t t_buf = 50;            // size of str_buffer
  char str_buffer[t_buf];       // auxiliary buffer 
  size_t e_buf;             // effective size of str_buffer in use
        
  size_t global_size;                       // global domain size for our calculation
  size_t local_size;                        // local domain size for our calculation

  const cl_uint num_platforms_ids = 10;             // max of allocatable platforms
  cl_platform_id platforms_ids[num_platforms_ids];      // array of platforms
  cl_uint n_platforms;                      // effective number of platforms in use
  const cl_uint num_devices_ids = 10;               // max of allocatable devices
  cl_device_id devices_ids[num_platforms_ids][num_devices_ids]; // array of devices
  cl_uint n_devices[num_platforms_ids];             // effective number of devices in use for each platform
    
  cl_device_id device_id;                           // compute device id 
  cl_context context;                               // compute context
  cl_command_queue command_queue;                   // compute command queue
    

  // 1. Scan the available platforms:
  err = clGetPlatformIDs (num_platforms_ids, platforms_ids, &n_platforms);
  cl_error(err, "Error: Failed to Scan for Platforms IDs");
  printf("Number of available platforms: %d\n\n", n_platforms);

  for (int i = 0; i < n_platforms; i++ ){
    err= clGetPlatformInfo(platforms_ids[i], CL_PLATFORM_NAME, t_buf, str_buffer, &e_buf);
    cl_error (err, "Error: Failed to get info of the platform\n");
    printf( "\t[%d]-Platform Name: %s\n", i, str_buffer);

    // ***Task***: print on the screen the name, host_timer_resolution, vendor, versionm, ...

    // Platform Name
    err = clGetPlatformInfo(platforms_ids[i], CL_PLATFORM_NAME, sizeof(str_buffer), str_buffer, NULL);
    cl_error(err, "clGetPlatformInfo: Getting platform name");
    printf("\t[%d]-Platform Name: %s\n", i, str_buffer);

    // Host timer resolution
    // CL_PLATFORM_HOST_TIMER_RESOLUTION is not defined for MACOS and OpenCL 1.2
    #ifndef CL_PLATFORM_HOST_TIMER_RESOLUTION
    #define CL_PLATFORM_HOST_TIMER_RESOLUTION 0x0905
    #endif
    size_t host_timer_resolution;
    err = clGetPlatformInfo(platforms_ids[i], CL_PLATFORM_HOST_TIMER_RESOLUTION, sizeof(host_timer_resolution), &host_timer_resolution, NULL);
    printf("\t[%d]-Platform Host Timer Resolution: %zu nanoseconds\n", i, host_timer_resolution);
    

    // Vendor
    err = clGetPlatformInfo(platforms_ids[i], CL_PLATFORM_VENDOR, sizeof(str_buffer), str_buffer, NULL);
    cl_error(err, "clGetPlatformInfo: Getting platform vendor");
    printf("\t[%d]-Platform Vendor: %s\n", i, str_buffer);

    // Version
    err = clGetPlatformInfo(platforms_ids[i], CL_PLATFORM_VERSION, sizeof(str_buffer), str_buffer, NULL);
    cl_error(err, "clGetPlatformInfo: Getting platform version");
    printf("\t[%d]-Platform Version: %s\n\n", i, str_buffer);
  }
  printf("\n");
    
  // 2. Scan for devices in each platform
  for (cl_uint i = 0; i < n_platforms; i++ ){
    err = clGetDeviceIDs( platforms_ids[i], CL_DEVICE_TYPE_ALL, num_devices_ids, devices_ids[i], &(n_devices[i]));
    cl_error(err, "Error: Failed to Scan for Devices IDs");
    printf("\t[%d]-Platform. Number of available devices: %d\n", i, n_devices[i]);

    for(cl_uint j = 0; j < n_devices[i]; j++){
      // Device name
      err = clGetDeviceInfo(devices_ids[i][j], CL_DEVICE_NAME, sizeof(str_buffer), &str_buffer, NULL);
      cl_error(err, "clGetDeviceInfo: Getting device name");
      printf("\t\t [%d]-Platform [%d]-Device CL_DEVICE_NAME: %s\n", i, j,str_buffer);

      // Max compute units available
      cl_uint max_compute_units_available;
      err = clGetDeviceInfo(devices_ids[i][j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_compute_units_available), &max_compute_units_available, NULL);
      cl_error(err, "clGetDeviceInfo: Getting device max compute units available");
      printf("\t\t [%d]-Platform [%d]-Device CL_DEVICE_MAX_COMPUTE_UNITS: %d\n\n", i, j, max_compute_units_available);
      
      // ***Task***: print on the screen the cache size, global mem size, local memsize, max work group size, profiling timer resolution and ... of each device

      // Cache size
      cl_ulong cache_size;
      err = clGetDeviceInfo(devices_ids[i][j], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cache_size), &cache_size, NULL);
      cl_error(err, "clGetDeviceInfo: Getting device cache size");
      printf("\t\t [%d]-Platform [%d]-Device CL_DEVICE_GLOBAL_MEM_CACHE_SIZE: %llu\n\n", i, j, cache_size);

      // Global mem size
      cl_ulong global_mem_size;
      err = clGetDeviceInfo(devices_ids[i][j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, NULL);
      cl_error(err, "clGetDeviceInfo: Getting device global mem size");
      printf("\t\t [%d]-Platform [%d]-Device CL_DEVICE_GLOBAL_MEM_SIZE: %llu\n\n", i, j, global_mem_size);

      // Local mem size
      cl_ulong local_mem_size;
      err = clGetDeviceInfo(devices_ids[i][j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem_size), &local_mem_size, NULL);
      cl_error(err, "clGetDeviceInfo: Getting device local mem size");
      printf("\t\t [%d]-Platform [%d]-Device CL_DEVICE_LOCAL_MEM_SIZE: %llu\n\n", i, j, local_mem_size);

      // Max work group size
      size_t max_work_group_size;
      err = clGetDeviceInfo(devices_ids[i][j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);
      cl_error(err, "clGetDeviceInfo: Getting device max work group size");
      printf("\t\t [%d]-Platform [%d]-Device CL_DEVICE_MAX_WORK_GROUP_SIZE: %lu\n\n", i, j, max_work_group_size);

      // Profiling timer resolution
      size_t profiling_timer_resolution;
      err = clGetDeviceInfo(devices_ids[i][j], CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(profiling_timer_resolution), &profiling_timer_resolution, NULL);
      cl_error(err, "clGetDeviceInfo: Getting device profiling timer resolution");
      printf("\t\t [%d]-Platform [%d]-Device CL_DEVICE_PROFILING_TIMER_RESOLUTION: %lu\n\n", i, j, profiling_timer_resolution);

    }
  } 

  // 3. Create a context, with a device
  device_id = devices_ids[0][0]; 
  cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platforms_ids[0], 0};
  context = clCreateContext(properties, 1, &device_id, NULL, NULL, &err);
  cl_error(err, "Failed to create a compute context\n");


  // 4. Create a command queue
  cl_command_queue_properties proprt[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
  // clCreateCommandQueueWithProperties is not supported for MACOS and OpenCL 1.2
  // command_queue = clCreateCommandQueueWithProperties( context, device_id, proprt, &err);
  command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
  cl_error(err, "Failed to create a command queue\n");


  /* It is still missing the runtime part of the OpenCL program: createBuffers, createProgram, createKernel, setKernelArg, ... */
  //2. Calculate size of the file
  FILE *fileHandler = fopen("../kernel.cl", "r");
  fseek(fileHandler, 0, SEEK_END);
  size_t fileSize = ftell(fileHandler);
  rewind(fileHandler);

  // read kernel source into buffer
  char * sourceCode = (char*) malloc(fileSize + 1);
  sourceCode[fileSize] = '\0';
  fread(sourceCode, sizeof(char), fileSize, fileHandler);
  fclose(fileHandler);

  // create program from buffer
  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&sourceCode, &fileSize, &err);
  cl_error(err, "Failed to create program with source\n");
  free(sourceCode);


  //3. Build the executable and check errors
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS){
    size_t len;
    char buffer[2048];

    printf("Error: Some error at building process.\n");
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
    printf("%s\n", buffer);
    exit(-1);
  }

  //4. Create a compute kernel with the program we want to run
  cl_kernel kernel = clCreateKernel(program, "pow_of_two", &err);
  cl_error(err, "Failed to create kernel from the program\n");


  // 5. host_memory
  size_t count = 10000;  // Number of elements in the array
  float *in_host_object = (float *)malloc(count * sizeof(float));
  float *out_host_object = (float *)malloc(count * sizeof(float));

  // Initialize input data
  for (size_t i = 0; i < count; i++) {
      in_host_object[i] = (float)i;
  }


  //6. Create OpenCL buffer visible to the OpenCl runtime
  cl_mem in_device_object  = clCreateBuffer(context, CL_MEM_READ_ONLY, count * sizeof(float), NULL, &err);
  cl_error(err, "Failed to create memory buffer at device\n");
  cl_mem out_device_object = clCreateBuffer(context, CL_MEM_WRITE_ONLY, count * sizeof(float), NULL, &err);
  cl_error(err, "Failed to create memory buffer at device\n");

  // 7. Write date into the memory object 
  err = clEnqueueWriteBuffer(command_queue, in_device_object, CL_TRUE, 0, sizeof(float) * count, in_host_object, 0, NULL, NULL);
  cl_error(err, "Failed to enqueue a write command\n");
  

  //8. Set the arguments to the kernel
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_device_object);
  cl_error(err, "Failed to set argument 0\n");
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_device_object);
  cl_error(err, "Failed to set argument 1\n");
  err = clSetKernelArg(kernel, 2, sizeof(size_t), &count);
  cl_error(err, "Failed to set argument 2\n");


  //9. Launch Kernel
  // local_size = 256; // Apple M1
  // global_size = count;
  // err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
  // cl_error(err, "Failed to launch kernel to the device\n");

  // Query max work group size
  size_t max_work_group_size;
  err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);
  cl_error(err, "Failed to query max work group size");
  printf("Max Work Group Size: %zu\n", max_work_group_size);

  // Set local and global sizes
  local_size = (max_work_group_size < count) ? max_work_group_size : count;
  global_size = ((count + local_size - 1) / local_size) * local_size;

  printf("Launching kernel with global_size=%zu, local_size=%zu\n", global_size, local_size);
  printf("Count: %zu\n", count);

  // Launch the kernel
  err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
  cl_error(err, "Failed to launch kernel");


  //10. Read data form device memory back to host memory
  err = clEnqueueReadBuffer(command_queue, out_device_object, CL_TRUE, 0, sizeof(float) * count, out_host_object, 0, NULL, NULL);
  cl_error(err, "Failed to enqueue a read command\n");

  // //11. Print results
  // for (size_t i = 0; i < count; i++) {
  //     printf("Input: %f -> Output: %f\n", in_host_object[i], out_host_object[i]);
  // }

  printf("Input: %f -> Output: %f\n", in_host_object[0], out_host_object[0]);
  printf("Input: %f -> Output: %f\n", in_host_object[1], out_host_object[1]);
  printf("Input: %f -> Output: %f\n", in_host_object[2], out_host_object[2]);
  printf("Input: %f -> Output: %f\n", in_host_object[count-1], out_host_object[count-1]);

  //12. Release all the OpenCL memory objects allocated along this program.
  clReleaseMemObject(in_device_object);
  clReleaseMemObject(out_device_object);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(command_queue);
  clReleaseContext(context);

  return 0;
}

