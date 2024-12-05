#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#else
    #include <CL/cl.h>
#endif

int main() {
    cl_int err;
    cl_uint num_platforms;

    // Get the number of OpenCL platforms available
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS) {
        printf("Failed to get platform IDs\n");
        return -1;
    }

    printf("Number of platforms: %u\n", num_platforms);

    // Get the platform IDs
    cl_platform_id *platforms = (cl_platform_id *)malloc(num_platforms * sizeof(cl_platform_id));
    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (err != CL_SUCCESS) {
        printf("Failed to get platform IDs\n");
        free(platforms);
        return -1;
    }

    // Print platform information
    for (cl_uint i = 0; i < num_platforms; ++i) {
        char platform_name[128];
        err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
        if (err != CL_SUCCESS) {
            printf("Failed to get platform name\n");
            free(platforms);
            return -1;
        }
        printf("Platform %u: %s\n", i, platform_name);

        // Get devices for this platform
        cl_uint num_devices;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        if (err != CL_SUCCESS) {
            printf("Failed to get device IDs\n");
            continue;
        }

        printf("\tNumber of devices: %u\n", num_devices);

        // Get device IDs
        cl_device_id *devices = (cl_device_id *)malloc(num_devices * sizeof(cl_device_id));
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
        if (err != CL_SUCCESS) {
            printf("Failed to get device IDs\n");
            free(devices);
            continue;
        }

        // Print device names
        for (cl_uint j = 0; j < num_devices; ++j) {
            char device_name[128];
            err = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
            if (err != CL_SUCCESS) {
                printf("Failed to get device name\n");
                continue;
            }
            printf("\tDevice %u: %s\n", j, device_name);
        }

        free(devices);
    }

    free(platforms);
    return 0;
}