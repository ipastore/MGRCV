__constant float gaussian_kernel[25] = {
    1/273.0f,  4/273.0f,  7/273.0f,  4/273.0f, 1/273.0f,
    4/273.0f, 16/273.0f, 26/273.0f, 16/273.0f, 4/273.0f,
    7/273.0f, 26/273.0f, 41/273.0f, 26/273.0f, 7/273.0f,
    4/273.0f, 16/273.0f, 26/273.0f, 16/273.0f, 4/273.0f,
    1/273.0f,  4/273.0f,  7/273.0f,  4/273.0f, 1/273.0f
};

__kernel void gaussian_filter(__read_only image2d_t input,
                              __write_only image2d_t output,
                              sampler_t sampler) {
    int2 coords = (int2)(get_global_id(0), get_global_id(1));
    float4 pixel = (float4)(0.0f);

    // Apply Gaussian Kernel
    int index = 0;
    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            float4 neighbor = read_imagef(input, sampler, coords + (int2)(dx, dy));
            pixel += neighbor * gaussian_kernel[index++];
        }
    }

    // Write the result back
    write_imagef(output, coords, pixel);
}