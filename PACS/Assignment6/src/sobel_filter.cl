__kernel void sobel_filter(
    __read_only image2d_t input_image,      // Input image
    __write_only image2d_t output_image,   // Output image
    sampler_t sampler,                     // Sampler
    __local float* local_block             // Shared memory for local computations
) {
    // Global and local IDs
    int global_x = get_global_id(0);
    int global_y = get_global_id(1);
    int local_x = get_local_id(0);
    int local_y = get_local_id(1);

    // Workgroup size
    int local_size_x = get_local_size(0);
    int local_size_y = get_local_size(1);

    // Image size
    int2 size = (int2)(get_image_width(input_image), get_image_height(input_image));

    // Shared memory dimensions (include halo for convolution)
    int shared_width = local_size_x + 2; // Add 1-pixel halo on each side
    int shared_height = local_size_y + 2;

    // Load pixel into shared memory (including halo)
    int shared_x = local_x + 1;
    int shared_y = local_y + 1;

    // Initialize shared memory with zeros
    local_block[shared_y * shared_width + shared_x] = 0.0f;

    // Load main pixel
    local_block[shared_y * shared_width + shared_x] = 0.0f;
    if (global_x < size.x && global_y < size.y) {
        local_block[shared_y * shared_width + shared_x] = read_imagef(input_image, sampler, (int2)(global_x, global_y)).x;
    }

    // Load halo pixels
    // Left
    if (local_x == 0 && global_x > 0) {
        local_block[shared_y * shared_width] = read_imagef(input_image, sampler, (int2)(global_x - 1, global_y)).x;
    // Right
    } else if (local_x == local_size_x - 1 && global_x < size.x - 1) {
        local_block[shared_y * shared_width + shared_width - 1] =
            read_imagef(input_image, sampler, (int2)(global_x + 1, global_y)).x;
    }
    // Top
    if (local_y == 0 && global_y > 0) {
        local_block[shared_x] = read_imagef(input_image, sampler, (int2)(global_x, global_y - 1)).x;
    // Bottom
    } else if (local_y == local_size_y - 1 && global_y < size.y - 1) {
        local_block[(shared_height - 1) * shared_width + shared_x] =
            read_imagef(input_image, sampler, (int2)(global_x, global_y + 1)).x;
    }

    // Handle corners
    if (local_x == 0 && local_y == 0 && global_x > 0 && global_y > 0) {
        local_block[0] = read_imagef(input_image, sampler, (int2)(global_x - 1, global_y - 1)).x;
    } else if (local_x == 0 && local_y == local_size_y - 1 && global_x > 0 && global_y < size.y - 1) {
        local_block[(shared_height - 1) * shared_width] =
            read_imagef(input_image, sampler, (int2)(global_x - 1, global_y + 1)).x;
    } else if (local_x == local_size_x - 1 && local_y == 0 && global_x < size.x - 1 && global_y > 0) {
        local_block[shared_width - 1] = read_imagef(input_image, sampler, (int2)(global_x + 1, global_y - 1)).x;
    } else if (local_x == local_size_x - 1 && local_y == local_size_y - 1 && global_x < size.x - 1 && global_y < size.y - 1) {
        local_block[(shared_height - 1) * shared_width + shared_width - 1] =
            read_imagef(input_image, sampler, (int2)(global_x + 1, global_y + 1)).x;
    }

    // Barrier to ensure all threads have written their data
    barrier(CLK_LOCAL_MEM_FENCE);

    // Skip edge pixels
    if (global_x == 0 || global_y == 0 || global_x >= size.x - 1 || global_y >= size.y - 1) {
        write_imagef(output_image, (int2)(global_x, global_y), (float4)(0.0f, 0.0f, 0.0f, 1.0f));
        return;
    }

    // Sobel Kernels
    const int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const int Gy[3][3] = {{ 1,  2,  1}, { 0,  0,  0}, {-1, -2, -1}};

    // Compute Sobel gradient using shared memory
    float gx = 0.0f;
    float gy = 0.0f;

    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            int pixel_x = shared_x + dx;
            int pixel_y = shared_y + dy;
            float pixel = local_block[pixel_y * shared_width + pixel_x];

            gx += pixel * Gx[dy + 1][dx + 1];
            gy += pixel * Gy[dy + 1][dx + 1];
        }
    }

    // Compute gradient magnitude and normalize
    float gradient = sqrt(gx * gx + gy * gy);
    gradient = clamp(gradient, 0.0f, 1.0f);

    // Write to output image
    write_imagef(output_image, (int2)(global_x, global_y), (float4)(gradient, gradient, gradient, 1.0f));
}
