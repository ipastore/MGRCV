__kernel void sobel_filter(
    __read_only image2d_t input_image,     // Input image
    __write_only image2d_t output_image,  // Output image
    sampler_t sampler
    // ,
    // __global float *debug_buffer
    ) {                  // Sampler for reading input image
    // // Get global ID
    int x = get_global_id(0);
    int y = get_global_id(1);

    // DEBUG
    // int global_id = y * get_global_size(0) + x;


    // Sobel Kernels
    const int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const int Gy[3][3] = {{ 1, 2, 1}, { 0, 0, 0}, {-1, -2, -1}};


    // Image size (assume fixed for now or pass as args)
    int2 size = (int2)(get_image_width(input_image), get_image_height(input_image));

    // Accumulators for gradients
    float gx = 0.0f;
    float gy = 0.0f;

        // Skip edges
    if (x == 0 || y == 0 || x == size.x - 1 || y == size.y - 1) {
        // Optionally write zero to output image for edge pixels
        write_imagef(output_image, (int2)(x, y), (float4)(0.0f, 0.0f, 0.0f, 1.0f));

        // DEBUG
        // debug_buffer[global_id * 4 + 0] = 0.0f; // gx
        // debug_buffer[global_id * 4 + 1] = 0.0f; // gy
        // debug_buffer[global_id * 4 + 2] = 0.0f; // gradient
        // debug_buffer[global_id * 4 + 3] = 0.0f; // normalized gradient

        return;
    }

    // Apply convolution
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            int nx = x + dx;
            int ny = y + dy;

            // Check bounds
            if (nx >= 0 && nx < size.x && ny >= 0 && ny < size.y) {
                float pixel = read_imagef(input_image, sampler, (int2)(nx, ny)).x;
                int kernel_x = dx + 1;
                int kernel_y = dy + 1;
                gx += pixel * Gx[kernel_x][kernel_y];
                gy += pixel * Gy[kernel_x][kernel_y];
            }
        }
    }
    // DEBUG
    // debug_buffer[global_id * 4 + 0] = gx; // Save gx
    // debug_buffer[global_id * 4 + 1] = gy; // Save gy

    // Compute gradient magnitude
    float gradient = sqrt(gx * gx + gy * gy);

    // DEBUG
    // debug_buffer[global_id * 4 + 2] = gradient; // Save final gradient


    // // Normalize to [0, 1] range
    gradient = clamp(gradient, 0.0f, 1.0f);

    // DEBUG  
    // debug_buffer[global_id * 4 + 3 ]= gradient; // Save gradient after normalization

    // Write the result to output image
    write_imagef(output_image, (int2)(x, y), (float4)(gradient, gradient, gradient, 1.0f));
}