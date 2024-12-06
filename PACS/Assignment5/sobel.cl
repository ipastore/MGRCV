__kernel void sobel_filter(
    __read_only image2d_t input_image,     // Input image
    __write_only image2d_t output_image,  // Output image
    sampler_t sampler) {                  // Sampler for reading input image
    // Get global ID
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Sobel Kernels
    const int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const int Gy[3][3] = {{ 1, 2, 1}, { 0, 0, 0}, {-1, -2, -1}};

    // Accumulators for gradients
    float gx = 0.0f;
    float gy = 0.0f;

    // Apply convolution
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            int kernel_x = dx + 1;
            int kernel_y = dy + 1;

            // Read pixel from input image
            float4 pixel = read_imagef(input_image, sampler, (int2)(x + dx, y + dy));

            // Convert RGB to grayscale (luminosity method)
            float gray = 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;

            // Apply Sobel kernels
            gx += gray * Gx[kernel_x][kernel_y];
            gy += gray * Gy[kernel_x][kernel_y];
        }
    }

    // Compute gradient magnitude
    float gradient = sqrt(gx * gx + gy * gy);

    // Normalize to [0, 1] range
    gradient = clamp(gradient / 255.0f, 0.0f, 1.0f);

    // Write the result to output image
    write_imagef(output_image, (int2)(x, y), (float4)(gradient, gradient, gradient, 1.0f));
}