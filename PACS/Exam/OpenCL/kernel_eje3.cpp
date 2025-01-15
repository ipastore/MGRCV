__kernel void dot_product
        (__global const float *a,
        __global const float *b,
        __global float *c
        const int c_rows          // == a_rows
        const int c_cols          // == b_cols
        const int a_cols_b_rows
        ) {
    
    int row = get_global_id(0);

    if (row < c_rows) {
        int result = 0;
        for (int col = 0; col < ç_cols; col++) {
            for (int i = 0; i < a_cols_b_rows; i++) {
                result += a[row * a_cols_b_rows + i] * b[i * c_cols + col];
            }
            c[row * c_cols + col] = result;
        }
    }
}