
// CHW; stride = 1; padding = same; square filter
// naive implementation using global memory

__kernel void conv2d(__global float *input,
                     const unsigned int height,
                     const unsigned int width,
                     const unsigned int channel,
                     __constant float *filter_values,
                     const unsigned int filter_size,
                     __global float *output_buf){
    int row = get_global_id(0);
    int col = get_global_id(1);   

    if(row > height || col > width){
        return;
    }

    int in_g_row, in_g_col;
    float conv_res = 0;
    for(unsigned int kch = 0; kch < channel; kch++)
        for(unsigned int krow = 0; krow < filter_size; krow++)
            for(unsigned int kcol = 0; kcol < filter_size; kcol++){
                in_g_row = row - filter_size/2 + krow;
                in_g_col = col - filter_size/2 + kcol;
                // zero padding
                if(in_g_row >= height || in_g_col >= width || in_g_row < 0 || in_g_col < 0){
                    continue;
                }
                conv_res += input[width * height * kch + in_g_row * width + in_g_col] * \
                            filter_values[filter_size * filter_size * kch + filter_size * krow + kcol];
    }
    output_buf[row * width + col] = conv_res;
}

// HWC; stride = 1; padding = same; square filter
// naive implementation using global memory
__kernel void conv2d_vec16(__global float16 *input,
                     const unsigned int height,
                     const unsigned int width,
                     const unsigned int dummy,
                     __constant float16 *filter_values,
                     const unsigned int filter_size,
                     __global float *output_buf){
    int row = get_global_id(0);
    int col = get_global_id(1);   

    if(row > height || col > width){
        return;
    }

    int in_g_row, in_g_col;
    float16 conv_res = (float16)(0.0);
    for(unsigned int krow = 0; krow < filter_size; krow++)
        for(unsigned int kcol = 0; kcol < filter_size; kcol++){
                in_g_row = row - filter_size/2 + krow;
                in_g_col = col - filter_size/2 + kcol;
                // zero padding
                if(in_g_row >= height || in_g_col >= width || in_g_row < 0 || in_g_col < 0){
                    continue;
                }
                conv_res += input[in_g_row * width + in_g_col] * \
                            filter_values[krow * filter_size + kcol];
    }
    // output_buf[row * width + col] = dot(conv_res, (float16)(1));
    output_buf[row * width + col] = conv_res.s0 + conv_res.s1 + conv_res.s2 + conv_res.s3 + \
                                    conv_res.s4 + conv_res.s5 + conv_res.s6 + conv_res.s7 +\
                                    conv_res.s8 + conv_res.s9 + conv_res.sa + conv_res.sb +\
                                    conv_res.sc + conv_res.sd + conv_res.se + conv_res.sf;
}
