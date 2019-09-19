// HWC; stride = 1; padding = same; square filter
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

    if(row >= height || col >= width){
        return;
    }

    int in_g_row, in_g_col;
    unsigned int half_filter_size = filter_size/2;
    unsigned int gr2l_off = row - half_filter_size;
    unsigned int gc2l_off = col - half_filter_size;
    float conv_res = 0;
    for(unsigned int krow = 0; krow < filter_size; krow++)
        for(unsigned int kcol = 0; kcol < filter_size; kcol++)
            for(unsigned int kch = 0; kch < channel; kch++){
                in_g_row = gr2l_off + krow;
                in_g_col = gc2l_off + kcol;
                // zero padding
                if(in_g_row >= height || in_g_col >= width || in_g_row < 0 || in_g_col < 0){
                    continue;
                }
                conv_res += input[channel * width * in_g_row + channel * in_g_col + kch] * \
                            filter_values[channel * filter_size * krow + channel * kcol + kch];
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
    unsigned int ch_offset = 0, ch_to_16 = 1;
    int row = get_global_id(0);
    int col = get_global_id(1);
    unsigned int half_filter_size = filter_size/2;
    unsigned int gr2l_off = row - half_filter_size;
    unsigned int gc2l_off = col - half_filter_size;

    if(row >= height || col >= width){
        return;
    }

    int in_g_row, in_g_col;
    float16 conv_res = (float16)(0.0);
    for(unsigned int krow = 0; krow < filter_size; krow++)
        for(unsigned int kcol = 0; kcol < filter_size; kcol++){
                in_g_row = gr2l_off + krow;
                in_g_col = gc2l_off + kcol;
                // zero padding
                if(in_g_row >= height || in_g_col >= width || in_g_row < 0 || in_g_col < 0){
                    continue;
                }
                conv_res += input[ch_to_16*(in_g_row * width + in_g_col) + ch_offset] * \
                            filter_values[ch_to_16*(krow * filter_size + kcol) + ch_offset];
    }
    // output_buf[row * width + col] = dot(conv_res, (float16)(1));
    output_buf[row * width + col] = conv_res.s0 + conv_res.s1 + conv_res.s2 + conv_res.s3 + \
                                    conv_res.s4 + conv_res.s5 + conv_res.s6 + conv_res.s7 +\
                                    conv_res.s8 + conv_res.s9 + conv_res.sa + conv_res.sb +\
                                    conv_res.sc + conv_res.sd + conv_res.se + conv_res.sf;
}

// HWC; stride = 1; padding = same; square filter
// naive implementation using global memory
#define BLOCK_DIM 16
__kernel void conv2d_vec16_local(__global float16 *input,
                     const unsigned int height,
                     const unsigned int width,
                     const unsigned int dummy,
                     __constant float16 *filter_values,
                     const unsigned int filter_size,
                     __global float *output_buf){
    // only support kernel size of 3
    __local float16 input_local[BLOCK_DIM+2][BLOCK_DIM+2]; 
    if(filter_size != 3){
        return;
    }
    int lrow = get_local_id(0);
    int lcol = get_local_id(1);
    int row = get_global_id(0);
    int col = get_global_id(1); 
    if(row >= height || col >= width){
        return;
    }

    input_local[lrow+1][lcol+1] = input[width * row + col];

    // top line
    if(lrow==0){
        if(row == 0){
            input_local[0][lcol+1] = 0;
        }else{
            input_local[0][lcol+1] = input[width * (row-1) + col];
        }
    }
    // bottom line
    if(lrow==BLOCK_DIM-1){
        if(row >= height-1){
            input_local[BLOCK_DIM+1][lcol+1] = 0;
        }else{
            input_local[BLOCK_DIM+1][lcol+1] = input[width * (row+1) + col];
        }
    }
    // left line
    if(lcol==0){
        if(col==0){
            input_local[lrow+1][0] = 0;
        }else{
            input_local[lrow+1][0] = input[width * row + col-1];
        }
    }
    // right line
    if(lcol==BLOCK_DIM-1){
        if(col>=width-1){
            input_local[lrow+1][BLOCK_DIM+1] = 0;
        }else{
            input_local[lrow+1][BLOCK_DIM+1] = input[width * row + col+1];
        }
    }

    // top left corner-1, -1
    if(lrow==0 && lcol==0){
        if(row==0 || col==0){
            input_local[0][0] = 0;
        }else{
            input_local[0][0] = input[width * (row-1) + (col-1)];
        }
    }

    // top right corner -1, +1
    if(lrow==0 && lcol==BLOCK_DIM-1){
        if(col>=width-1 || row==0){
            input_local[0][BLOCK_DIM+1] = 0;
        }else{
            input_local[0][BLOCK_DIM+1] = input[width * (row-1) + (col+1)];;
        }
    }

    // bottom left corner +1, -1
    if(lrow==BLOCK_DIM-1 && lcol==0){
        if(row>=height-1 || col==0)
            input_local[BLOCK_DIM+1][0] = 0;
        else{
            input_local[BLOCK_DIM+1][0] = input[width * (row+1) + col-1];
        }
    }

    // bottom right corner, +1, +1
    if(lrow==BLOCK_DIM-1 && lcol==BLOCK_DIM-1){
        if(row>=height-1 || col>=width-1){
            input_local[BLOCK_DIM+1][BLOCK_DIM+1] = 0;
        }else{
            input_local[BLOCK_DIM+1][BLOCK_DIM+1] = input[width * (row+1) + col+1];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float16 conv_res = (float16)(0.0);
    conv_res += input_local[lrow][lcol]     * filter_values[0];
    conv_res += input_local[lrow][lcol+1]   * filter_values[1];
    conv_res += input_local[lrow][lcol+2]   * filter_values[2];

    conv_res += input_local[lrow+1][lcol]   * filter_values[3];
    conv_res += input_local[lrow+1][lcol+1] * filter_values[4];
    conv_res += input_local[lrow+1][lcol+2] * filter_values[5];

    conv_res += input_local[lrow+2][lcol]   * filter_values[6];
    conv_res += input_local[lrow+2][lcol+1] * filter_values[7];
    conv_res += input_local[lrow+2][lcol+2] * filter_values[8];

    // output_buf[row * width + col] = 33;
    // return;
    output_buf[row * width + col] = conv_res.s0 + conv_res.s1 + conv_res.s2 + conv_res.s3 + \
                                    conv_res.s4 + conv_res.s5 + conv_res.s6 + conv_res.s7 +\
                                    conv_res.s8 + conv_res.s9 + conv_res.sa + conv_res.sb +\
                                    conv_res.sc + conv_res.sd + conv_res.se + conv_res.sf;
}


__kernel void upsample2d(__global float   *input,
                        const unsigned int height,
                        const unsigned int width,
                        const unsigned int channel,
                        __global float    *output){

    int row = get_global_id(0);
    int col = get_global_id(1);   
    if(row >= height || col >= width){
        return;
    }
    unsigned int urow = 2 * row;
    unsigned int ucol = 2 * col;
    unsigned int uwidth = 2 * width;
    unsigned int uheight= 2 * height;
    for(unsigned int ch = 0; ch < channel; ch++){
        float pixel = input[width * row * channel + col * channel + ch];
        output[uwidth * urow     * channel + ucol     * channel + ch] = pixel;  // [urow][ucol][ch] 
        output[uwidth * (urow+1) * channel + ucol     * channel + ch] = pixel;  // [urow+1][ucol][ch] 
        output[uwidth * urow     * channel + (ucol+1) * channel + ch] = pixel;  // [urow][ucol+1][ch]
        output[uwidth * (urow+1) * channel + (ucol+1) * channel + ch] = pixel;  // [urow+1][ucol+1][ch]
    }
}

__kernel void maxpooling2d(__global float   *input,
                           const unsigned int height,
                           const unsigned int width,
                           const unsigned int channel,
                           __global float    *output){

    int row = get_global_id(0);
    int col = get_global_id(1);
    if(row >= height || col >= width){
        return;
    }
    unsigned int urow = 2 * row;
    unsigned int ucol = 2 * col;
    unsigned int uwidth = 2 * width;
    unsigned int uheight= 2 * height;
    for(unsigned int ch = 0; ch < channel; ch++){
        float pixel =       input[uwidth * urow     * channel + ucol     * channel + ch];  // [urow][ucol][ch] 
        pixel = fmax(pixel, input[uwidth * (urow+1) * channel + ucol     * channel + ch]); // [urow+1][ucol][ch] 
        pixel = fmax(pixel, input[uwidth * urow     * channel + (ucol+1) * channel + ch]); // [urow][ucol+1][ch] 
        pixel = fmax(pixel, input[uwidth * (urow+1) * channel + (ucol+1) * channel + ch]); // [urow+1][ucol+1][ch]
        output[width * row * channel + col * channel + ch] = pixel;
    }
}