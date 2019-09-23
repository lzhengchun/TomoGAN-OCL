#include <iostream>

#ifdef __APPLE__
    #define CL_SILENCE_DEPRECATION 
    #include <OpenCL/opencl.h>
#else
    #define CL_TARGET_OPENCL_VERSION 210
    #include <CL/cl.h>
#endif

void conv2d_set_arg(cl_kernel *kernel,
                    cl_mem *input_d,
                    unsigned int img_height,
                    unsigned int img_width,
                    unsigned int img_channel,
                    cl_mem *filter_d,
                    unsigned int filter_size,
                    unsigned int num_filter,
                    cl_mem *output_d,
                    unsigned char apply_relu){
    int err;
    err  = 0;
    err  = clSetKernelArg(*kernel, 0, sizeof(cl_mem), input_d);
    err |= clSetKernelArg(*kernel, 1, sizeof(unsigned int), &img_height);
    err |= clSetKernelArg(*kernel, 2, sizeof(unsigned int), &img_width);
    err |= clSetKernelArg(*kernel, 3, sizeof(unsigned int), &img_channel);
    err |= clSetKernelArg(*kernel, 4, sizeof(cl_mem), filter_d);
    err |= clSetKernelArg(*kernel, 5, sizeof(unsigned int), &filter_size);
    err |= clSetKernelArg(*kernel, 6, sizeof(unsigned int), &num_filter);
    err |= clSetKernelArg(*kernel, 7, sizeof(cl_mem), output_d);
    err |= clSetKernelArg(*kernel, 8, sizeof(unsigned char), &apply_relu);
    if (err != CL_SUCCESS){
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }
}