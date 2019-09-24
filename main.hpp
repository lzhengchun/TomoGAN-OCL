#include <iostream>

#ifdef __APPLE__
    #define CL_SILENCE_DEPRECATION 
    #include <OpenCL/opencl.h>
#else
    #define CL_TARGET_OPENCL_VERSION 210
    #include <CL/cl.h>
#endif

using namespace std;
#define oclErrchk(ans)  OCLAssert((ans), __FILE__, __LINE__) 
inline void OCLAssert(int code, string file, int line){
    if (code != CL_SUCCESS){
        cerr << "OpenCL Error: " << code << "; file: " << file << ", line:" << line << endl;
        exit(-1);
    }
}

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
        printf("Error: Failed to set kernel arguments for conv2d! %d\n", err);
        exit(1);
    }
    else{
        printf("Conv2d H:%4d, W:%4d, C:%3d, FS:%3d, NF:%3d, Relu:%1d\n", \
               img_height, img_width, img_channel, filter_size, num_filter, apply_relu);
    }
}


void maxpool_set_arg(cl_kernel *kernel,
                    cl_mem *input_d,
                    unsigned int img_height,
                    unsigned int img_width,
                    unsigned int img_channel,
                    cl_mem *output_d){
    int err;
    err  = 0;
    err  = clSetKernelArg(*kernel, 0, sizeof(cl_mem), input_d);
    err |= clSetKernelArg(*kernel, 1, sizeof(unsigned int), &img_height);
    err |= clSetKernelArg(*kernel, 2, sizeof(unsigned int), &img_width);
    err |= clSetKernelArg(*kernel, 3, sizeof(unsigned int), &img_channel);
    err |= clSetKernelArg(*kernel, 4, sizeof(cl_mem), output_d);
    if (err != CL_SUCCESS){
        printf("Error: Failed to set kernel arguments for max pooling! %d\n", err);
        exit(1);
    }
}

void upsample_set_arg(cl_kernel *kernel,
                    cl_mem *input_d,
                    unsigned int img_height,
                    unsigned int img_width,
                    unsigned int img_channel,
                    cl_mem *output_d){
    int err;
    err  = 0;
    err  = clSetKernelArg(*kernel, 0, sizeof(cl_mem), input_d);
    err |= clSetKernelArg(*kernel, 1, sizeof(unsigned int), &img_height);
    err |= clSetKernelArg(*kernel, 2, sizeof(unsigned int), &img_width);
    err |= clSetKernelArg(*kernel, 3, sizeof(unsigned int), &img_channel);
    err |= clSetKernelArg(*kernel, 4, sizeof(cl_mem), output_d);
    if (err != CL_SUCCESS){
        printf("Error: Failed to set kernel arguments for max pooling! %d\n", err);
        exit(1);
    }
}

void concat_set_arg(cl_kernel *kernel,
                    cl_mem *input_d1,
                    cl_mem *input_d2,
                    unsigned int img_height,
                    unsigned int img_width,
                    unsigned int img_channel1,
                    unsigned int img_channel2,
                    cl_mem *output_d){
    printf("Concat H:%4d, W:%4d, C1:%3d, C2:%3d\n", \
            img_height, img_width, img_channel1, img_channel2);
    int err;
    err  = 0;
    err  = clSetKernelArg(*kernel, 0, sizeof(cl_mem), input_d1);
    err |= clSetKernelArg(*kernel, 1, sizeof(cl_mem), input_d2);
    err |= clSetKernelArg(*kernel, 2, sizeof(unsigned int), &img_height);
    err |= clSetKernelArg(*kernel, 3, sizeof(unsigned int), &img_width);
    err |= clSetKernelArg(*kernel, 4, sizeof(unsigned int), &img_channel1);
    err |= clSetKernelArg(*kernel, 5, sizeof(unsigned int), &img_channel2);
    err |= clSetKernelArg(*kernel, 6, sizeof(cl_mem), output_d);
    if (err != CL_SUCCESS){
        printf("Error: Failed to set kernel arguments for concat! %d\n", err);
        exit(1);
    }
}

void swap_buf(float** a, float** b) {
    float* temp = *a;
    *a = *b;
    *b = temp;
}