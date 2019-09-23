#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <chrono>

#include "main.hpp"

#ifdef __APPLE__
    #define CL_SILENCE_DEPRECATION 
    #include <OpenCL/opencl.h>
#else
    #define CL_TARGET_OPENCL_VERSION 210
    #include <CL/cl.h>
#endif

using namespace std;

// Use a static data size for simplicity
#define IMG_SIZE    (1024)
#define IMG_WIDTH   IMG_SIZE
#define IMG_HEIGHT  IMG_SIZE
#define IMG_CH      (3)
#define INPUT_SIZE  (IMG_SIZE * IMG_SIZE * IMG_CH)
#define FILTER_SIZE (3)
#define BOX1_IMG_SIZE (IMG_SIZE)
#define BOX2_IMG_SIZE (IMG_SIZE/2)
#define BOX3_IMG_SIZE (IMG_SIZE/4)
#define INTR_IMG_SIZE (IMG_SIZE/8)

#define MAX_SOURCE_SIZE (0x10000)

int main(int argc, char** argv)
{
    int err;                            // error code returned from api calls
    float* input_h = new float[INPUT_SIZE]();
    cl_mem conv_kernels_d[16];
    float* conv_kernels_h[16];
    //                                   0    1   2   3   4   5    6    7    8    9   10   11  12  13  14  15
    const unsigned int conv_ch[16] = {IMG_CH, 8,  32, 32, 64,  64, 128, 128, 256, 64, 128, 32, 64, 32, 32, 16};
    const unsigned int  n_conv[16] = {8,      32, 32, 64, 64, 128, 128, 128,  64, 64, 32,  32, 32, 32, 16, 1};
    // const unsigned int conv_ch[16] = {IMG_CH, 8,  32, 32, 64, 64,  128, 128, 256, 128, 192, 64, 96, 32, 32, 16};
    // const unsigned int  n_conv[16] = {8,      32, 32, 64, 64, 128, 128, 128, 128, 128, 64,  64, 32, 32, 16, 1};
    const unsigned int conv_sz[16] = {1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1};

    std::ifstream fin("../../tomogan_weights_sterilize.bin", std::ios::binary);
    unsigned int total_params = 0;
    for(int i = 0; i < 16; i++){
        unsigned int n_weights = (conv_sz[i] * conv_sz[i] * conv_ch[i] * n_conv[i]);
        unsigned int buf_size = sizeof(float) * n_weights;
        total_params += n_weights;
        printf("%6d paras for conv2d_%02d kernel in_ch: %3d, no_ch: %3d\n", n_weights, i, conv_ch[i], n_conv[i]);
        conv_kernels_h[i] = new float[buf_size]();
        // load weights to host memory
        for (size_t inc_idx = 0; inc_idx < n_weights; inc_idx++){
            fin.read(reinterpret_cast<char*>(conv_kernels_h[i]+inc_idx), sizeof(float));
        }
    }
    printf("Total params: %d\n", total_params);
    
    float *results_h   = new float[IMG_SIZE*IMG_SIZE]();  // results returned from device

    // Set up platform 
    cl_platform_id platform_ids[2];
    cl_uint num_platforms = 0;
    err = clGetPlatformIDs(2, platform_ids, &num_platforms);
    if (err != CL_SUCCESS){ 
        printf("Failed to query platforms. Error:%i\n", err);
    }else{
        printf("There are %d platform(s).\n", num_platforms);
    }
    if(num_platforms == 0){
        printf("Exit because there is no platform support OpenCL\n");
        return EXIT_FAILURE;
    }

    unsigned int platform_idx = 0;
    char *profile = NULL;
    size_t size;
    clGetPlatformInfo(platform_ids[platform_idx], CL_PLATFORM_VERSION, 0, profile, &size); // get size of profile char array
    profile = new char[size]();
    clGetPlatformInfo(platform_ids[platform_idx], CL_PLATFORM_VERSION,size, profile, NULL); // get profile char array
    cout << profile << " is used to support computing." << endl;
    delete[] profile;

    unsigned int num_devices = 0;
    cl_device_id device_ids[2];
    err = clGetDeviceIDs(platform_ids[platform_idx], CL_DEVICE_TYPE_GPU, 2, device_ids, &num_devices);
    if (err != CL_SUCCESS){
        printf("Error: Failed to create a device group: %d\n", err);
        return EXIT_FAILURE;
    }else{
        printf("There is(are) %d GPU device(s) support OCL.\n", num_devices);
    }
    if(num_devices == 0){
        printf("Exit because there is no device support OpenCL\n");
        return EXIT_FAILURE;
    }

    unsigned int dev_idx = 0;

    cl_ulong local_mem_size;
    clGetDeviceInfo(device_ids[dev_idx], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem_size, 0);
    char *vendor = NULL;
    clGetDeviceInfo(device_ids[dev_idx], CL_DEVICE_VENDOR, 0, NULL, &size);
    vendor = new char[size]();
    clGetDeviceInfo(device_ids[dev_idx], CL_DEVICE_VENDOR, size, vendor, NULL);
    printf("Device produced by %s with %lldKB local mem will be used.\n", vendor, local_mem_size/1024);
    delete[] vendor;

    // Create a compute context 
    cl_context context = clCreateContext(0, 1, &(device_ids[dev_idx]), NULL, NULL, &err);
    if (!context){
        printf("Error: Failed to create a compute context! %d\n", err);
        return EXIT_FAILURE;
    }

    // Create a command commands
    #ifdef __APPLE__
        cl_command_queue commands = clCreateCommandQueue(context, device_ids[dev_idx], 0, &err);
    #else
        cl_command_queue commands = clCreateCommandQueueWithProperties(context, device_ids[dev_idx], 0, &err);
    #endif
    if (!commands){
        printf("Error: Failed to create a command commands! %d\n", err);
        return EXIT_FAILURE;
    }

    auto compile_st = chrono::steady_clock::now();
    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;
 
    fp = fopen("conv2d.cl", "r");
    if (!fp){
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = new char[MAX_SOURCE_SIZE]();
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );
    
    // Create the compute program from the source buffer
    cl_program program = clCreateProgramWithSource(context, 1, (const char **) &source_str, NULL, &err);
    if (!program){
        printf("Error: Failed to create compute program! %d\n", err);
        return EXIT_FAILURE;
    }
    delete[] source_str;

    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS){
        size_t len;
        char buffer[2048];
        printf("Error: Failed to build program executable!: %d\n", err);
        clGetProgramBuildInfo(program, device_ids[dev_idx], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("build error: %s\n", buffer);
        exit(1);
    }

    // Create the compute kernel in the program we wish to run
    cl_kernel kernel_conv2d = clCreateKernel(program, "conv2d_vec16_mk", &err);
    if (!kernel_conv2d || err != CL_SUCCESS){
        printf("Error: Failed to create conv2d-v16 kernel! %d\n", err);
        exit(1);
    }
    cl_kernel kernel_conv2d_v8 = clCreateKernel(program, "conv2d_vec8_mk", &err);
    if (!kernel_conv2d_v8 || err != CL_SUCCESS){
        printf("Error: Failed to create conv2d-v8 kernel! %d\n", err);
        exit(1);
    }
    cl_kernel kernel_pool = clCreateKernel(program, "maxpooling2d", &err);
    if (!kernel_pool || err != CL_SUCCESS){
        printf("Error: Failed to create maxpooling2d kernel! %d\n", err);
        exit(1);
    }
    cl_kernel kernel_concat = clCreateKernel(program, "concatenate", &err);
    if (!kernel_concat || err != CL_SUCCESS){
        printf("Error: Failed to create concat kernel! %d\n", err);
        exit(1);
    }
    cl_kernel kernel_upsample = clCreateKernel(program, "upsample2d", &err);
    if (!kernel_upsample || err != CL_SUCCESS){
        printf("Error: Failed to create upsample2d kernel! %d\n", err);
        exit(1);
    }

    auto compile_ed = chrono::steady_clock::now();
    printf("It takes %.3f ms to compile OCL kernel\n", \
           chrono::duration_cast<chrono::microseconds>(compile_ed - compile_st).count()/1000.);

    // Create the input and output arrays in device memory for our calculation
    cl_mem input_d    = clCreateBuffer(context, CL_MEM_READ_ONLY,   sizeof(float) * INPUT_SIZE, NULL, NULL);
    cl_mem layer_buf1 = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float) * IMG_SIZE * IMG_SIZE * 32,     NULL, NULL);
    cl_mem layer_buf2 = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float) * IMG_SIZE * IMG_SIZE * 32,     NULL, NULL);
    cl_mem box1_out   = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float) * BOX1_IMG_SIZE * BOX1_IMG_SIZE * 32,  NULL, NULL);
    cl_mem box2_out   = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float) * BOX2_IMG_SIZE * BOX2_IMG_SIZE * 64,  NULL, NULL);
    cl_mem box3_out   = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float) * BOX3_IMG_SIZE * BOX3_IMG_SIZE * 128, NULL, NULL);
    cl_mem output_d   = clCreateBuffer(context, CL_MEM_WRITE_ONLY,  sizeof(float) * IMG_SIZE * IMG_SIZE, NULL, NULL);

    if (!input_d || !layer_buf1 || !layer_buf2 || !box1_out || !box2_out || !box3_out || !output_d){
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }    

    // transfer input data to device 
    err = clEnqueueWriteBuffer(commands, input_d, CL_TRUE, 0, INPUT_SIZE, input_h, 0, NULL, NULL);  
    if (err != CL_SUCCESS){
        printf("Error occured when copy input data to device memory %d\n", err);
        exit(1);
    }

    // allocate device memory for model weights and copy weights to device
    auto weights_cp_st = chrono::steady_clock::now();
    for(int i = 0; i < 16; i++){
        unsigned int buf_size = sizeof(float) * (conv_sz[i] * conv_sz[i] * conv_ch[i] * n_conv[i]);
        conv_kernels_d[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, buf_size, NULL, NULL);
        if(!conv_kernels_d[i]){
            printf("Error: Failed to allocate device memory for kernel of layer %d!\n", i);
            exit(1);
        }
        err = clEnqueueWriteBuffer(commands, conv_kernels_d[i], CL_TRUE, 0, buf_size, conv_kernels_h[i], 0, NULL, NULL);  
        if (err != CL_SUCCESS){
            printf("Error occured when copy conv kernel %d! %d\n", i+1, err);
            exit(1);
        }
    }
    auto weights_cp_ed = chrono::steady_clock::now();
    printf("It takes %.3f ms to transfer weights from host to device!\n", \
           chrono::duration_cast<chrono::microseconds>(weights_cp_ed - weights_cp_st).count()/1000.);

    // start computing
    auto comp_st = chrono::steady_clock::now();
    size_t local[2] = {16, 16};
    size_t global[2] = {IMG_SIZE, IMG_SIZE};
    // conv layer 0
    // conv2d_set_arg(&kernel_conv2d, &input_d, BOX1_IMG_SIZE, BOX1_IMG_SIZE, conv_ch[0], &conv_kernels_d[0], 1, n_conv[0], &layer_buf1, 1);
    // err = clEnqueueNDRangeKernel(commands, kernel_conv2d, 2, NULL, global, local, 0, NULL, NULL);
    // oclErrchk(err);
    // clFinish(commands);

    // conv layer 1
    conv2d_set_arg(&kernel_conv2d_v8, &layer_buf1, BOX1_IMG_SIZE, BOX1_IMG_SIZE, conv_ch[1], &conv_kernels_d[1], 3, n_conv[1], &layer_buf2, 1);
    err = clEnqueueNDRangeKernel(commands, kernel_conv2d_v8, 2, NULL, global, local, 0, NULL, NULL);
    oclErrchk(err);
    clFinish(commands);

    // conv layer 2
    conv2d_set_arg(&kernel_conv2d, &layer_buf2, BOX1_IMG_SIZE, BOX1_IMG_SIZE, conv_ch[2], &conv_kernels_d[2], 3, n_conv[2], &box1_out, 1);
    err = clEnqueueNDRangeKernel(commands, kernel_conv2d, 2, NULL, global, local, 0, NULL, NULL);
    oclErrchk(err);
    clFinish(commands);

    // max pooling
    maxpool_set_arg(&kernel_pool, &box1_out, BOX1_IMG_SIZE, BOX1_IMG_SIZE, n_conv[2], &layer_buf1);
    err = clEnqueueNDRangeKernel(commands, kernel_pool, 2, NULL, global, local, 0, NULL, NULL);
    oclErrchk(err);
    clFinish(commands);

    // conv layer 3
    conv2d_set_arg(&kernel_conv2d, &layer_buf1, BOX2_IMG_SIZE, BOX2_IMG_SIZE, conv_ch[3], &conv_kernels_d[3], 3, n_conv[3], &layer_buf2, 1);
    err = clEnqueueNDRangeKernel(commands, kernel_conv2d, 2, NULL, global, local, 0, NULL, NULL);
    oclErrchk(err);
    clFinish(commands);

    // conv layer 4
    conv2d_set_arg(&kernel_conv2d, &layer_buf2, BOX2_IMG_SIZE, BOX2_IMG_SIZE, conv_ch[4], &conv_kernels_d[4], 3, n_conv[4], &box2_out, 1);
    err = clEnqueueNDRangeKernel(commands, kernel_conv2d, 2, NULL, global, local, 0, NULL, NULL);
    oclErrchk(err);
    clFinish(commands);

    // max pooling 1
    maxpool_set_arg(&kernel_pool, &box2_out, BOX2_IMG_SIZE, BOX2_IMG_SIZE, n_conv[4], &layer_buf1);
    err = clEnqueueNDRangeKernel(commands, kernel_pool, 2, NULL, global, local, 0, NULL, NULL);
    oclErrchk(err);
    clFinish(commands);

    // conv layer 5
    conv2d_set_arg(&kernel_conv2d, &layer_buf1, BOX3_IMG_SIZE, BOX3_IMG_SIZE, conv_ch[5], &conv_kernels_d[5], 3, n_conv[5], &layer_buf2, 1);
    err = clEnqueueNDRangeKernel(commands, kernel_conv2d, 2, NULL, global, local, 0, NULL, NULL);
    oclErrchk(err);
    clFinish(commands);

    // conv layer 6
    conv2d_set_arg(&kernel_conv2d, &layer_buf2, BOX3_IMG_SIZE, BOX3_IMG_SIZE, conv_ch[6], &conv_kernels_d[6], 3, n_conv[6], &box3_out, 1);
    err = clEnqueueNDRangeKernel(commands, kernel_conv2d, 2, NULL, global, local, 0, NULL, NULL);
    oclErrchk(err);
    clFinish(commands);

    // max pooling 2
    maxpool_set_arg(&kernel_pool, &box3_out, BOX3_IMG_SIZE, BOX3_IMG_SIZE, n_conv[6], &layer_buf1);
    err = clEnqueueNDRangeKernel(commands, kernel_pool, 2, NULL, global, local, 0, NULL, NULL);
    oclErrchk(err);
    clFinish(commands);

    // conv layer 7
    conv2d_set_arg(&kernel_conv2d, &layer_buf1, INTR_IMG_SIZE, INTR_IMG_SIZE, conv_ch[7], &conv_kernels_d[7], 3, n_conv[7], &layer_buf2, 1);
    err = clEnqueueNDRangeKernel(commands, kernel_conv2d, 2, NULL, global, local, 0, NULL, NULL);
    oclErrchk(err);
    clFinish(commands);

    // upsample 0
    maxpool_set_arg(&kernel_upsample, &layer_buf2, INTR_IMG_SIZE, INTR_IMG_SIZE, n_conv[7], &layer_buf1);
    err = clEnqueueNDRangeKernel(commands, kernel_upsample, 2, NULL, global, local, 0, NULL, NULL);
    oclErrchk(err);
    clFinish(commands);

    // concat 0
    concat_set_arg(&kernel_concat, &box3_out, &layer_buf1, BOX3_IMG_SIZE, BOX3_IMG_SIZE, n_conv[6], n_conv[7], &layer_buf2);
    err = clEnqueueNDRangeKernel(commands, kernel_concat, 2, NULL, global, local, 0, NULL, NULL);
    oclErrchk(err);
    clFinish(commands);

    // conv layer 8
    conv2d_set_arg(&kernel_conv2d, &layer_buf2, BOX3_IMG_SIZE, BOX3_IMG_SIZE, conv_ch[8], &conv_kernels_d[8], 3, n_conv[8], &layer_buf1, 1);
    err = clEnqueueNDRangeKernel(commands, kernel_conv2d, 2, NULL, global, local, 0, NULL, NULL);
    oclErrchk(err);
    clFinish(commands);

    // conv layer 9
    conv2d_set_arg(&kernel_conv2d, &layer_buf1, BOX3_IMG_SIZE, BOX3_IMG_SIZE, conv_ch[8], &conv_kernels_d[8], 3, n_conv[8], &layer_buf2, 1);
    err = clEnqueueNDRangeKernel(commands, kernel_conv2d, 2, NULL, global, local, 0, NULL, NULL);
    oclErrchk(err);
    clFinish(commands);

    // upsample 1
    maxpool_set_arg(&kernel_upsample, &layer_buf2, BOX3_IMG_SIZE, BOX3_IMG_SIZE, n_conv[9], &layer_buf1);
    err = clEnqueueNDRangeKernel(commands, kernel_upsample, 2, NULL, global, local, 0, NULL, NULL);
    oclErrchk(err);
    clFinish(commands);

    // concat 1
    concat_set_arg(&kernel_concat, &box2_out, &layer_buf1, BOX2_IMG_SIZE, BOX2_IMG_SIZE, n_conv[4], n_conv[9], &layer_buf2);
    err = clEnqueueNDRangeKernel(commands, kernel_concat, 2, NULL, global, local, 0, NULL, NULL);
    oclErrchk(err);
    clFinish(commands);

    // conv layer 10
    conv2d_set_arg(&kernel_conv2d, &layer_buf2, BOX2_IMG_SIZE, BOX2_IMG_SIZE, conv_ch[10], &conv_kernels_d[10], 3, n_conv[10], &layer_buf1, 1);
    err = clEnqueueNDRangeKernel(commands, kernel_conv2d, 2, NULL, global, local, 0, NULL, NULL);
    oclErrchk(err);
    clFinish(commands);

    // conv layer 11
    conv2d_set_arg(&kernel_conv2d, &layer_buf1, BOX2_IMG_SIZE, BOX2_IMG_SIZE, conv_ch[11], &conv_kernels_d[11], 3, n_conv[11], &layer_buf2, 1);
    err = clEnqueueNDRangeKernel(commands, kernel_conv2d, 2, NULL, global, local, 0, NULL, NULL);
    oclErrchk(err);
    clFinish(commands);


   // upsample 1
    maxpool_set_arg(&kernel_upsample, &layer_buf2, BOX2_IMG_SIZE, BOX2_IMG_SIZE, n_conv[11], &layer_buf1);
    err = clEnqueueNDRangeKernel(commands, kernel_upsample, 2, NULL, global, local, 0, NULL, NULL);
    oclErrchk(err);
    clFinish(commands);

    // concat 1
    concat_set_arg(&kernel_concat, &box1_out, &layer_buf1, BOX1_IMG_SIZE, BOX1_IMG_SIZE, n_conv[2], n_conv[11], &layer_buf2);
    err = clEnqueueNDRangeKernel(commands, kernel_concat, 2, NULL, global, local, 0, NULL, NULL);
    oclErrchk(err);
    clFinish(commands);

    // conv layer 12
    conv2d_set_arg(&kernel_conv2d, &layer_buf2, BOX1_IMG_SIZE, BOX1_IMG_SIZE, conv_ch[12], &conv_kernels_d[12], 3, n_conv[12], &layer_buf1, 1);
    err = clEnqueueNDRangeKernel(commands, kernel_conv2d, 2, NULL, global, local, 0, NULL, NULL);
    oclErrchk(err);
    clFinish(commands);

    // conv layer 13
    conv2d_set_arg(&kernel_conv2d, &layer_buf1, BOX1_IMG_SIZE, BOX1_IMG_SIZE, conv_ch[13], &conv_kernels_d[13], 3, n_conv[13], &layer_buf2, 1);
    err = clEnqueueNDRangeKernel(commands, kernel_conv2d, 2, NULL, global, local, 0, NULL, NULL);
    oclErrchk(err);
    clFinish(commands);

    // conv layer 14
    conv2d_set_arg(&kernel_conv2d, &layer_buf2, BOX1_IMG_SIZE, BOX1_IMG_SIZE, conv_ch[14], &conv_kernels_d[14], 1, n_conv[14], &layer_buf1, 1);
    err = clEnqueueNDRangeKernel(commands, kernel_conv2d, 2, NULL, global, local, 0, NULL, NULL);
    oclErrchk(err);
    clFinish(commands);

    // conv layer 15
    conv2d_set_arg(&kernel_conv2d, &layer_buf1, BOX1_IMG_SIZE, BOX1_IMG_SIZE, conv_ch[15], &conv_kernels_d[15], 1, n_conv[15], &layer_buf2, 0);
    err = clEnqueueNDRangeKernel(commands, kernel_conv2d, 2, NULL, global, local, 0, NULL, NULL);
    oclErrchk(err);
    clFinish(commands);

    auto comp_ed = chrono::steady_clock::now();
    printf("It takes %.3f ms to compute on device!\n", \
           chrono::duration_cast<chrono::microseconds>(comp_ed - comp_st).count()/1000.);

    // Read back the results from the device 
    err = clEnqueueReadBuffer(commands, output_d, CL_TRUE, 0, sizeof(float) * IMG_SIZE * IMG_SIZE, results_h, 0, NULL, NULL );  
    oclErrchk(err);

    for(int i = 0; i < 16; i++){
        clReleaseMemObject(conv_kernels_d[i]);
        delete[] conv_kernels_h[i];
    }
    clReleaseMemObject(input_d);
    clReleaseMemObject(layer_buf1);
    clReleaseMemObject(layer_buf2);
    clReleaseMemObject(box1_out);
    clReleaseMemObject(box2_out);
    clReleaseMemObject(box3_out);

    clReleaseProgram(program);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    delete[] results_h;
}