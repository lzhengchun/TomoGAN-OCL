#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <chrono>
#include "../utils.hpp"
#ifdef __APPLE__
    #define CL_SILENCE_DEPRECATION 
    #include <OpenCL/opencl.h>
#else
    #define CL_TARGET_OPENCL_VERSION 210
    #include <CL/cl.h>
#endif

using namespace std;

// Use a static data size for simplicity
#define IMG_SIZE    (512)
#define IMG_WIDTH   IMG_SIZE
#define IMG_HEIGHT  IMG_SIZE
#define IMG_CH1     (64)
#define IMG_CH2     (64)
#define IMG_CH_OUT  (IMG_CH1 + IMG_CH2)
#define IMG1_SIZE   (IMG_SIZE * IMG_SIZE * IMG_CH1)
#define IMG2_SIZE   (IMG_SIZE * IMG_SIZE * IMG_CH2)
#define OUTPUT_DATA_SIZE (IMG1_SIZE + IMG2_SIZE)
#define MAX_SOURCE_SIZE (0x10000)

int main(int argc, char** argv){
    int err;                            // error code returned from api calls
    float *input_img1_h = new float[IMG1_SIZE]();   // original data set given to device
    float *input_img2_h = new float[IMG2_SIZE]();
    float *results_h    = new float[OUTPUT_DATA_SIZE]();  // results returned from device

    unsigned int img_width   = IMG_SIZE;
    unsigned int img_height  = IMG_SIZE;
    unsigned int img_channel1 = IMG_CH1;
    unsigned int img_channel2 = IMG_CH2;
    
    float inc_val = 0;
    for(int h = 0; h < IMG_SIZE; h++)
        for(int w = 0; w < IMG_SIZE; w++)
            for(int c = 0; c < IMG_CH1; c++){
                unsigned int gidx = IMG_CH1 * IMG_WIDTH * h + IMG_CH1 * w + c;
                input_img1_h[gidx] = inc_val;
                inc_val += 1;
    }

    // inc_val = 513;
    for(int h = 0; h < IMG_SIZE; h++)
        for(int w = 0; w < IMG_SIZE; w++)
            for(int c = 0; c < IMG_CH2; c++){
                unsigned int gidx = IMG_CH2 * IMG_WIDTH * h + IMG_CH2 * w + c;
                input_img2_h[gidx] = inc_val;
                inc_val += 1;
    }
    
    auto start_cpu = chrono::steady_clock::now();
    concatenate(input_img1_h, input_img2_h, img_height, img_width, img_channel1, img_channel2, results_h);
    auto end_cpu = chrono::steady_clock::now();
    printf("It takes %.3f ms to concatenate using CPU (exclude data transfer)\n", \
           chrono::duration_cast<chrono::microseconds>(end_cpu - start_cpu).count()/1000.);

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
 
    fp = fopen("../conv2d.cl", "r");
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
    cl_kernel kernel = clCreateKernel(program, "concatenate_vec16", &err);
    if (!kernel || err != CL_SUCCESS){
        printf("Error: Failed to create compute kernel! %d\n", err);
        exit(1);
    }

    size_t max_group_size;
    // Get the maximum work group size for executing the kernel on the device
    err = clGetKernelWorkGroupInfo(kernel, device_ids[dev_idx], CL_KERNEL_WORK_GROUP_SIZE, sizeof(max_group_size), &max_group_size, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }else{
        printf("Max kernel work group size: %ld\n", max_group_size);
    }

    auto compile_ed = chrono::steady_clock::now();
    printf("It takes %.3f ms to compile OCL kernel\n", \
           chrono::duration_cast<chrono::microseconds>(compile_ed - compile_st).count()/1000.);

    // Create the input and output arrays in device memory for our calculation
    cl_mem in_img1_d = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float) * IMG1_SIZE, NULL, NULL);
    cl_mem in_img2_d = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float) * IMG2_SIZE, NULL, NULL);
    cl_mem out_img_d = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * OUTPUT_DATA_SIZE, NULL, NULL);

    if (!in_img1_d || !in_img1_d || !out_img_d){
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }    

    // Write our data set into the input array in device memory 
    err  = clEnqueueWriteBuffer(commands, in_img1_d,  CL_TRUE, 0, sizeof(float) * IMG1_SIZE, input_img1_h, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commands, in_img2_d,  CL_TRUE, 0, sizeof(float) * IMG2_SIZE, input_img2_h, 0, NULL, NULL);
    if (err != CL_SUCCESS){
        printf("Error: Failed to write to device array!\n");
        exit(1);
    }

    const unsigned int n_reps = 105;
    auto start = chrono::steady_clock::now();
    for (size_t rep_idx = 0; rep_idx < n_reps; rep_idx++){
        // ignore as warmup
        if(rep_idx==5){
            start = chrono::steady_clock::now();
        }
        // Set the arguments to our compute kernel
        err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_img1_d);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &in_img2_d);
        err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &img_height);
        err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &img_width);
        err |= clSetKernelArg(kernel, 4, sizeof(unsigned int), &img_channel1);
        err |= clSetKernelArg(kernel, 5, sizeof(unsigned int), &img_channel2);
        err |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &out_img_d);
        if (err != CL_SUCCESS){
            printf("Error: Failed to set kernel arguments! %d\n", err);
            exit(1);
        }

        size_t local[2] = {16, 16};
        size_t global[2] = {IMG_SIZE, IMG_SIZE};
        err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, global, local, 0, NULL, NULL);
        if (err){
            printf("Error: Failed to execute kernel! %d\n", err);
            return EXIT_FAILURE;
        }
        clFinish(commands);
    }
    
    // Read back the results from the device 
    err = clEnqueueReadBuffer(commands, out_img_d, CL_TRUE, 0, sizeof(float) * OUTPUT_DATA_SIZE, results_h, 0, NULL, NULL );  
    if (err != CL_SUCCESS){
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }

    auto end = chrono::steady_clock::now();
    printf("It takes %.3f ms to compute using GPU\n", \
           chrono::duration_cast<chrono::microseconds>(end - start).count()/1000./(n_reps-5));
    // release memory obj
    clReleaseMemObject(in_img1_d);
    clReleaseMemObject(in_img2_d);
    clReleaseMemObject(out_img_d);
    clReleaseProgram(program);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    if(IMG_SIZE > 32){
        for (size_t i = 0; i < 10; i++){
            printf("%4.0f ", results_h[222*IMG_SIZE + i]);
        }
        printf("\n");
    }else{
        for(size_t ch = 0; ch < IMG_CH_OUT; ch++){
            printf("Channel %ld\n", ch + 1);
            for (size_t r = 0; r < IMG_SIZE; r++){
                for (size_t c = 0; c < IMG_SIZE; c++){
                    printf("%4.0f ", results_h[IMG_CH_OUT * IMG_SIZE * r + IMG_CH_OUT * c + ch]);
                }
                printf("\n");
            }
        }
    }
    
    // for(size_t ch = 0; ch < IMG_CH2; ch++){
    //     printf("Channel %ld\n", ch + 1);
    //     for (size_t r = 0; r < IMG_SIZE; r++){
    //         for (size_t c = 0; c < IMG_SIZE; c++){
    //             printf("%4.0f ", input_img2_h[IMG_CH2 * IMG_SIZE * r + IMG_CH2 * c + ch]);
    //         }
    //         printf("\n");
    //     }
    // }

    double res_sum = 0;
    for (size_t i = 0; i < OUTPUT_DATA_SIZE; i++){
        res_sum += results_h[i];
    }
    printf("sum of output: %lf\n", res_sum);

    delete[] results_h;
    delete[] input_img1_h;
    delete[] input_img2_h;
}