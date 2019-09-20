#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <chrono>

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
#define FILTER_SIZE (3)
#define BOX1_IMG_SIZE (IMG_SIZE)
#define BOX2_IMG_SIZE (IMG_SIZE/2)
#define BOX3_IMG_SIZE (IMG_SIZE/4)
#define BOX4_IMG_SIZE (IMG_SIZE/8)

#define MAX_SOURCE_SIZE (0x10000)

int main(int argc, char** argv)
{
    int err;                            // error code returned from api calls

    cl_mem conv_kernels_d[16];
    float* conv_kernels_h[16];
    const unsigned int conv_ch[16] = {IMG_CH, 8, 32, 32, 32, 64, 128, 128, 256, 64, 128, 32, 64, 32, 32, 16};
    const unsigned int conv_sz[16] = {1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1};

    for(int i = 0; i < 16; i++){
        int buf_size = sizeof(float) * conv_sz[i] * conv_sz[i] * conv_ch[i];
        conv_kernels_h[i] = new float[buf_size]();
        // TODO: code to load weights
    }
    
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
    cl_kernel kernel = clCreateKernel(program, "conv2d", &err);
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
    cl_mem input_d  = clCreateBuffer(context, CL_MEM_READ_ONLY,   sizeof(float) * IMG_SIZE * IMG_SIZE * IMG_CH, NULL, NULL);
    cl_mem layer_buf= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float) * IMG_SIZE * IMG_SIZE * 32,     NULL, NULL);
    cl_mem box1_out = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float) * BOX1_IMG_SIZE * BOX1_IMG_SIZE * 32,  NULL, NULL);
    cl_mem box2_out = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float) * BOX2_IMG_SIZE * BOX2_IMG_SIZE * 64,  NULL, NULL);
    cl_mem box3_out = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float) * BOX3_IMG_SIZE * BOX3_IMG_SIZE * 128, NULL, NULL);
    cl_mem output_d = clCreateBuffer(context, CL_MEM_WRITE_ONLY,  sizeof(float) * IMG_SIZE * IMG_SIZE, NULL, NULL);

    if (!input_d || !layer_buf || !box1_out || !box2_out || !box3_out || !output_d){
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }    

    // allocate device memory for model weights and copy weights to device
    for(int i = 0; i < 16; i++){
        int buf_size = sizeof(float) * conv_sz[i] * conv_sz[i] * conv_ch[i];
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

    // start computing

    // Read back the results from the device 
    err = clEnqueueReadBuffer(commands, output_d, CL_TRUE, 0, sizeof(float) * IMG_SIZE * IMG_SIZE, results_h, 0, NULL, NULL );  
    if (err != CL_SUCCESS){
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }


    for(int i = 0; i < 16; i++){
        clReleaseMemObject(conv_kernels_d[i]);
        delete[] conv_kernels_h[i];
    }
    clReleaseMemObject(input_d);
    clReleaseMemObject(layer_buf);
    clReleaseMemObject(box1_out);
    clReleaseMemObject(box2_out);
    clReleaseMemObject(box3_out);

    clReleaseProgram(program);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    delete[] results_h;
}