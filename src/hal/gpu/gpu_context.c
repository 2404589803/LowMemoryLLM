#include "gpu_context.h"
#include <stdlib.h>
#include <string.h>

// OpenCL内核源代码
static const char* kernel_source = R"(
__kernel void matrix_multiply(
    __global const float* a,
    __global const float* b,
    __global float* c,
    const int M, const int N, const int K
) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += a[row * K + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

__kernel void vector_add(
    __global const float* a,
    __global const float* b,
    __global float* c,
    const int size
) {
    int i = get_global_id(0);
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

__kernel void activation_relu(
    __global const float* input,
    __global float* output,
    const int size
) {
    int i = get_global_id(0);
    if (i < size) {
        output[i] = max(0.0f, input[i]);
    }
}
)";

// GPU设备初始化
int gpu_init(GPUContext** ctx) {
    cl_int err;
    *ctx = (GPUContext*)malloc(sizeof(GPUContext));
    if (!*ctx) return -1;
    memset(*ctx, 0, sizeof(GPUContext));
    
    // 获取平台
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) goto cleanup;
    
    // 获取GPU设备
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &(*ctx)->device_id, NULL);
    if (err != CL_SUCCESS) goto cleanup;
    
    // 创建上下文
    (*ctx)->context = clCreateContext(NULL, 1, &(*ctx)->device_id, NULL, NULL, &err);
    if (err != CL_SUCCESS) goto cleanup;
    
    // 创建命令队列
    (*ctx)->command_queue = clCreateCommandQueue((*ctx)->context, (*ctx)->device_id, 0, &err);
    if (err != CL_SUCCESS) goto cleanup;
    
    // 创建并编译程序
    (*ctx)->program = clCreateProgramWithSource((*ctx)->context, 1, &kernel_source, NULL, &err);
    if (err != CL_SUCCESS) goto cleanup;
    
    err = clBuildProgram((*ctx)->program, 1, &(*ctx)->device_id, NULL, NULL, NULL);
    if (err != CL_SUCCESS) goto cleanup;
    
    // 创建内核
    (*ctx)->matrix_multiply_kernel = clCreateKernel((*ctx)->program, "matrix_multiply", &err);
    if (err != CL_SUCCESS) goto cleanup;
    
    (*ctx)->vector_add_kernel = clCreateKernel((*ctx)->program, "vector_add", &err);
    if (err != CL_SUCCESS) goto cleanup;
    
    (*ctx)->activation_kernel = clCreateKernel((*ctx)->program, "activation_relu", &err);
    if (err != CL_SUCCESS) goto cleanup;
    
    // 获取设备内存信息
    cl_ulong total_memory;
    err = clGetDeviceInfo((*ctx)->device_id, CL_DEVICE_GLOBAL_MEM_SIZE,
                         sizeof(cl_ulong), &total_memory, NULL);
    if (err != CL_SUCCESS) goto cleanup;
    
    (*ctx)->total_memory = (size_t)total_memory;
    (*ctx)->used_memory = 0;
    
    return 0;
    
cleanup:
    gpu_cleanup(*ctx);
    *ctx = NULL;
    return -1;
}

// GPU设备清理
void gpu_cleanup(GPUContext* ctx) {
    if (!ctx) return;
    
    if (ctx->activation_kernel)
        clReleaseKernel(ctx->activation_kernel);
    if (ctx->vector_add_kernel)
        clReleaseKernel(ctx->vector_add_kernel);
    if (ctx->matrix_multiply_kernel)
        clReleaseKernel(ctx->matrix_multiply_kernel);
    if (ctx->program)
        clReleaseProgram(ctx->program);
    if (ctx->command_queue)
        clReleaseCommandQueue(ctx->command_queue);
    if (ctx->context)
        clReleaseContext(ctx->context);
    
    free(ctx);
}

// GPU内存分配
void* gpu_allocate_memory(GPUContext* ctx, size_t size) {
    if (!ctx || size == 0) return NULL;
    
    if (ctx->used_memory + size > ctx->total_memory) {
        return NULL;  // 内存不足
    }
    
    cl_int err;
    cl_mem buffer = clCreateBuffer(ctx->context, CL_MEM_READ_WRITE, size, NULL, &err);
    if (err != CL_SUCCESS) return NULL;
    
    ctx->used_memory += size;
    return (void*)buffer;
}

// GPU内存释放
void gpu_free_memory(GPUContext* ctx, void* ptr) {
    if (!ctx || !ptr) return;
    
    cl_mem buffer = (cl_mem)ptr;
    size_t size;
    cl_int err = clGetMemObjectInfo(buffer, CL_MEM_SIZE, sizeof(size_t), &size, NULL);
    if (err == CL_SUCCESS) {
        ctx->used_memory -= size;
    }
    
    clReleaseMemObject(buffer);
}

// GPU矩阵乘法
int gpu_matrix_multiply(GPUContext* ctx, const void* a, const void* b, void* c,
                       size_t m, size_t n, size_t k) {
    if (!ctx) return -1;
    
    cl_int err;
    size_t global_work_size[2] = {m, n};
    
    err = clSetKernelArg(ctx->matrix_multiply_kernel, 0, sizeof(cl_mem), &a);
    if (err != CL_SUCCESS) return -1;
    err = clSetKernelArg(ctx->matrix_multiply_kernel, 1, sizeof(cl_mem), &b);
    if (err != CL_SUCCESS) return -1;
    err = clSetKernelArg(ctx->matrix_multiply_kernel, 2, sizeof(cl_mem), &c);
    if (err != CL_SUCCESS) return -1;
    err = clSetKernelArg(ctx->matrix_multiply_kernel, 3, sizeof(int), &m);
    if (err != CL_SUCCESS) return -1;
    err = clSetKernelArg(ctx->matrix_multiply_kernel, 4, sizeof(int), &n);
    if (err != CL_SUCCESS) return -1;
    err = clSetKernelArg(ctx->matrix_multiply_kernel, 5, sizeof(int), &k);
    if (err != CL_SUCCESS) return -1;
    
    err = clEnqueueNDRangeKernel(ctx->command_queue, ctx->matrix_multiply_kernel,
                                2, NULL, global_work_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) return -1;
    
    return 0;
}

// GPU向量加法
int gpu_vector_add(GPUContext* ctx, const void* a, const void* b, void* c,
                   size_t size) {
    if (!ctx) return -1;
    
    cl_int err;
    size_t global_work_size = size;
    
    err = clSetKernelArg(ctx->vector_add_kernel, 0, sizeof(cl_mem), &a);
    if (err != CL_SUCCESS) return -1;
    err = clSetKernelArg(ctx->vector_add_kernel, 1, sizeof(cl_mem), &b);
    if (err != CL_SUCCESS) return -1;
    err = clSetKernelArg(ctx->vector_add_kernel, 2, sizeof(cl_mem), &c);
    if (err != CL_SUCCESS) return -1;
    err = clSetKernelArg(ctx->vector_add_kernel, 3, sizeof(int), &size);
    if (err != CL_SUCCESS) return -1;
    
    err = clEnqueueNDRangeKernel(ctx->command_queue, ctx->vector_add_kernel,
                                1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) return -1;
    
    return 0;
}

// GPU激活函数
int gpu_activation(GPUContext* ctx, void* output, const void* input,
                  size_t size, const char* type) {
    if (!ctx) return -1;
    
    // 目前只实现了ReLU激活函数
    if (strcmp(type, "relu") != 0) return -1;
    
    cl_int err;
    size_t global_work_size = size;
    
    err = clSetKernelArg(ctx->activation_kernel, 0, sizeof(cl_mem), &input);
    if (err != CL_SUCCESS) return -1;
    err = clSetKernelArg(ctx->activation_kernel, 1, sizeof(cl_mem), &output);
    if (err != CL_SUCCESS) return -1;
    err = clSetKernelArg(ctx->activation_kernel, 2, sizeof(int), &size);
    if (err != CL_SUCCESS) return -1;
    
    err = clEnqueueNDRangeKernel(ctx->command_queue, ctx->activation_kernel,
                                1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) return -1;
    
    return 0;
} 