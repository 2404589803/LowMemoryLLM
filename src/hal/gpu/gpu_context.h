#ifndef GPU_CONTEXT_H
#define GPU_CONTEXT_H

#include <CL/cl.h>

// GPU上下文结构
typedef struct {
    cl_context context;           // OpenCL上下文
    cl_command_queue command_queue; // 命令队列
    cl_device_id device_id;       // GPU设备ID
    cl_program program;           // 编译后的OpenCL程序
    
    // 常用内核
    cl_kernel matrix_multiply_kernel;
    cl_kernel vector_add_kernel;
    cl_kernel activation_kernel;
    
    // 内存管理
    size_t total_memory;         // 总可用显存
    size_t used_memory;          // 已使用显存
} GPUContext;

// GPU设备初始化
int gpu_init(GPUContext** ctx);

// GPU设备清理
void gpu_cleanup(GPUContext* ctx);

// GPU内存分配
void* gpu_allocate_memory(GPUContext* ctx, size_t size);

// GPU内存释放
void gpu_free_memory(GPUContext* ctx, void* ptr);

// GPU计算函数
int gpu_matrix_multiply(GPUContext* ctx, const void* a, const void* b, void* c,
                       size_t m, size_t n, size_t k);
int gpu_vector_add(GPUContext* ctx, const void* a, const void* b, void* c,
                   size_t size);
int gpu_activation(GPUContext* ctx, void* output, const void* input,
                  size_t size, const char* type);

#endif // GPU_CONTEXT_H 