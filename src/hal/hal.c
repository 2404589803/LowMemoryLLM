#include "hal.h"
#include <stdlib.h>
#include <string.h>

// CPU设备实现声明
extern void matrix_multiply_asm(const float* a, const float* b, float* c, 
                              size_t m, size_t n, size_t k);
extern void vector_add_asm(const float* a, const float* b, float* c, size_t size);

// 静态设备列表
static HAL_Device* devices = NULL;
static int num_devices = 0;

// CPU设备内存管理实现
static void* cpu_allocate_memory(size_t size) {
    return aligned_alloc(32, size); // 32字节对齐以支持AVX/NEON
}

static void cpu_free_memory(void* ptr) {
    free(ptr);
}

static void cpu_memcpy_to_device(void* dst, const void* src, size_t size) {
    memcpy(dst, src, size);
}

static void cpu_memcpy_from_device(void* dst, const void* src, size_t size) {
    memcpy(dst, src, size);
}

// CPU设备计算实现包装
static void cpu_matrix_multiply(const void* a, const void* b, void* c,
                              size_t m, size_t n, size_t k) {
    matrix_multiply_asm((const float*)a, (const float*)b, (float*)c, m, n, k);
}

static void cpu_vector_add(const void* a, const void* b, void* c, size_t size) {
    vector_add_asm((const float*)a, (const float*)b, (float*)c, size);
}

// 初始化CPU设备
static HAL_Device* init_cpu_device(void) {
    HAL_Device* dev = (HAL_Device*)malloc(sizeof(HAL_Device));
    if (!dev) return NULL;
    
    dev->device_type = DEVICE_TYPE_CPU;
    
    // 获取CPU信息
#ifdef _WIN32
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    dev->capabilities.compute_units = sysInfo.dwNumberOfProcessors;
#else
    dev->capabilities.compute_units = sysconf(_SC_NPROCESSORS_ONLN);
#endif
    
    dev->capabilities.memory_size = SIZE_MAX; // 使用系统内存
    dev->capabilities.max_threads = dev->capabilities.compute_units * 2;
    
    // 设置函数指针
    dev->allocate_memory = cpu_allocate_memory;
    dev->free_memory = cpu_free_memory;
    dev->memcpy_to_device = cpu_memcpy_to_device;
    dev->memcpy_from_device = cpu_memcpy_from_device;
    dev->matrix_multiply = cpu_matrix_multiply;
    dev->vector_add = cpu_vector_add;
    
    return dev;
}

// HAL系统初始化
int hal_init(void) {
    // 初始化CPU设备
    HAL_Device* cpu_dev = init_cpu_device();
    if (!cpu_dev) return -1;
    
    devices = (HAL_Device**)malloc(sizeof(HAL_Device*));
    if (!devices) {
        free(cpu_dev);
        return -1;
    }
    
    devices[0] = cpu_dev;
    num_devices = 1;
    
    // TODO: 初始化其他设备（GPU等）
    
    return 0;
}

// 获取可用设备列表
int hal_get_devices(HAL_Device** out_devices, int* out_num_devices) {
    if (!devices || !out_devices || !out_num_devices) return -1;
    
    *out_devices = *devices;
    *out_num_devices = num_devices;
    return 0;
}

// 选择最优设备
HAL_Device* hal_select_optimal_device(void) {
    if (!devices || num_devices == 0) return NULL;
    
    // 简单策略：选择计算单元最多的设备
    HAL_Device* best_device = devices[0];
    uint32_t max_compute_units = devices[0]->capabilities.compute_units;
    
    for (int i = 1; i < num_devices; i++) {
        if (devices[i]->capabilities.compute_units > max_compute_units) {
            max_compute_units = devices[i]->capabilities.compute_units;
            best_device = devices[i];
        }
    }
    
    return best_device;
} 