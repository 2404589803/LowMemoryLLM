#ifndef HAL_H
#define HAL_H

#include <stdint.h>
#include <stddef.h>

// 硬件抽象层接口定义
typedef struct {
    // 设备类型枚举
    enum {
        DEVICE_TYPE_CPU,
        DEVICE_TYPE_GPU,
        DEVICE_TYPE_TPU,
        DEVICE_TYPE_OTHER
    } device_type;
    
    // 设备能力描述
    struct {
        uint32_t compute_units;
        uint64_t memory_size;
        uint32_t max_threads;
    } capabilities;
    
    // 设备操作函数指针
    void* (*allocate_memory)(size_t size);
    void (*free_memory)(void* ptr);
    void (*memcpy_to_device)(void* dst, const void* src, size_t size);
    void (*memcpy_from_device)(void* dst, const void* src, size_t size);
    
    // 计算核心函数
    void (*matrix_multiply)(const void* a, const void* b, void* c, 
                          size_t m, size_t n, size_t k);
    void (*vector_add)(const void* a, const void* b, void* c, size_t size);
} HAL_Device;

// 初始化HAL系统
int hal_init(void);

// 获取可用设备列表
int hal_get_devices(HAL_Device** devices, int* num_devices);

// 选择最优设备
HAL_Device* hal_select_optimal_device(void);

#endif // HAL_H 