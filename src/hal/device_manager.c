#include "device_manager.h"
#include <stdlib.h>
#include <string.h>

// 全局设备管理器实例
static DeviceManager* g_device_manager = NULL;

// 初始化设备管理器
int device_manager_init(void) {
    if (g_device_manager) return 0; // 已经初始化
    
    g_device_manager = (DeviceManager*)malloc(sizeof(DeviceManager));
    if (!g_device_manager) return -1;
    
    g_device_manager->devices = NULL;
    g_device_manager->num_devices = 0;
    g_device_manager->current_device = NULL;
    
    // 初始化HAL系统
    if (hal_init() != 0) {
        free(g_device_manager);
        g_device_manager = NULL;
        return -1;
    }
    
    return device_manager_scan_devices();
}

// 扫描可用设备
int device_manager_scan_devices(void) {
    if (!g_device_manager) return -1;
    
    HAL_Device* devices;
    int num_devices;
    
    if (hal_get_devices(&devices, &num_devices) != 0) {
        return -1;
    }
    
    // 分配设备列表内存
    g_device_manager->devices = (HAL_Device**)malloc(sizeof(HAL_Device*) * num_devices);
    if (!g_device_manager->devices) return -1;
    
    // 复制设备列表
    for (int i = 0; i < num_devices; i++) {
        g_device_manager->devices[i] = &devices[i];
    }
    g_device_manager->num_devices = num_devices;
    
    // 设置默认设备
    if (num_devices > 0) {
        g_device_manager->current_device = hal_select_optimal_device();
    }
    
    return 0;
}

// 根据任务特征选择最优设备
HAL_Device* device_manager_select_device(const char* task_type, size_t memory_requirement) {
    if (!g_device_manager || !task_type) return NULL;
    
    HAL_Device* best_device = NULL;
    float best_score = -1.0f;
    
    for (int i = 0; i < g_device_manager->num_devices; i++) {
        HAL_Device* device = g_device_manager->devices[i];
        
        // 检查内存要求
        if (device->capabilities.memory_size < memory_requirement) {
            continue;
        }
        
        // 计算设备得分
        float score = 0.0f;
        
        // 根据任务类型评分
        if (strcmp(task_type, "matrix_multiply") == 0) {
            // 矩阵乘法更看重计算单元数量
            score = device->capabilities.compute_units * 2.0f;
        } else if (strcmp(task_type, "vector_add") == 0) {
            // 向量加法更看重内存带宽
            score = device->capabilities.compute_units * 1.0f;
        } else {
            // 默认评分
            score = device->capabilities.compute_units * 1.5f;
        }
        
        // 考虑内存大小
        score *= (float)device->capabilities.memory_size / (float)memory_requirement;
        
        if (score > best_score) {
            best_score = score;
            best_device = device;
        }
    }
    
    return best_device;
}

// 获取当前设备
HAL_Device* device_manager_get_current_device(void) {
    if (!g_device_manager) return NULL;
    return g_device_manager->current_device;
}

// 切换设备
int device_manager_switch_device(HAL_Device* device) {
    if (!g_device_manager || !device) return -1;
    
    // 验证设备是否在列表中
    int found = 0;
    for (int i = 0; i < g_device_manager->num_devices; i++) {
        if (g_device_manager->devices[i] == device) {
            found = 1;
            break;
        }
    }
    
    if (!found) return -1;
    
    g_device_manager->current_device = device;
    return 0;
}

// 释放设备管理器资源
void device_manager_cleanup(void) {
    if (!g_device_manager) return;
    
    if (g_device_manager->devices) {
        // 清理设备资源
        for (int i = 0; i < g_device_manager->num_devices; i++) {
            if (g_device_manager->devices[i]) {
                // 释放设备特定资源
                if (g_device_manager->devices[i]->device_type == DEVICE_TYPE_GPU) {
                    // 释放GPU内存
                    if (g_device_manager->devices[i]->allocate_memory) {
                        void* gpu_memory = g_device_manager->devices[i]->device_specific_data;
                        if (gpu_memory) {
                            g_device_manager->devices[i]->free_memory(gpu_memory);
                        }
                    }
                    
                    // 释放GPU上下文
                    if (g_device_manager->devices[i]->device_specific_data) {
                        // 假设device_specific_data指向GPU上下文结构
                        GPUContext* ctx = (GPUContext*)g_device_manager->devices[i]->device_specific_data;
                        if (ctx->command_queue) {
                            clReleaseCommandQueue(ctx->command_queue);
                        }
                        if (ctx->context) {
                            clReleaseContext(ctx->context);
                        }
                        free(ctx);
                    }
                }
            }
        }
        free(g_device_manager->devices);
    }
    
    free(g_device_manager);
    g_device_manager = NULL;
} 