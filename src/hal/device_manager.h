#ifndef DEVICE_MANAGER_H
#define DEVICE_MANAGER_H

#include "hal.h"

// 设备管理器结构体
typedef struct {
    HAL_Device** devices;
    int num_devices;
    HAL_Device* current_device;
} DeviceManager;

// 初始化设备管理器
int device_manager_init(void);

// 扫描可用设备
int device_manager_scan_devices(void);

// 根据任务特征选择最优设备
HAL_Device* device_manager_select_device(const char* task_type, size_t memory_requirement);

// 获取当前设备
HAL_Device* device_manager_get_current_device(void);

// 切换设备
int device_manager_switch_device(HAL_Device* device);

// 释放设备管理器资源
void device_manager_cleanup(void);

#endif // DEVICE_MANAGER_H 