#include "qat.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// 内部辅助函数：计算伪量化
static void fake_quantize(float* data, size_t size, const QuantParams* params, const QuantConfig* config) {
    // 分配临时缓冲区
    void* quant_buffer = malloc(quant_get_size(size, config->type));
    if (!quant_buffer) return;
    
    // 量化
    quant_quantize(quant_buffer, data, size, params, config);
    
    // 反量化回float
    quant_dequantize(data, quant_buffer, size, params, config);
    
    free(quant_buffer);
}

// 内部辅助函数：更新移动平均
static void update_running_stats(float* running_val, float new_val, float smooth_factor) {
    *running_val = *running_val * smooth_factor + new_val * (1.0f - smooth_factor);
}

// 初始化QAT
int qat_init(QATState** state, size_t num_tensors, const QATConfig* config) {
    if (!state || !config || num_tensors == 0) return -1;
    
    *state = (QATState*)malloc(sizeof(QATState));
    if (!*state) return -1;
    
    // 初始化状态
    (*state)->num_tensors = num_tensors;
    (*state)->current_step = 0;
    (*state)->is_calibrating = 1;
    
    // 分配参数数组
    (*state)->params = (QuantParams*)malloc(sizeof(QuantParams) * num_tensors);
    (*state)->running_min = (float*)malloc(sizeof(float) * num_tensors);
    (*state)->running_max = (float*)malloc(sizeof(float) * num_tensors);
    
    if (!(*state)->params || !(*state)->running_min || !(*state)->running_max) {
        qat_cleanup(*state);
        return -1;
    }
    
    // 初始化参数
    for (size_t i = 0; i < num_tensors; i++) {
        memset(&(*state)->params[i], 0, sizeof(QuantParams));
        (*state)->running_min[i] = FLT_MAX;
        (*state)->running_max[i] = -FLT_MAX;
    }
    
    return 0;
}

// 清理QAT资源
void qat_cleanup(QATState* state) {
    if (!state) return;
    
    if (state->params) free(state->params);
    if (state->running_min) free(state->running_min);
    if (state->running_max) free(state->running_max);
    
    free(state);
}

// 前向传播中的量化操作
int qat_forward_quant(QATState* state, 
                     size_t tensor_idx,
                     float* data,
                     size_t size,
                     const QATConfig* config) {
    if (!state || !data || !config || tensor_idx >= state->num_tensors) return -1;
    
    // 在校准阶段收集统计信息
    if (state->is_calibrating) {
        float min_val = FLT_MAX;
        float max_val = -FLT_MAX;
        
        for (size_t i = 0; i < size; i++) {
            if (data[i] < min_val) min_val = data[i];
            if (data[i] > max_val) max_val = data[i];
        }
        
        // 更新运行时统计信息
        update_running_stats(&state->running_min[tensor_idx], min_val, config->smooth_factor);
        update_running_stats(&state->running_max[tensor_idx], max_val, config->smooth_factor);
        
        // 检查是否完成校准
        if (state->current_step >= config->calibration_steps) {
            state->is_calibrating = 0;
            
            // 使用收集到的统计信息初始化量化参数
            QuantParams* params = &state->params[tensor_idx];
            params->min_value = state->running_min[tensor_idx];
            params->max_value = state->running_max[tensor_idx];
            
            if (config->quant_config.symmetric) {
                float abs_max = fmaxf(fabsf(params->min_value), fabsf(params->max_value));
                params->scale = abs_max / 127.0f;
                params->zero_point = 0;
            } else {
                params->scale = (params->max_value - params->min_value) / 255.0f;
                params->zero_point = (int32_t)(-params->min_value / params->scale + 0.5f);
            }
        }
    }
    
    // 在训练阶段使用伪量化
    if (!state->is_calibrating && config->fake_quant) {
        fake_quantize(data, size, &state->params[tensor_idx], &config->quant_config);
    }
    
    return 0;
}

// 反向传播中的量化梯度计算
int qat_backward_quant(QATState* state,
                      size_t tensor_idx,
                      float* grad_output,
                      const float* grad_input,
                      const float* original_input,
                      size_t size,
                      const QATConfig* config) {
    if (!state || !grad_output || !grad_input || !original_input || 
        !config || tensor_idx >= state->num_tensors) return -1;
    
    // 只在非校准阶段且启用伪量化时计算梯度
    if (!state->is_calibrating && config->fake_quant) {
        const QuantParams* params = &state->params[tensor_idx];
        
        // 计算量化梯度
        for (size_t i = 0; i < size; i++) {
            float x = original_input[i];
            float scale = params->scale;
            
            // 检查是否在可量化范围内
            if (x >= params->min_value && x <= params->max_value) {
                // 直通估计器（straight-through estimator）
                grad_output[i] = grad_input[i];
            } else {
                // 量化边界外的梯度为0
                grad_output[i] = 0;
            }
        }
    } else {
        // 在校准阶段直接传递梯度
        memcpy(grad_output, grad_input, size * sizeof(float));
    }
    
    return 0;
}

// 更新量化参数
int qat_update_params(QATState* state, const QATConfig* config) {
    if (!state || !config) return -1;
    
    state->current_step++;
    
    // 在训练阶段定期更新量化参数
    if (!state->is_calibrating && 
        config->update_step > 0 && 
        state->current_step % config->update_step == 0) {
        
        for (size_t i = 0; i < state->num_tensors; i++) {
            QuantParams* params = &state->params[i];
            
            // 使用运行时统计更新参数
            if (config->quant_config.symmetric) {
                float abs_max = fmaxf(fabsf(state->running_min[i]), 
                                    fabsf(state->running_max[i]));
                params->scale = abs_max / 127.0f;
                params->zero_point = 0;
            } else {
                params->scale = (state->running_max[i] - state->running_min[i]) / 255.0f;
                params->zero_point = (int32_t)(-state->running_min[i] / params->scale + 0.5f);
            }
            
            // 更新参数范围
            params->min_value = state->running_min[i];
            params->max_value = state->running_max[i];
        }
    }
    
    return 0;
}

// 获取量化参数
const QuantParams* qat_get_params(const QATState* state, size_t tensor_idx) {
    if (!state || tensor_idx >= state->num_tensors) return NULL;
    return &state->params[tensor_idx];
}

// 保存QAT状态
int qat_save_state(const QATState* state, const char* path) {
    if (!state || !path) return -1;
    
    FILE* fp = fopen(path, "wb");
    if (!fp) return -1;
    
    // 写入基本信息
    fwrite(&state->num_tensors, sizeof(size_t), 1, fp);
    fwrite(&state->current_step, sizeof(size_t), 1, fp);
    fwrite(&state->is_calibrating, sizeof(int), 1, fp);
    
    // 写入参数数组
    fwrite(state->params, sizeof(QuantParams), state->num_tensors, fp);
    fwrite(state->running_min, sizeof(float), state->num_tensors, fp);
    fwrite(state->running_max, sizeof(float), state->num_tensors, fp);
    
    fclose(fp);
    return 0;
}

// 加载QAT状态
int qat_load_state(QATState* state, const char* path) {
    if (!state || !path) return -1;
    
    FILE* fp = fopen(path, "rb");
    if (!fp) return -1;
    
    // 读取基本信息
    size_t num_tensors;
    if (fread(&num_tensors, sizeof(size_t), 1, fp) != 1 ||
        num_tensors != state->num_tensors) {
        fclose(fp);
        return -1;
    }
    
    if (fread(&state->current_step, sizeof(size_t), 1, fp) != 1 ||
        fread(&state->is_calibrating, sizeof(int), 1, fp) != 1) {
        fclose(fp);
        return -1;
    }
    
    // 读取参数数组
    if (fread(state->params, sizeof(QuantParams), state->num_tensors, fp) != state->num_tensors ||
        fread(state->running_min, sizeof(float), state->num_tensors, fp) != state->num_tensors ||
        fread(state->running_max, sizeof(float), state->num_tensors, fp) != state->num_tensors) {
        fclose(fp);
        return -1;
    }
    
    fclose(fp);
    return 0;
} 