#include "quantization.h"
#include "fp8.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// FP16相关的辅助函数
static uint16_t float_to_fp16(float value) {
    uint32_t x = *(uint32_t*)&value;
    uint32_t sign = (x >> 31) & 0x1;
    uint32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t frac = (x >> 13) & 0x3FF;
    
    if (exp > 31) exp = 31;
    else if (exp < 0) exp = 0;
    
    return (sign << 15) | (exp << 10) | frac;
}

static float fp16_to_float(uint16_t value) {
    uint32_t sign = (value >> 15) & 0x1;
    uint32_t exp = ((value >> 10) & 0x1F) - 15 + 127;
    uint32_t frac = (value & 0x3FF) << 13;
    
    uint32_t x = (sign << 31) | (exp << 23) | frac;
    return *(float*)&x;
}

// 查找数据范围
static void find_data_range(const float* data, size_t size, float* min_val, float* max_val) {
    *min_val = FLT_MAX;
    *max_val = -FLT_MAX;
    
    for (size_t i = 0; i < size; i++) {
        if (data[i] < *min_val) *min_val = data[i];
        if (data[i] > *max_val) *max_val = data[i];
    }
}

// 计算量化参数
int quant_calibrate(QuantParams* params, const float* data, size_t size, const QuantConfig* config) {
    if (!params || !data || !config || size == 0) return -1;
    
    // 查找数据范围
    float min_val, max_val;
    find_data_range(data, size, &min_val, &max_val);
    
    // 应用裁剪比例
    if (config->clip_ratio > 0 && config->clip_ratio < 1) {
        float range = max_val - min_val;
        min_val += range * config->clip_ratio;
        max_val -= range * config->clip_ratio;
    }
    
    params->min_value = min_val;
    params->max_value = max_val;
    
    // 计算量化参数
    switch (config->type) {
        case QUANT_TYPE_INT8: {
            if (config->symmetric) {
                float abs_max = fmaxf(fabsf(min_val), fabsf(max_val));
                params->scale = abs_max / 127.0f;
                params->zero_point = 0;
            } else {
                params->scale = (max_val - min_val) / 255.0f;
                params->zero_point = (int32_t)(-min_val / params->scale + 0.5f);
                if (params->zero_point < 0) params->zero_point = 0;
                if (params->zero_point > 255) params->zero_point = 255;
            }
            break;
        }
        case QUANT_TYPE_INT4: {
            if (config->symmetric) {
                float abs_max = fmaxf(fabsf(min_val), fabsf(max_val));
                params->scale = abs_max / 7.0f;
                params->zero_point = 0;
            } else {
                params->scale = (max_val - min_val) / 15.0f;
                params->zero_point = (int32_t)(-min_val / params->scale + 0.5f);
                if (params->zero_point < 0) params->zero_point = 0;
                if (params->zero_point > 15) params->zero_point = 15;
            }
            break;
        }
        case QUANT_TYPE_FP16:
        case QUANT_TYPE_FP8:
            params->scale = 1.0f;
            params->zero_point = 0;
            break;
        case QUANT_TYPE_DYNAMIC:
            // 动态量化在运行时计算参数
            break;
    }
    
    return 0;
}

// 量化数据
int quant_quantize(void* output, const float* input, size_t size, 
                  const QuantParams* params, const QuantConfig* config) {
    if (!output || !input || !params || !config || size == 0) return -1;
    
    switch (config->type) {
        case QUANT_TYPE_INT8: {
            uint8_t* out = (uint8_t*)output;
            for (size_t i = 0; i < size; i++) {
                float scaled = input[i] / params->scale + params->zero_point;
                if (scaled > 255) scaled = 255;
                if (scaled < 0) scaled = 0;
                out[i] = (uint8_t)(scaled + 0.5f);
            }
            break;
        }
        case QUANT_TYPE_INT4: {
            uint8_t* out = (uint8_t*)output;
            for (size_t i = 0; i < size; i += 2) {
                float scaled1 = input[i] / params->scale + params->zero_point;
                if (scaled1 > 15) scaled1 = 15;
                if (scaled1 < 0) scaled1 = 0;
                
                float scaled2 = (i + 1 < size) ? 
                    input[i + 1] / params->scale + params->zero_point : 0;
                if (scaled2 > 15) scaled2 = 15;
                if (scaled2 < 0) scaled2 = 0;
                
                out[i/2] = ((uint8_t)(scaled1 + 0.5f) << 4) | 
                          ((uint8_t)(scaled2 + 0.5f) & 0x0F);
            }
            break;
        }
        case QUANT_TYPE_FP16: {
            uint16_t* out = (uint16_t*)output;
            for (size_t i = 0; i < size; i++) {
                out[i] = float_to_fp16(input[i]);
            }
            break;
        }
        case QUANT_TYPE_FP8: {
            FP8* out = (FP8*)output;
            // 对于权重使用E4M3，对于激活值使用E5M2
            FP8Format format = config->per_channel ? FP8_E4M3 : FP8_E5M2;
            for (size_t i = 0; i < size; i++) {
                out[i] = float_to_fp8(input[i], format);
            }
            break;
        }
        case QUANT_TYPE_DYNAMIC: {
            // 动态量化：根据每个tensor块动态计算量化参数
            size_t block_size = 256;  // 可配置的块大小
            uint8_t* out = (uint8_t*)output;
            
            for (size_t block = 0; block < size; block += block_size) {
                size_t curr_size = (block + block_size > size) ? (size - block) : block_size;
                
                // 计算当前块的量化参数
                QuantParams block_params;
                quant_calibrate(&block_params, &input[block], curr_size, config);
                
                // 量化当前块
                for (size_t i = 0; i < curr_size; i++) {
                    float scaled = input[block + i] / block_params.scale + block_params.zero_point;
                    if (scaled > 255) scaled = 255;
                    if (scaled < 0) scaled = 0;
                    out[block + i] = (uint8_t)(scaled + 0.5f);
                }
            }
            break;
        }
    }
    
    return 0;
}

// 反量化数据
int quant_dequantize(float* output, const void* input, size_t size,
                    const QuantParams* params, const QuantConfig* config) {
    if (!output || !input || !params || !config || size == 0) return -1;
    
    switch (config->type) {
        case QUANT_TYPE_INT8: {
            const uint8_t* in = (const uint8_t*)input;
            for (size_t i = 0; i < size; i++) {
                output[i] = (in[i] - params->zero_point) * params->scale;
            }
            break;
        }
        case QUANT_TYPE_INT4: {
            const uint8_t* in = (const uint8_t*)input;
            for (size_t i = 0; i < size; i += 2) {
                uint8_t val = in[i/2];
                output[i] = ((val >> 4) - params->zero_point) * params->scale;
                if (i + 1 < size) {
                    output[i + 1] = ((val & 0x0F) - params->zero_point) * params->scale;
                }
            }
            break;
        }
        case QUANT_TYPE_FP16: {
            const uint16_t* in = (const uint16_t*)input;
            for (size_t i = 0; i < size; i++) {
                output[i] = fp16_to_float(in[i]);
            }
            break;
        }
        case QUANT_TYPE_FP8: {
            const FP8* in = (const FP8*)input;
            // 对于权重使用E4M3，对于激活值使用E5M2
            FP8Format format = config->per_channel ? FP8_E4M3 : FP8_E5M2;
            for (size_t i = 0; i < size; i++) {
                output[i] = fp8_to_float(in[i], format);
            }
            break;
        }
        case QUANT_TYPE_DYNAMIC: {
            // 动态量化的反量化需要存储每个块的量化参数
            // 这里假设量化参数被存储在输入数据之前
            const uint8_t* in = (const uint8_t*)input;
            size_t block_size = 256;
            
            for (size_t block = 0; block < size; block += block_size) {
                size_t curr_size = (block + block_size > size) ? (size - block) : block_size;
                
                // 读取当前块的量化参数
                QuantParams block_params;
                memcpy(&block_params, in + block, sizeof(QuantParams));
                
                // 反量化当前块
                for (size_t i = 0; i < curr_size; i++) {
                    output[block + i] = (in[block + sizeof(QuantParams) + i] - 
                                       block_params.zero_point) * block_params.scale;
                }
            }
            break;
        }
    }
    
    return 0;
}

// 初始化量化参数
int quant_init_params(QuantParams* params, const float* data, size_t size, const QuantConfig* config) {
    if (!params || !config) return -1;
    
    memset(params, 0, sizeof(QuantParams));
    
    if (data && size > 0) {
        return quant_calibrate(params, data, size, config);
    }
    
    return 0;
}

// 获取量化类型的位宽
int quant_get_bitwidth(QuantType type) {
    switch (type) {
        case QUANT_TYPE_INT8:
        case QUANT_TYPE_FP8:
            return 8;
        case QUANT_TYPE_INT4:
            return 4;
        case QUANT_TYPE_FP16:
            return 16;
        case QUANT_TYPE_DYNAMIC:
            return 8;  // 默认使用8位
        default:
            return -1;
    }
}

// 获取量化后的数据大小
size_t quant_get_size(size_t num_elements, QuantType type) {
    switch (type) {
        case QUANT_TYPE_INT8:
        case QUANT_TYPE_FP8:
            return num_elements;
        case QUANT_TYPE_INT4:
            return (num_elements + 1) / 2;
        case QUANT_TYPE_FP16:
            return num_elements * 2;
        case QUANT_TYPE_DYNAMIC:
            // 动态量化需要额外存储量化参数
            return num_elements + (num_elements / 256 + 1) * sizeof(QuantParams);
        default:
            return 0;
    }
} 