#include "low_memory_llm.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

// 创建张量
Tensor* tensor_create(size_t* shape, size_t ndim, QuantType quant_type) {
    if (!shape || ndim == 0) return NULL;

    Tensor* tensor = (Tensor*)calloc(1, sizeof(Tensor));
    if (!tensor) return NULL;

    // 计算总元素数
    size_t total_size = 1;
    for (size_t i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }

    // 分配形状数组
    tensor->shape = (size_t*)malloc(ndim * sizeof(size_t));
    if (!tensor->shape) {
        free(tensor);
        return NULL;
    }
    memcpy(tensor->shape, shape, ndim * sizeof(size_t));

    // 设置维度信息
    tensor->ndim = ndim;
    tensor->size = total_size;
    tensor->quant_type = quant_type;
    tensor->is_view = 0;
    tensor->scales = NULL;
    tensor->zero_points = NULL;

    // 根据量化类型分配数据空间
    size_t elem_size;
    switch (quant_type) {
        case QUANT_INT8:
            elem_size = sizeof(int8_t);
            break;
        case QUANT_INT4:
        case QUANT_INT2:
            elem_size = sizeof(uint8_t);  // 压缩存储
            break;
        case QUANT_NONE:
        default:
            elem_size = sizeof(float);
            break;
    }

    tensor->data = calloc(total_size, elem_size);
    if (!tensor->data) {
        free(tensor->shape);
        free(tensor);
        return NULL;
    }

    return tensor;
}

// 释放张量
void tensor_free(Tensor* tensor) {
    if (!tensor) return;

    if (tensor->data && !tensor->is_view) {
        free(tensor->data);
    }
    
    if (tensor->shape) {
        free(tensor->shape);
    }
    
    if (tensor->scales) {
        free(tensor->scales);
    }
    
    if (tensor->zero_points) {
        free(tensor->zero_points);
    }
    
    free(tensor);
}

// 创建张量视图
Tensor* tensor_view(Tensor* tensor, size_t* new_shape, size_t new_ndim) {
    if (!tensor || !new_shape || new_ndim == 0) return NULL;

    // 计算新形状的总大小
    size_t new_size = 1;
    for (size_t i = 0; i < new_ndim; i++) {
        new_size *= new_shape[i];
    }

    // 检查总大小是否匹配
    if (new_size != tensor->size) return NULL;

    Tensor* view = (Tensor*)calloc(1, sizeof(Tensor));
    if (!view) return NULL;

    // 分配新的形状数组
    view->shape = (size_t*)malloc(new_ndim * sizeof(size_t));
    if (!view->shape) {
        free(view);
        return NULL;
    }
    memcpy(view->shape, new_shape, new_ndim * sizeof(size_t));

    // 设置视图属性
    view->ndim = new_ndim;
    view->size = new_size;
    view->data = tensor->data;  // 共享数据
    view->quant_type = tensor->quant_type;
    view->scales = tensor->scales;  // 共享量化参数
    view->zero_points = tensor->zero_points;
    view->is_view = 1;  // 标记为视图

    return view;
}

// 张量量化
int tensor_quantize(Tensor* tensor, QuantConfig* config) {
    if (!tensor || !config) return -1;
    if (tensor->quant_type != QUANT_NONE) return -1;  // 已经量化
    
    size_t num_channels = tensor->shape[tensor->ndim - 1];
    size_t elements_per_channel = tensor->size / num_channels;
    
    // 分配量化后的内存
    size_t quant_elem_size;
    switch (config->quant_type) {
        case QUANT_INT8:
            quant_elem_size = sizeof(int8_t);
            break;
        case QUANT_INT4:
            quant_elem_size = sizeof(uint8_t) / 2;  // 每字节存2个INT4
            break;
        case QUANT_INT2:
            quant_elem_size = sizeof(uint8_t) / 4;  // 每字节存4个INT2
            break;
        default:
            return -1;
    }
    
    void* quant_data = calloc(tensor->size, quant_elem_size);
    if (!quant_data) return -1;
    
    float* scales = calloc(num_channels, sizeof(float));
    float* zero_points = calloc(num_channels, sizeof(float));
    if (!scales || !zero_points) {
        free(quant_data);
        free(scales);
        free(zero_points);
        return -1;
    }
    
    // 对每个通道进行量化
    float* float_data = (float*)tensor->data;
    for (size_t c = 0; c < num_channels; c++) {
        // 1. 计算该通道的最大最小值
        float min_val = FLT_MAX;
        float max_val = -FLT_MAX;
        for (size_t i = 0; i < elements_per_channel; i++) {
            size_t idx = c + i * num_channels;
            float val = float_data[idx];
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }
        
        // 2. 计算量化参数
        float scale, zero_point;
        int num_bits;
        switch (config->quant_type) {
            case QUANT_INT8:
                num_bits = 8;
                break;
            case QUANT_INT4:
                num_bits = 4;
                break;
            case QUANT_INT2:
                num_bits = 2;
                break;
            default:
                num_bits = 8;
        }
        
        int max_quant = (1 << (num_bits - 1)) - 1;
        int min_quant = -(1 << (num_bits - 1));
        
        if (config->symmetric) {
            float max_abs = fmaxf(fabsf(min_val), fabsf(max_val));
            scale = max_abs / (float)max_quant;
            zero_point = 0.0f;
        } else {
            scale = (max_val - min_val) / (float)(max_quant - min_quant);
            zero_point = (min_val + max_val) / 2.0f;
        }
        
        if (!config->per_channel) {
            // 使用全局量化参数
            scale = config->scale;
            zero_point = config->zero_point;
        }
        
        scales[c] = scale;
        zero_points[c] = zero_point;
        
        // 3. 量化数据
        for (size_t i = 0; i < elements_per_channel; i++) {
            size_t idx = c + i * num_channels;
            float val = float_data[idx];
            
            // 量化公式: q = round((x - zero_point) / scale)
            int quant_val = roundf((val - zero_point) / scale);
            
            // 限制在量化范围内
            if (quant_val > max_quant) quant_val = max_quant;
            if (quant_val < min_quant) quant_val = min_quant;
            
            // 存储量化值
            switch (config->quant_type) {
                case QUANT_INT8:
                    ((int8_t*)quant_data)[idx] = (int8_t)quant_val;
                    break;
                case QUANT_INT4:
                    {
                        size_t byte_idx = idx / 2;
                        uint8_t* byte_ptr = &((uint8_t*)quant_data)[byte_idx];
                        if (idx % 2 == 0) {
                            *byte_ptr = (*byte_ptr & 0x0F) | ((quant_val & 0x0F) << 4);
                        } else {
                            *byte_ptr = (*byte_ptr & 0xF0) | (quant_val & 0x0F);
                        }
                    }
                    break;
                case QUANT_INT2:
                    {
                        size_t byte_idx = idx / 4;
                        uint8_t* byte_ptr = &((uint8_t*)quant_data)[byte_idx];
                        int shift = 6 - (idx % 4) * 2;
                        *byte_ptr = (*byte_ptr & ~(0x03 << shift)) | ((quant_val & 0x03) << shift);
                    }
                    break;
            }
        }
    }
    
    // 更新张量
    free(tensor->data);
    tensor->data = quant_data;
    tensor->scales = scales;
    tensor->zero_points = zero_points;
    tensor->quant_type = config->quant_type;
    
    return 0;
}

// 张量反量化
int tensor_dequantize(Tensor* tensor) {
    if (!tensor) return -1;
    if (tensor->quant_type == QUANT_NONE) return 0;  // 已经是非量化
    
    size_t num_channels = tensor->shape[tensor->ndim - 1];
    size_t elements_per_channel = tensor->size / num_channels;
    
    // 分配反量化后的内存
    float* float_data = calloc(tensor->size, sizeof(float));
    if (!float_data) return -1;
    
    // 对每个通道进行反量化
    for (size_t c = 0; c < num_channels; c++) {
        float scale = tensor->scales[c];
        float zero_point = tensor->zero_points[c];
        
        for (size_t i = 0; i < elements_per_channel; i++) {
            size_t idx = c + i * num_channels;
            int quant_val;
            
            // 读取量化值
            switch (tensor->quant_type) {
                case QUANT_INT8:
                    quant_val = ((int8_t*)tensor->data)[idx];
                    break;
                case QUANT_INT4:
                    {
                        size_t byte_idx = idx / 2;
                        uint8_t byte = ((uint8_t*)tensor->data)[byte_idx];
                        quant_val = (idx % 2 == 0) ? (byte >> 4) : (byte & 0x0F);
                        if (quant_val > 7) quant_val -= 16;  // 符号扩展
                    }
                    break;
                case QUANT_INT2:
                    {
                        size_t byte_idx = idx / 4;
                        uint8_t byte = ((uint8_t*)tensor->data)[byte_idx];
                        int shift = 6 - (idx % 4) * 2;
                        quant_val = (byte >> shift) & 0x03;
                        if (quant_val > 1) quant_val -= 4;  // 符号扩展
                    }
                    break;
                default:
                    free(float_data);
                    return -1;
            }
            
            // 反量化公式: x = q * scale + zero_point
            float_data[idx] = quant_val * scale + zero_point;
        }
    }
    
    // 更新张量
    free(tensor->data);
    free(tensor->scales);
    free(tensor->zero_points);
    
    tensor->data = float_data;
    tensor->scales = NULL;
    tensor->zero_points = NULL;
    tensor->quant_type = QUANT_NONE;
    
    return 0;
}

// 张量加法
int tensor_add(const Tensor* a, const Tensor* b, Tensor* c) {
    if (!a || !b || !c) return -1;
    if (a->size != b->size || a->size != c->size) return -1;
    
    // 处理量化情况
    if (a->quant_type != QUANT_NONE || b->quant_type != QUANT_NONE) {
        // 创建临时反量化缓冲区
        Tensor temp_a = *a;
        Tensor temp_b = *b;
        if (tensor_dequantize(&temp_a) != 0 || tensor_dequantize(&temp_b) != 0) {
            return -1;
        }
        
        // 执行加法
        float* a_data = (float*)temp_a.data;
        float* b_data = (float*)temp_b.data;
        float* c_data = (float*)c->data;
        
        for (size_t i = 0; i < c->size; i++) {
            c_data[i] = a_data[i] + b_data[i];
        }
        
        // 清理临时缓冲区
        free(temp_a.data);
        free(temp_b.data);
    } else {
        // 直接进行浮点运算
        float* a_data = (float*)a->data;
        float* b_data = (float*)b->data;
        float* c_data = (float*)c->data;
        
        for (size_t i = 0; i < c->size; i++) {
            c_data[i] = a_data[i] + b_data[i];
        }
    }
    
    return 0;
}

// 张量乘法（逐元素）
int tensor_multiply(const Tensor* a, const Tensor* b, Tensor* c) {
    if (!a || !b || !c) return -1;
    if (a->size != b->size || a->size != c->size) return -1;
    
    // 处理量化情况
    if (a->quant_type != QUANT_NONE || b->quant_type != QUANT_NONE) {
        // 创建临时反量化缓冲区
        Tensor temp_a = *a;
        Tensor temp_b = *b;
        if (tensor_dequantize(&temp_a) != 0 || tensor_dequantize(&temp_b) != 0) {
            return -1;
        }
        
        // 执行乘法
        float* a_data = (float*)temp_a.data;
        float* b_data = (float*)temp_b.data;
        float* c_data = (float*)c->data;
        
        for (size_t i = 0; i < c->size; i++) {
            c_data[i] = a_data[i] * b_data[i];
        }
        
        // 清理临时缓冲区
        free(temp_a.data);
        free(temp_b.data);
    } else {
        // 直接进行浮点运算
        float* a_data = (float*)a->data;
        float* b_data = (float*)b->data;
        float* c_data = (float*)c->data;
        
        for (size_t i = 0; i < c->size; i++) {
            c_data[i] = a_data[i] * b_data[i];
        }
    }
    
    return 0;
}

// 张量矩阵乘法
int tensor_matmul(const Tensor* a, const Tensor* b, Tensor* c) {
    if (!a || !b || !c) return -1;
    if (a->ndim != 2 || b->ndim != 2 || c->ndim != 2) return -1;
    if (a->shape[1] != b->shape[0] || 
        a->shape[0] != c->shape[0] || 
        b->shape[1] != c->shape[1]) return -1;
    
    // 处理量化情况
    if (a->quant_type != QUANT_NONE || b->quant_type != QUANT_NONE) {
        // 创建临时反量化缓冲区
        Tensor temp_a = *a;
        Tensor temp_b = *b;
        if (tensor_dequantize(&temp_a) != 0 || tensor_dequantize(&temp_b) != 0) {
            return -1;
        }
        
        // 执行矩阵乘法
        float* a_data = (float*)temp_a.data;
        float* b_data = (float*)temp_b.data;
        float* c_data = (float*)c->data;
        
        size_t M = a->shape[0];
        size_t K = a->shape[1];
        size_t N = b->shape[1];
        
        // 使用分块计算减少内存访问
        #define BLOCK_SIZE 32
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; k += BLOCK_SIZE) {
                    size_t block_size = (k + BLOCK_SIZE > K) ? (K - k) : BLOCK_SIZE;
                    for (size_t b = 0; b < block_size; b++) {
                        sum += a_data[i * K + k + b] * b_data[(k + b) * N + j];
                    }
                }
                c_data[i * N + j] = sum;
            }
        }
        
        // 清理临时缓冲区
        free(temp_a.data);
        free(temp_b.data);
    } else {
        // 直接进行浮点运算
        float* a_data = (float*)a->data;
        float* b_data = (float*)b->data;
        float* c_data = (float*)c->data;
        
        size_t M = a->shape[0];
        size_t K = a->shape[1];
        size_t N = b->shape[1];
        
        // 使用分块计算减少内存访问
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; k += BLOCK_SIZE) {
                    size_t block_size = (k + BLOCK_SIZE > K) ? (K - k) : BLOCK_SIZE;
                    for (size_t b = 0; b < block_size; b++) {
                        sum += a_data[i * K + k + b] * b_data[(k + b) * N + j];
                    }
                }
                c_data[i * N + j] = sum;
            }
        }
    }
    
    return 0;
}

// 张量转置
int tensor_transpose(const Tensor* input, Tensor* output) {
    if (!input || !output) return -1;
    if (input->ndim != 2 || output->ndim != 2) return -1;
    if (input->shape[0] != output->shape[1] || 
        input->shape[1] != output->shape[0]) return -1;
    
    // 处理量化情况
    if (input->quant_type != QUANT_NONE) {
        // 创建临时反量化缓冲区
        Tensor temp = *input;
        if (tensor_dequantize(&temp) != 0) {
            return -1;
        }
        
        // 执行转置
        float* in_data = (float*)temp.data;
        float* out_data = (float*)output->data;
        
        size_t M = input->shape[0];
        size_t N = input->shape[1];
        
        // 使用分块转置减少缓存未命中
        #define TRANS_BLOCK_SIZE 32
        for (size_t i = 0; i < M; i += TRANS_BLOCK_SIZE) {
            for (size_t j = 0; j < N; j += TRANS_BLOCK_SIZE) {
                size_t block_m = (i + TRANS_BLOCK_SIZE > M) ? (M - i) : TRANS_BLOCK_SIZE;
                size_t block_n = (j + TRANS_BLOCK_SIZE > N) ? (N - j) : TRANS_BLOCK_SIZE;
                
                for (size_t bi = 0; bi < block_m; bi++) {
                    for (size_t bj = 0; bj < block_n; bj++) {
                        out_data[(j + bj) * M + (i + bi)] = in_data[(i + bi) * N + (j + bj)];
                    }
                }
            }
        }
        
        // 清理临时缓冲区
        free(temp.data);
    } else {
        // 直接进行浮点运算
        float* in_data = (float*)input->data;
        float* out_data = (float*)output->data;
        
        size_t M = input->shape[0];
        size_t N = input->shape[1];
        
        // 使用分块转置减少缓存未命中
        for (size_t i = 0; i < M; i += TRANS_BLOCK_SIZE) {
            for (size_t j = 0; j < N; j += TRANS_BLOCK_SIZE) {
                size_t block_m = (i + TRANS_BLOCK_SIZE > M) ? (M - i) : TRANS_BLOCK_SIZE;
                size_t block_n = (j + TRANS_BLOCK_SIZE > N) ? (N - j) : TRANS_BLOCK_SIZE;
                
                for (size_t bi = 0; bi < block_m; bi++) {
                    for (size_t bj = 0; bj < block_n; bj++) {
                        out_data[(j + bj) * M + (i + bi)] = in_data[(i + bi) * N + (j + bj)];
                    }
                }
            }
        }
    }
    
    return 0;
} 