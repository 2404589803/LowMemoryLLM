#include "mixed_precision.h"
#include "fp8.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <zlib.h>

// CRC32表
static const uint32_t crc32_table[256] = {
    0x00000000, 0x77073096, 0xEE0E612C, 0x990951BA,
    // ... 此处省略其余表项，实际使用时需要完整的CRC32表
};

// 计算CRC32校验和
static uint32_t calculate_crc32(const void* data, size_t size) {
    const uint8_t* buf = (const uint8_t*)data;
    uint32_t crc = 0xFFFFFFFF;
    
    for (size_t i = 0; i < size; i++) {
        crc = (crc >> 8) ^ crc32_table[(crc & 0xFF) ^ buf[i]];
    }
    
    return crc ^ 0xFFFFFFFF;
}

// 文件头结构
typedef struct {
    uint32_t magic;              // 魔数，用于识别文件类型
    uint32_t version;            // 文件版本
    uint32_t header_crc;         // 头部校验和
    uint32_t data_crc;           // 数据校验和
    size_t original_size;        // 原始数据大小
    size_t compressed_size;      // 压缩后数据大小
    int is_compressed;           // 是否压缩
} FileHeader;

#define MP_STATE_MAGIC 0x4D505354  // "MPST"

// 压缩数据
static int compress_data(const void* input, size_t input_size,
                        void** output, size_t* output_size,
                        int level) {
    // 预估压缩后的大小
    *output_size = compressBound(input_size);
    *output = malloc(*output_size);
    if (!*output) return -1;
    
    // 压缩数据
    int ret = compress2(*output, (uLongf*)output_size,
                       input, input_size, level);
    if (ret != Z_OK) {
        free(*output);
        *output = NULL;
        return -1;
    }
    
    return 0;
}

// 解压数据
static int decompress_data(const void* input, size_t input_size,
                          void* output, size_t output_size) {
    uLongf dest_len = output_size;
    int ret = uncompress(output, &dest_len,
                        input, input_size);
    return (ret == Z_OK) ? 0 : -1;
}

// 带选项的保存函数实现
int mixed_precision_save_state_with_options(const MixedPrecisionState* state,
                                          const char* path,
                                          const SaveOptions* options) {
    if (!state || !path || !options) return -1;
    
    FILE* fp = fopen(path, "wb");
    if (!fp) return -1;
    
    // 准备文件头
    FileHeader header = {
        .magic = MP_STATE_MAGIC,
        .version = MP_STATE_VERSION,
        .is_compressed = options->use_compression,
        .original_size = 0,
        .compressed_size = 0
    };
    
    // 写入临时头部（稍后更新）
    fwrite(&header, sizeof(header), 1, fp);
    
    // 创建临时缓冲区存储所有数据
    size_t total_size = sizeof(float) + 2 * sizeof(int) + sizeof(size_t);
    total_size += state->num_layers * sizeof(size_t);  // weight_sizes
    
    for (size_t i = 0; i < state->num_layers; i++) {
        if (state->fp32_weights[i] && state->weight_sizes[i] > 0) {
            total_size += sizeof(int) + state->weight_sizes[i] * sizeof(float);
        } else {
            total_size += sizeof(int);
        }
    }
    
    void* buffer = malloc(total_size);
    if (!buffer) {
        fclose(fp);
        return -1;
    }
    
    // 写入数据到缓冲区
    char* ptr = buffer;
    memcpy(ptr, &state->current_loss_scale, sizeof(float));
    ptr += sizeof(float);
    memcpy(ptr, &state->overflow_count, sizeof(int));
    ptr += sizeof(int);
    memcpy(ptr, &state->step_count, sizeof(int));
    ptr += sizeof(int);
    memcpy(ptr, &state->num_layers, sizeof(size_t));
    ptr += sizeof(size_t);
    
    // 写入权重大小信息
    memcpy(ptr, state->weight_sizes, state->num_layers * sizeof(size_t));
    ptr += state->num_layers * sizeof(size_t);
    
    // 写入权重数据
    for (size_t i = 0; i < state->num_layers; i++) {
        if (state->fp32_weights[i] && state->weight_sizes[i] > 0) {
            int has_backup = 1;
            memcpy(ptr, &has_backup, sizeof(int));
            ptr += sizeof(int);
            memcpy(ptr, state->fp32_weights[i], state->weight_sizes[i] * sizeof(float));
            ptr += state->weight_sizes[i] * sizeof(float);
        } else {
            int has_backup = 0;
            memcpy(ptr, &has_backup, sizeof(int));
            ptr += sizeof(int);
        }
    }
    
    // 计算数据校验和
    header.data_crc = calculate_crc32(buffer, total_size);
    header.original_size = total_size;
    
    void* compressed_data = NULL;
    size_t compressed_size = 0;
    
    // 如果需要压缩
    if (options->use_compression) {
        if (compress_data(buffer, total_size,
                         &compressed_data, &compressed_size,
                         options->compression_level) != 0) {
            free(buffer);
            fclose(fp);
            return -1;
        }
        header.compressed_size = compressed_size;
    } else {
        compressed_data = buffer;
        compressed_size = total_size;
        header.compressed_size = total_size;
    }
    
    // 计算头部校验和
    header.header_crc = calculate_crc32(&header, 
                                      offsetof(FileHeader, header_crc));
    
    // 更新文件头
    fseek(fp, 0, SEEK_SET);
    fwrite(&header, sizeof(header), 1, fp);
    
    // 写入数据
    fwrite(compressed_data, 1, compressed_size, fp);
    
    // 清理
    if (options->use_compression) {
        free(compressed_data);
    }
    free(buffer);
    fclose(fp);
    
    return 0;
}

// 带选项的加载函数实现
int mixed_precision_load_state_with_options(MixedPrecisionState* state,
                                          const char* path,
                                          const SaveOptions* options) {
    if (!state || !path || !options) return -1;
    
    FILE* fp = fopen(path, "rb");
    if (!fp) return -1;
    
    // 读取文件头
    FileHeader header;
    if (fread(&header, sizeof(header), 1, fp) != 1) {
        fclose(fp);
        return -1;
    }
    
    // 验证魔数和版本
    if (header.magic != MP_STATE_MAGIC || 
        header.version != MP_STATE_VERSION) {
        fclose(fp);
        return -1;
    }
    
    // 验证头部校验和
    uint32_t header_crc = calculate_crc32(&header, 
                                        offsetof(FileHeader, header_crc));
    if (options->verify_checksum && header_crc != header.header_crc) {
        fclose(fp);
        return -1;
    }
    
    // 读取压缩数据
    void* compressed_data = malloc(header.compressed_size);
    if (!compressed_data) {
        fclose(fp);
        return -1;
    }
    
    if (fread(compressed_data, 1, header.compressed_size, fp) 
        != header.compressed_size) {
        free(compressed_data);
        fclose(fp);
        return -1;
    }
    
    // 准备解压缩缓冲区
    void* buffer = malloc(header.original_size);
    if (!buffer) {
        free(compressed_data);
        fclose(fp);
        return -1;
    }
    
    // 如果数据是压缩的，进行解压缩
    if (header.is_compressed) {
        if (decompress_data(compressed_data, header.compressed_size,
                           buffer, header.original_size) != 0) {
            free(buffer);
            free(compressed_data);
            fclose(fp);
            return -1;
        }
        free(compressed_data);
    } else {
        memcpy(buffer, compressed_data, header.original_size);
        free(compressed_data);
    }
    
    // 验证数据校验和
    if (options->verify_checksum) {
        uint32_t data_crc = calculate_crc32(buffer, header.original_size);
        if (data_crc != header.data_crc) {
            free(buffer);
            fclose(fp);
            return -1;
        }
    }
    
    // 读取数据
    char* ptr = buffer;
    memcpy(&state->current_loss_scale, ptr, sizeof(float));
    ptr += sizeof(float);
    memcpy(&state->overflow_count, ptr, sizeof(int));
    ptr += sizeof(int);
    memcpy(&state->step_count, ptr, sizeof(int));
    ptr += sizeof(int);
    
    size_t num_layers;
    memcpy(&num_layers, ptr, sizeof(size_t));
    ptr += sizeof(size_t);
    
    if (num_layers != state->num_layers) {
        free(buffer);
        fclose(fp);
        return -1;
    }
    
    // 读取权重大小信息
    memcpy(state->weight_sizes, ptr, state->num_layers * sizeof(size_t));
    ptr += state->num_layers * sizeof(size_t);
    
    // 清理现有的权重备份
    for (size_t i = 0; i < state->num_layers; i++) {
        if (state->fp32_weights[i]) {
            free(state->fp32_weights[i]);
            state->fp32_weights[i] = NULL;
        }
    }
    
    // 读取权重数据
    for (size_t i = 0; i < state->num_layers; i++) {
        int has_backup;
        memcpy(&has_backup, ptr, sizeof(int));
        ptr += sizeof(int);
        
        if (has_backup && state->weight_sizes[i] > 0) {
            state->fp32_weights[i] = malloc(state->weight_sizes[i] * sizeof(float));
            if (!state->fp32_weights[i]) {
                free(buffer);
                fclose(fp);
                return -1;
            }
            
            memcpy(state->fp32_weights[i], ptr, 
                   state->weight_sizes[i] * sizeof(float));
            ptr += state->weight_sizes[i] * sizeof(float);
        }
    }
    
    free(buffer);
    fclose(fp);
    return 0;
}

// 验证保存文件的完整性
int mixed_precision_verify_state_file(const char* path) {
    if (!path) return -1;
    
    FILE* fp = fopen(path, "rb");
    if (!fp) return -1;
    
    // 读取文件头
    FileHeader header;
    if (fread(&header, sizeof(header), 1, fp) != 1) {
        fclose(fp);
        return -1;
    }
    
    // 验证魔数和版本
    if (header.magic != MP_STATE_MAGIC || 
        header.version != MP_STATE_VERSION) {
        fclose(fp);
        return -1;
    }
    
    // 验证头部校验和
    uint32_t header_crc = calculate_crc32(&header, 
                                        offsetof(FileHeader, header_crc));
    if (header_crc != header.header_crc) {
        fclose(fp);
        return -1;
    }
    
    // 读取数据
    void* data = malloc(header.compressed_size);
    if (!data) {
        fclose(fp);
        return -1;
    }
    
    if (fread(data, 1, header.compressed_size, fp) != header.compressed_size) {
        free(data);
        fclose(fp);
        return -1;
    }
    
    // 如果数据是压缩的，需要解压缩后验证
    if (header.is_compressed) {
        void* decompressed = malloc(header.original_size);
        if (!decompressed) {
            free(data);
            fclose(fp);
            return -1;
        }
        
        if (decompress_data(data, header.compressed_size,
                           decompressed, header.original_size) != 0) {
            free(decompressed);
            free(data);
            fclose(fp);
            return -1;
        }
        
        // 验证解压后数据的校验和
        uint32_t data_crc = calculate_crc32(decompressed, header.original_size);
        free(decompressed);
        free(data);
        fclose(fp);
        
        return (data_crc == header.data_crc) ? 0 : -1;
    } else {
        // 直接验证未压缩数据的校验和
        uint32_t data_crc = calculate_crc32(data, header.original_size);
        free(data);
        fclose(fp);
        
        return (data_crc == header.data_crc) ? 0 : -1;
    }
}

// 为了保持向后兼容性，保留原有的函数
int mixed_precision_save_state(const MixedPrecisionState* state,
                             const char* path) {
    SaveOptions options = DEFAULT_SAVE_OPTIONS;
    return mixed_precision_save_state_with_options(state, path, &options);
}

int mixed_precision_load_state(MixedPrecisionState* state,
                             const char* path) {
    SaveOptions options = DEFAULT_SAVE_OPTIONS;
    return mixed_precision_load_state_with_options(state, path, &options);
}

// 内部辅助函数：精度转换
static void convert_precision(void* output, const void* input, size_t size,
                            PrecisionType from_precision, PrecisionType to_precision) {
    if (from_precision == to_precision) {
        memcpy(output, input, size * sizeof(float));
        return;
    }
    
    const float* in_f32 = (const float*)input;
    
    switch (to_precision) {
        case PRECISION_FP16: {
            uint16_t* out_f16 = (uint16_t*)output;
            for (size_t i = 0; i < size; i++) {
                out_f16[i] = float_to_fp16(in_f32[i]);
            }
            break;
        }
        case PRECISION_FP8: {
            FP8* out_f8 = (FP8*)output;
            for (size_t i = 0; i < size; i++) {
                out_f8[i] = float_to_fp8(in_f32[i], FP8_E5M2);
            }
            break;
        }
        case PRECISION_INT8: {
            int8_t* out_i8 = (int8_t*)output;
            for (size_t i = 0; i < size; i++) {
                float val = in_f32[i] * 127.0f;
                if (val > 127.0f) val = 127.0f;
                if (val < -128.0f) val = -128.0f;
                out_i8[i] = (int8_t)(val + 0.5f);
            }
            break;
        }
        default:
            memcpy(output, input, size * sizeof(float));
            break;
    }
}

// 内部辅助函数：检查数值范围
static int check_value_range(float value, PrecisionType precision) {
    switch (precision) {
        case PRECISION_FP16: {
            float max_val = 65504.0f;  // FP16最大值
            return (fabsf(value) <= max_val);
        }
        case PRECISION_FP8: {
            float max_val = 448.0f;    // FP8 E5M2最大值
            return (fabsf(value) <= max_val);
        }
        case PRECISION_INT8: {
            return (value >= -128.0f && value <= 127.0f);
        }
        default:
            return 1;
    }
}

// 初始化混合精度训练
int mixed_precision_init(MixedPrecisionState** state, 
                        const MixedPrecisionConfig* config,
                        QATState* qat_state) {
    if (!state || !config) return -1;
    
    *state = (MixedPrecisionState*)malloc(sizeof(MixedPrecisionState));
    if (!*state) return -1;
    
    // 初始化状态
    (*state)->current_loss_scale = config->init_loss_scale;
    (*state)->overflow_count = 0;
    (*state)->step_count = 0;
    (*state)->qat_state = qat_state;
    (*state)->num_layers = config->num_layers;
    
    // 分配FP32权重备份空间
    (*state)->fp32_weights = (void**)malloc(sizeof(void*) * config->num_layers);
    if (!(*state)->fp32_weights) {
        free(*state);
        return -1;
    }
    
    // 分配权重大小数组空间
    (*state)->weight_sizes = (size_t*)malloc(sizeof(size_t) * config->num_layers);
    if (!(*state)->weight_sizes) {
        free((*state)->fp32_weights);
        free(*state);
        return -1;
    }
    
    memset((*state)->fp32_weights, 0, sizeof(void*) * config->num_layers);
    memset((*state)->weight_sizes, 0, sizeof(size_t) * config->num_layers);
    
    return 0;
}

// 清理混合精度训练资源
void mixed_precision_cleanup(MixedPrecisionState* state) {
    if (!state) return;
    
    if (state->fp32_weights) {
        for (size_t i = 0; i < state->num_layers; i++) {
            if (state->fp32_weights[i]) {
                free(state->fp32_weights[i]);
            }
        }
        free(state->fp32_weights);
    }
    
    if (state->weight_sizes) {
        free(state->weight_sizes);
    }
    
    free(state);
}

// 前向传播中的精度转换
int mixed_precision_forward(MixedPrecisionState* state,
                          size_t layer_idx,
                          void* data,
                          size_t size,
                          const LayerPrecisionConfig* config) {
    if (!state || !data || !config || layer_idx >= state->num_layers) return -1;
    
    // 更新权重大小信息
    state->weight_sizes[layer_idx] = size;
    
    // 备份FP32权重（如果需要）
    if (config->weight_precision != PRECISION_FP32 && !state->fp32_weights[layer_idx]) {
        state->fp32_weights[layer_idx] = malloc(size * sizeof(float));
        if (!state->fp32_weights[layer_idx]) return -1;
        memcpy(state->fp32_weights[layer_idx], data, size * sizeof(float));
    }
    
    // 转换权重精度
    void* temp_buffer = malloc(size * sizeof(float));
    if (!temp_buffer) return -1;
    
    convert_precision(temp_buffer, data, size, PRECISION_FP32, config->weight_precision);
    memcpy(data, temp_buffer, size * sizeof(float));
    
    free(temp_buffer);
    return 0;
}

// 反向传播中的精度转换和梯度缩放
int mixed_precision_backward(MixedPrecisionState* state,
                           size_t layer_idx,
                           void* grad_data,
                           size_t size,
                           const LayerPrecisionConfig* config) {
    if (!state || !grad_data || !config || layer_idx >= state->num_layers) return -1;
    
    float* grad = (float*)grad_data;
    
    // 应用损失缩放
    for (size_t i = 0; i < size; i++) {
        grad[i] *= state->current_loss_scale;
    }
    
    // 检查梯度溢出
    int has_overflow = 0;
    for (size_t i = 0; i < size; i++) {
        if (!check_value_range(grad[i], config->grad_precision)) {
            has_overflow = 1;
            break;
        }
    }
    
    if (has_overflow) {
        state->overflow_count++;
        return 1;  // 表示发生溢出
    }
    
    // 转换梯度精度
    void* temp_buffer = malloc(size * sizeof(float));
    if (!temp_buffer) return -1;
    
    convert_precision(temp_buffer, grad_data, size, PRECISION_FP32, config->grad_precision);
    memcpy(grad_data, temp_buffer, size * sizeof(float));
    
    free(temp_buffer);
    return 0;
}

// 权重更新前的精度转换
int mixed_precision_pre_update(MixedPrecisionState* state,
                             size_t layer_idx,
                             void* weight_data,
                             void* grad_data,
                             size_t size,
                             const LayerPrecisionConfig* config) {
    if (!state || !weight_data || !grad_data || !config || layer_idx >= state->num_layers) return -1;
    
    // 如果存在FP32备份，使用备份进行更新
    if (state->fp32_weights[layer_idx]) {
        memcpy(weight_data, state->fp32_weights[layer_idx], size * sizeof(float));
    }
    
    // 转换梯度回FP32
    float* grad = (float*)grad_data;
    for (size_t i = 0; i < size; i++) {
        grad[i] /= state->current_loss_scale;  // 反向缩放梯度
    }
    
    return 0;
}

// 权重更新后的精度转换
int mixed_precision_post_update(MixedPrecisionState* state,
                              size_t layer_idx,
                              void* weight_data,
                              size_t size,
                              const LayerPrecisionConfig* config) {
    if (!state || !weight_data || !config || layer_idx >= state->num_layers) return -1;
    
    // 更新FP32备份
    if (state->fp32_weights[layer_idx]) {
        memcpy(state->fp32_weights[layer_idx], weight_data, size * sizeof(float));
    }
    
    // 转换权重到目标精度
    void* temp_buffer = malloc(size * sizeof(float));
    if (!temp_buffer) return -1;
    
    convert_precision(temp_buffer, weight_data, size, PRECISION_FP32, config->weight_precision);
    memcpy(weight_data, temp_buffer, size * sizeof(float));
    
    free(temp_buffer);
    return 0;
}

// 检查数值溢出
int mixed_precision_check_overflow(const void* data,
                                 size_t size,
                                 PrecisionType precision) {
    if (!data) return -1;
    
    const float* values = (const float*)data;
    for (size_t i = 0; i < size; i++) {
        if (!check_value_range(values[i], precision)) {
            return 1;  // 发生溢出
        }
    }
    
    return 0;
}

// 更新损失缩放因子
int mixed_precision_update_loss_scale(MixedPrecisionState* state,
                                    const MixedPrecisionConfig* config) {
    if (!state || !config) return -1;
    
    state->step_count++;
    
    if (config->dynamic_loss_scale && 
        state->step_count >= config->loss_scale_window) {
        
        float overflow_ratio = (float)state->overflow_count / state->step_count;
        
        if (overflow_ratio > config->overflow_threshold) {
            // 降低损失缩放因子
            state->current_loss_scale /= config->loss_scale_factor;
        } else if (state->overflow_count == 0) {
            // 提高损失缩放因子
            state->current_loss_scale *= config->loss_scale_factor;
        }
        
        // 重置计数器
        state->overflow_count = 0;
        state->step_count = 0;
    }
    
    return 0;
} 