#ifndef QUANTIZATION_H
#define QUANTIZATION_H

#include <stdint.h>
#include <stddef.h>

// 量化类型
typedef enum {
    QUANT_TYPE_INT8,      // INT8量化
    QUANT_TYPE_INT4,      // INT4量化
    QUANT_TYPE_FP16,      // FP16量化
    QUANT_TYPE_FP8,       // FP8量化
    QUANT_TYPE_DYNAMIC    // 动态量化
} QuantType;

// 量化参数
typedef struct {
    float scale;          // 量化比例
    int32_t zero_point;   // 零点
    float min_value;      // 最小值
    float max_value;      // 最大值
} QuantParams;

// 量化配置
typedef struct {
    QuantType type;       // 量化类型
    int per_channel;      // 是否按通道量化
    int symmetric;        // 是否对称量化
    float clip_ratio;     // 裁剪比例
} QuantConfig;

// 初始化量化参数
int quant_init_params(QuantParams* params, const float* data, size_t size, const QuantConfig* config);

// 量化数据
int quant_quantize(void* output, const float* input, size_t size, 
                  const QuantParams* params, const QuantConfig* config);

// 反量化数据
int quant_dequantize(float* output, const void* input, size_t size,
                    const QuantParams* params, const QuantConfig* config);

// 计算量化参数
int quant_calibrate(QuantParams* params, const float* data, size_t size,
                   const QuantConfig* config);

// 获取量化类型的位宽
int quant_get_bitwidth(QuantType type);

// 获取量化后的数据大小
size_t quant_get_size(size_t num_elements, QuantType type);

#endif // QUANTIZATION_H 