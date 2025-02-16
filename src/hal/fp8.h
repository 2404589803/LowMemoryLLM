#ifndef FP8_H
#define FP8_H

#include <stdint.h>

// FP8格式类型
typedef enum {
    FP8_E4M3,    // 4位指数3位尾数（适用于权重）
    FP8_E5M2     // 5位指数2位尾数（适用于激活值）
} FP8Format;

// FP8数据结构
typedef struct {
    uint8_t bits;  // 8位存储
} FP8;

// 转换函数
FP8 float_to_fp8(float value, FP8Format format);
float fp8_to_float(FP8 value, FP8Format format);

// 数学运算
FP8 fp8_add(FP8 a, FP8 b, FP8Format format);
FP8 fp8_multiply(FP8 a, FP8 b, FP8Format format);

// 辅助函数
int fp8_is_nan(FP8 value);
int fp8_is_inf(FP8 value);
FP8 fp8_abs(FP8 value);

#endif // FP8_H 