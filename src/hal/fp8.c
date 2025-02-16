#include "fp8.h"
#include <math.h>

// 常量定义
#define FP8_E4M3_BIAS 7
#define FP8_E5M2_BIAS 15
#define FP8_E4M3_MAX_EXP 8
#define FP8_E5M2_MAX_EXP 16
#define FP8_E4M3_INFINITY 0x7F
#define FP8_E5M2_INFINITY 0x7F
#define FP8_E4M3_NAN 0x7F
#define FP8_E5M2_NAN 0x7F

// 内部辅助函数
static uint8_t pack_e4m3(int sign, int exp, int mant) {
    return (sign << 7) | ((exp & 0xF) << 3) | (mant & 0x7);
}

static uint8_t pack_e5m2(int sign, int exp, int mant) {
    return (sign << 7) | ((exp & 0x1F) << 2) | (mant & 0x3);
}

static void unpack_e4m3(uint8_t bits, int* sign, int* exp, int* mant) {
    *sign = (bits >> 7) & 0x1;
    *exp = (bits >> 3) & 0xF;
    *mant = bits & 0x7;
}

static void unpack_e5m2(uint8_t bits, int* sign, int* exp, int* mant) {
    *sign = (bits >> 7) & 0x1;
    *exp = (bits >> 2) & 0x1F;
    *mant = bits & 0x3;
}

// 转换函数实现
FP8 float_to_fp8(float value, FP8Format format) {
    FP8 result;
    
    // 处理特殊值
    if (isnan(value)) {
        result.bits = (format == FP8_E4M3) ? FP8_E4M3_NAN : FP8_E5M2_NAN;
        return result;
    }
    
    // 提取符号位
    int sign = (value < 0) ? 1 : 0;
    value = fabsf(value);
    
    // 处理无穷大
    if (isinf(value)) {
        result.bits = (format == FP8_E4M3) ? FP8_E4M3_INFINITY : FP8_E5M2_INFINITY;
        if (sign) result.bits |= 0x80;
        return result;
    }
    
    if (format == FP8_E4M3) {
        // E4M3格式
        if (value == 0.0f) {
            result.bits = 0;
            return result;
        }
        
        int exp;
        float mant = frexpf(value, &exp);
        exp += FP8_E4M3_BIAS - 1;
        mant = mant * 16.0f - 8.0f;
        
        if (exp < -FP8_E4M3_BIAS) {
            // 下溢出到0
            result.bits = 0;
        } else if (exp > FP8_E4M3_MAX_EXP) {
            // 上溢出到无穷大
            result.bits = FP8_E4M3_INFINITY;
        } else {
            int mant_int = (int)(roundf(mant));
            if (mant_int == 8) {
                exp++;
                mant_int = 0;
            }
            result.bits = pack_e4m3(sign, exp, mant_int);
        }
    } else {
        // E5M2格式
        if (value == 0.0f) {
            result.bits = 0;
            return result;
        }
        
        int exp;
        float mant = frexpf(value, &exp);
        exp += FP8_E5M2_BIAS - 1;
        mant = mant * 8.0f - 4.0f;
        
        if (exp < -FP8_E5M2_BIAS) {
            // 下溢出到0
            result.bits = 0;
        } else if (exp > FP8_E5M2_MAX_EXP) {
            // 上溢出到无穷大
            result.bits = FP8_E5M2_INFINITY;
        } else {
            int mant_int = (int)(roundf(mant));
            if (mant_int == 4) {
                exp++;
                mant_int = 0;
            }
            result.bits = pack_e5m2(sign, exp, mant_int);
        }
    }
    
    return result;
}

float fp8_to_float(FP8 value, FP8Format format) {
    int sign, exp, mant;
    
    if (format == FP8_E4M3) {
        unpack_e4m3(value.bits, &sign, &exp, &mant);
        
        // 处理特殊值
        if (value.bits == FP8_E4M3_NAN) return NAN;
        if (value.bits == FP8_E4M3_INFINITY) return INFINITY;
        if (value.bits == 0) return 0.0f;
        
        float result = (mant + 8.0f) / 16.0f;
        result = ldexpf(result, exp - FP8_E4M3_BIAS + 1);
        return sign ? -result : result;
    } else {
        unpack_e5m2(value.bits, &sign, &exp, &mant);
        
        // 处理特殊值
        if (value.bits == FP8_E5M2_NAN) return NAN;
        if (value.bits == FP8_E5M2_INFINITY) return INFINITY;
        if (value.bits == 0) return 0.0f;
        
        float result = (mant + 4.0f) / 8.0f;
        result = ldexpf(result, exp - FP8_E5M2_BIAS + 1);
        return sign ? -result : result;
    }
}

// 数学运算实现
FP8 fp8_add(FP8 a, FP8 b, FP8Format format) {
    float fa = fp8_to_float(a, format);
    float fb = fp8_to_float(b, format);
    return float_to_fp8(fa + fb, format);
}

FP8 fp8_multiply(FP8 a, FP8 b, FP8Format format) {
    float fa = fp8_to_float(a, format);
    float fb = fp8_to_float(b, format);
    return float_to_fp8(fa * fb, format);
}

// 辅助函数实现
int fp8_is_nan(FP8 value) {
    return (value.bits & 0x7F) == 0x7F;
}

int fp8_is_inf(FP8 value) {
    return ((value.bits & 0x7F) == 0x7F) && !fp8_is_nan(value);
}

FP8 fp8_abs(FP8 value) {
    value.bits &= 0x7F;
    return value;
} 