#ifndef MIXED_PRECISION_H
#define MIXED_PRECISION_H

#include "quantization.h"
#include "qat.h"

// 精度策略
typedef enum {
    PRECISION_FP32,     // 全精度FP32
    PRECISION_FP16,     // 半精度FP16
    PRECISION_FP8,      // 8位浮点
    PRECISION_INT8,     // 8位定点
    PRECISION_DYNAMIC   // 动态精度
} PrecisionType;

// 层精度配置
typedef struct {
    PrecisionType weight_precision;    // 权重精度
    PrecisionType grad_precision;      // 梯度精度
    PrecisionType activation_precision; // 激活值精度
    PrecisionType momentum_precision;   // 动量精度
    float loss_scale;                  // 损失缩放因子
} LayerPrecisionConfig;

// 混合精度训练配置
typedef struct {
    LayerPrecisionConfig* layer_configs;  // 每层的精度配置
    size_t num_layers;                    // 层数
    float init_loss_scale;                // 初始损失缩放因子
    float loss_scale_factor;              // 损失缩放调整因子
    int loss_scale_window;               // 损失缩放更新窗口
    float overflow_threshold;             // 溢出阈值
    int dynamic_loss_scale;              // 是否使用动态损失缩放
} MixedPrecisionConfig;

// 混合精度状态
typedef struct {
    float current_loss_scale;            // 当前损失缩放因子
    int overflow_count;                  // 溢出计数
    int step_count;                      // 步数计数
    void** fp32_weights;                // FP32权重备份
    size_t* weight_sizes;               // 每层权重的大小
    QATState* qat_state;                // QAT状态（如果使用）
    size_t num_layers;                   // 层数
} MixedPrecisionState;

// 初始化混合精度训练
int mixed_precision_init(MixedPrecisionState** state, 
                        const MixedPrecisionConfig* config,
                        QATState* qat_state);

// 清理混合精度训练资源
void mixed_precision_cleanup(MixedPrecisionState* state);

// 前向传播中的精度转换
int mixed_precision_forward(MixedPrecisionState* state,
                          size_t layer_idx,
                          void* data,
                          size_t size,
                          const LayerPrecisionConfig* config);

// 反向传播中的精度转换和梯度缩放
int mixed_precision_backward(MixedPrecisionState* state,
                           size_t layer_idx,
                           void* grad_data,
                           size_t size,
                           const LayerPrecisionConfig* config);

// 权重更新前的精度转换
int mixed_precision_pre_update(MixedPrecisionState* state,
                             size_t layer_idx,
                             void* weight_data,
                             void* grad_data,
                             size_t size,
                             const LayerPrecisionConfig* config);

// 权重更新后的精度转换
int mixed_precision_post_update(MixedPrecisionState* state,
                              size_t layer_idx,
                              void* weight_data,
                              size_t size,
                              const LayerPrecisionConfig* config);

// 检查数值溢出
int mixed_precision_check_overflow(const void* data,
                                 size_t size,
                                 PrecisionType precision);

// 更新损失缩放因子
int mixed_precision_update_loss_scale(MixedPrecisionState* state,
                                    const MixedPrecisionConfig* config);

// 保存格式版本
#define MP_STATE_VERSION 1

// 保存选项
typedef struct {
    int use_compression;          // 是否使用压缩
    int compression_level;        // 压缩级别(0-9)
    int verify_checksum;         // 是否验证校验和
} SaveOptions;

// 默认保存选项
#define DEFAULT_SAVE_OPTIONS { \
    .use_compression = 1,     \
    .compression_level = 6,   \
    .verify_checksum = 1      \
}

// 带选项的保存函数
int mixed_precision_save_state_with_options(const MixedPrecisionState* state,
                                          const char* path,
                                          const SaveOptions* options);

// 带选项的加载函数
int mixed_precision_load_state_with_options(MixedPrecisionState* state,
                                          const char* path,
                                          const SaveOptions* options);

// 验证保存文件的完整性
int mixed_precision_verify_state_file(const char* path);

#endif // MIXED_PRECISION_H 