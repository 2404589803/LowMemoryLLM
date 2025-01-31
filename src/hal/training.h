#ifndef TRAINING_H
#define TRAINING_H

#include "hal.h"
#include <stdint.h>

// 优化器类型
typedef enum {
    OPTIMIZER_SGD,
    OPTIMIZER_ADAM,
    OPTIMIZER_ADAMW,
    OPTIMIZER_RMSPROP
} OptimizerType;

// 损失函数类型
typedef enum {
    LOSS_MSE,
    LOSS_CROSS_ENTROPY,
    LOSS_BINARY_CROSS_ENTROPY
} LossType;

// 优化器配置
typedef struct {
    OptimizerType type;
    float learning_rate;
    float beta1;        // For Adam/AdamW
    float beta2;        // For Adam/AdamW
    float weight_decay; // For AdamW
    float momentum;     // For SGD with momentum
    float epsilon;      // 数值稳定性
} OptimizerConfig;

// 训练配置
typedef struct {
    size_t batch_size;
    size_t num_epochs;
    LossType loss_type;
    OptimizerConfig optimizer;
    float gradient_clip_norm;
    int enable_mixed_precision;
} TrainingConfig;

// 训练状态
typedef struct {
    size_t current_epoch;
    size_t current_batch;
    float current_loss;
    float current_accuracy;
    float learning_rate;
    void* optimizer_state;
} TrainingState;

// 反向传播所需的设备功能扩展
typedef struct {
    // 基础计算函数
    void (*backward_matrix_multiply)(const void* grad_output, const void* input,
                                   void* grad_input, void* grad_weight,
                                   size_t m, size_t n, size_t k);
    
    void (*backward_vector_add)(const void* grad_output, void* grad_input,
                              void* grad_bias, size_t size);
    
    // 激活函数及其导数
    void (*forward_activation)(void* output, const void* input,
                             size_t size, const char* type);
    void (*backward_activation)(void* grad_input, const void* grad_output,
                              const void* output, size_t size, const char* type);
    
    // 损失函数及其导数
    float (*compute_loss)(const void* predictions, const void* targets,
                         size_t size, LossType type);
    void (*backward_loss)(void* grad_output, const void* predictions,
                         const void* targets, size_t size, LossType type);
    
    // 优化器相关操作
    void (*optimizer_step)(void* params, void* grad,
                          void* optimizer_state, const OptimizerConfig* config,
                          size_t size);
    
    // 梯度裁剪
    void (*clip_gradients)(void* grads, size_t size, float max_norm);
    
    // 混合精度训练支持
    void (*cast_to_fp16)(void* output, const void* input, size_t size);
    void (*cast_to_fp32)(void* output, const void* input, size_t size);
} TrainingExtension;

// 训练回调函数
typedef struct {
    void (*on_epoch_begin)(TrainingState* state);
    void (*on_epoch_end)(TrainingState* state);
    void (*on_batch_begin)(TrainingState* state);
    void (*on_batch_end)(TrainingState* state);
    void (*on_backward_begin)(TrainingState* state);
    void (*on_backward_end)(TrainingState* state);
} TrainingCallbacks;

// 初始化训练扩展
int training_init(HAL_Device* device, TrainingExtension* extension);

// 配置训练参数
int training_configure(TrainingConfig* config);

// 训练一个批次
int training_step(void* model, void* inputs, void* targets,
                 TrainingState* state, TrainingCallbacks* callbacks);

// 评估模型
int training_evaluate(void* model, void* inputs, void* targets,
                     float* metrics, size_t num_metrics);

// 清理训练资源
void training_cleanup(void);

#endif // TRAINING_H 