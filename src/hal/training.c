#include "training.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// 全局训练配置
static TrainingConfig* g_config = NULL;
static TrainingExtension* g_extension = NULL;
static HAL_Device* g_device = NULL;

// 优化器状态
typedef struct {
    void* m;           // 一阶矩
    void* v;           // 二阶矩
    size_t iter;       // 迭代次数
    float beta1_t;     // beta1^t
    float beta2_t;     // beta2^t
} AdamState;

// 初始化训练扩展
int training_init(HAL_Device* device, TrainingExtension* extension) {
    if (!device || !extension) return -1;
    
    g_device = device;
    g_extension = extension;
    
    return 0;
}

// 配置训练参数
int training_configure(TrainingConfig* config) {
    if (!config) return -1;
    
    // 分配并复制配置
    if (!g_config) {
        g_config = (TrainingConfig*)malloc(sizeof(TrainingConfig));
        if (!g_config) return -1;
    }
    
    memcpy(g_config, config, sizeof(TrainingConfig));
    
    return 0;
}

// 创建优化器状态
static void* create_optimizer_state(size_t param_size) {
    if (g_config->optimizer.type == OPTIMIZER_ADAM ||
        g_config->optimizer.type == OPTIMIZER_ADAMW) {
        AdamState* state = (AdamState*)malloc(sizeof(AdamState));
        if (!state) return NULL;
        
        state->m = g_device->allocate_memory(param_size);
        state->v = g_device->allocate_memory(param_size);
        if (!state->m || !state->v) {
            if (state->m) g_device->free_memory(state->m);
            if (state->v) g_device->free_memory(state->v);
            free(state);
            return NULL;
        }
        
        // 初始化为0
        memset(state->m, 0, param_size);
        memset(state->v, 0, param_size);
        state->iter = 0;
        state->beta1_t = 1.0f;
        state->beta2_t = 1.0f;
        
        return state;
    }
    
    return NULL;
}

// 释放优化器状态
static void free_optimizer_state(void* state) {
    if (!state) return;
    
    if (g_config->optimizer.type == OPTIMIZER_ADAM ||
        g_config->optimizer.type == OPTIMIZER_ADAMW) {
        AdamState* adam_state = (AdamState*)state;
        if (adam_state->m) g_device->free_memory(adam_state->m);
        if (adam_state->v) g_device->free_memory(adam_state->v);
        free(adam_state);
    }
}

// 执行优化器步骤
static void optimizer_step(void* params, void* grads, void* state,
                         size_t param_size) {
    if (g_config->optimizer.type == OPTIMIZER_ADAM ||
        g_config->optimizer.type == OPTIMIZER_ADAMW) {
        AdamState* adam_state = (AdamState*)state;
        
        // 更新迭代次数和衰减因子
        adam_state->iter++;
        adam_state->beta1_t *= g_config->optimizer.beta1;
        adam_state->beta2_t *= g_config->optimizer.beta2;
        
        // 计算学习率
        float lr = g_config->optimizer.learning_rate *
                  sqrt(1.0f - adam_state->beta2_t) /
                  (1.0f - adam_state->beta1_t);
        
        // 调用设备实现的优化器步骤
        g_extension->optimizer_step(params, grads, adam_state->m, adam_state->v,
                                  g_config->optimizer.beta1,
                                  g_config->optimizer.beta2,
                                  lr,
                                  g_config->optimizer.epsilon,
                                  param_size);
    }
}

// 前向传播
static int forward_pass(void* model, void* inputs, void* predictions,
                       size_t batch_size, TrainingCallbacks* callbacks) {
    if (!model || !inputs || !predictions) return -1;
    
    // 获取模型参数
    float* weights = (float*)model;
    float* input_data = (float*)inputs;
    float* output_data = (float*)predictions;
    
    // 分配中间结果缓冲区
    void* layer_output = g_device->allocate_memory(batch_size * sizeof(float));
    if (!layer_output) return -1;
    
    // 线性层前向传播
    g_extension->forward_matrix_multiply(weights, input_data, layer_output,
                                       batch_size, g_config->hidden_size,
                                       g_config->input_size);
    
    // 应用激活函数
    g_extension->forward_activation(output_data, layer_output,
                                  batch_size * g_config->hidden_size,
                                  "relu");
    
    // 释放中间缓冲区
    g_device->free_memory(layer_output);
    
    return 0;
}

// 训练一个批次
int training_step(void* model, void* inputs, void* targets,
                 TrainingState* state, TrainingCallbacks* callbacks) {
    if (!model || !inputs || !targets || !state || !g_config || !g_extension) {
        return -1;
    }
    
    // 前向传播
    if (callbacks && callbacks->on_batch_begin) {
        callbacks->on_batch_begin(state);
    }
    
    void* predictions = g_device->allocate_memory(g_config->batch_size * sizeof(float));
    if (!predictions) return -1;
    
    if (forward_pass(model, inputs, predictions, g_config->batch_size, callbacks) != 0) {
        g_device->free_memory(predictions);
        return -1;
    }
    
    // 计算损失
    state->current_loss = g_extension->compute_loss(predictions, targets,
                                                  g_config->batch_size,
                                                  g_config->loss_type);
    
    // 反向传播
    if (callbacks && callbacks->on_backward_begin) {
        callbacks->on_backward_begin(state);
    }
    
    void* grad_output = g_device->allocate_memory(g_config->batch_size * sizeof(float));
    if (!grad_output) {
        g_device->free_memory(predictions);
        return -1;
    }
    
    // 计算损失函数梯度
    g_extension->backward_loss(grad_output, predictions, targets,
                             g_config->batch_size, g_config->loss_type);
    
    // TODO: 实现反向传播
    
    // 梯度裁剪
    if (g_config->gradient_clip_norm > 0) {
        g_extension->clip_gradients(grad_output, g_config->batch_size,
                                  g_config->gradient_clip_norm);
    }
    
    // 优化器步骤
    optimizer_step(model, grad_output, state->optimizer_state,
                  g_config->batch_size * sizeof(float));
    
    if (callbacks && callbacks->on_backward_end) {
        callbacks->on_backward_end(state);
    }
    
    // 清理
    g_device->free_memory(predictions);
    g_device->free_memory(grad_output);
    
    if (callbacks && callbacks->on_batch_end) {
        callbacks->on_batch_end(state);
    }
    
    return 0;
}

// 评估模型
int training_evaluate(void* model, void* inputs, void* targets,
                     float* metrics, size_t num_metrics) {
    if (!model || !inputs || !targets || !metrics || !g_config || !g_extension) {
        return -1;
    }
    
    void* predictions = g_device->allocate_memory(g_config->batch_size * sizeof(float));
    if (!predictions) return -1;
    
    // TODO: 实现前向传播进行评估
    
    // 计算评估指标
    metrics[0] = g_extension->compute_loss(predictions, targets,
                                         g_config->batch_size,
                                         g_config->loss_type);
    
    // TODO: 计算其他评估指标
    
    g_device->free_memory(predictions);
    return 0;
}

// 清理训练资源
void training_cleanup(void) {
    if (g_config) {
        free(g_config);
        g_config = NULL;
    }
    
    g_extension = NULL;
    g_device = NULL;
} 