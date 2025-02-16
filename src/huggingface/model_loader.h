#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include <stdint.h>
#include <stddef.h>

// 模型配置
typedef struct {
    const char* model_name;          // 模型名称
    const char* revision;            // 模型版本
    const char* cache_dir;           // 本地缓存目录
    int use_auth_token;             // 是否使用认证令牌
    const char* auth_token;         // 认证令牌
    int trust_remote_code;          // 是否信任远程代码
} ModelConfig;

// 模型权重格式
typedef enum {
    WEIGHT_FORMAT_FP32,
    WEIGHT_FORMAT_FP16,
    WEIGHT_FORMAT_INT8,
    WEIGHT_FORMAT_INT4
} WeightFormat;

// 模型层类型
typedef enum {
    LAYER_TYPE_EMBEDDING,
    LAYER_TYPE_ATTENTION,
    LAYER_TYPE_MLP,
    LAYER_TYPE_LAYERNORM,
    LAYER_TYPE_OUTPUT
} LayerType;

// 模型层参数
typedef struct {
    LayerType type;                 // 层类型
    void* weights;                  // 权重数据
    void* bias;                     // 偏置数据
    size_t* shape;                  // 形状数组
    size_t num_dims;               // 维度数量
    WeightFormat format;           // 权重格式
} LayerParams;

// 模型结构
typedef struct {
    ModelConfig config;             // 模型配置
    LayerParams** layers;           // 层参数数组
    size_t num_layers;             // 层数量
    void* tokenizer;               // 分词器
    void* device;                  // 设备指针
} HFModel;

// 初始化模型加载器
int hf_model_init(void);

// 清理模型加载器
void hf_model_cleanup(void);

// 从Hugging Face Hub下载模型
int hf_model_download(const ModelConfig* config);

// 加载模型
int hf_model_load(HFModel** model, const ModelConfig* config, void* device);

// 释放模型
void hf_model_free(HFModel* model);

// 转换模型格式
int hf_model_convert(HFModel* model, WeightFormat target_format);

// 保存模型到本地
int hf_model_save(const HFModel* model, const char* path);

// 加载本地模型
int hf_model_load_local(HFModel** model, const char* path, void* device);

// 获取模型信息
const char* hf_model_get_info(const HFModel* model);

// 验证模型完整性
int hf_model_verify(const HFModel* model);

// 获取层参数
const LayerParams* hf_model_get_layer(const HFModel* model, size_t layer_idx);

// 获取分词器
void* hf_model_get_tokenizer(const HFModel* model);

#endif // MODEL_LOADER_H 