#ifndef LOW_MEMORY_LLM_H
#define LOW_MEMORY_LLM_H

#include <stddef.h>
#include <stdint.h>

// 虚拟内存页结构
typedef struct {
    void* data;            // 页面数据
    size_t size;          // 页面大小
    uint64_t page_id;     // 页面ID
    time_t last_access;   // 最后访问时间
    int is_dirty;         // 是否已修改
} VMPage;

// 权重缓存结构
typedef struct {
    VMPage pages[1024];   // 页面数组
    size_t active_pages;  // 活动页面数
    char cache_dir[256];  // 缓存目录
} WeightCache;

// 量化类型
typedef enum {
    QUANT_NONE = 0,    // 无量化
    QUANT_INT8,        // INT8量化
    QUANT_INT4,        // INT4量化
    QUANT_INT2         // INT2量化
} QuantType;

// 激活函数类型
typedef enum {
    ACT_NONE = 0,      // 无激活函数
    ACT_RELU,          // ReLU
    ACT_GELU,          // GELU
    ACT_SILU           // SiLU/Swish
} ActType;

// 张量结构
typedef struct {
    void* data;            // 数据指针
    size_t* shape;        // 维度大小数组
    size_t ndim;          // 维度数量
    size_t size;          // 总元素数量
    QuantType quant_type; // 量化类型
    float* scales;        // 量化比例因子
    float* zero_points;   // 量化零点
    int is_view;          // 是否是视图
} Tensor;

// 量化配置
typedef struct {
    QuantType quant_type;  // 量化类型
    float scale;           // 全局比例因子（可选）
    float zero_point;      // 全局零点（可选）
    int per_channel;       // 是否按通道量化
    int symmetric;         // 是否对称量化
} QuantConfig;

// 注意力缓存
typedef struct {
    float* key_cache;     // Key缓存
    float* value_cache;   // Value缓存
    size_t max_seq_len;   // 最大序列长度
    size_t head_dim;      // 注意力头维度
} AttentionCache;

// Transformer层权重
typedef struct {
    Tensor query_weight;  // 查询权重
    Tensor key_weight;    // 键权重
    Tensor value_weight;  // 值权重
    Tensor ffn_weight1;   // FFN第一层权重
    Tensor ffn_weight2;   // FFN第二层权重
    Tensor ffn_bias1;     // FFN第一层偏置
    Tensor ffn_bias2;     // FFN第二层偏置
} TransformerWeights;

// 模型配置
typedef struct {
    size_t vocab_size;     // 词表大小
    size_t hidden_size;    // 隐藏层大小
    size_t num_layers;     // 层数
    size_t num_heads;      // 注意力头数
    size_t ffn_hidden_size;// FFN隐藏层大小
    ActType act_type;      // 激活函数类型
    float dropout;         // Dropout比例
} LLMConfig;

// 模型状态
typedef struct {
    LLMConfig config;              // 模型配置
    TransformerWeights* weights;   // 层权重
    Tensor activations;            // 激活值
    AttentionCache* cache;         // KV缓存
    size_t current_position;       // 当前位置
    int is_initialized;            // 初始化标志
} LLMState;

// 内存管理器
typedef struct {
    size_t total_size;     // 总内存大小
    size_t used_size;      // 已使用内存
    void* base_ptr;        // 基础指针
} MemoryManager;

// 核心函数声明
int llm_init(LLMConfig* config, MemoryManager* mem_manager);
int llm_forward(const int* input_tokens, size_t input_length, float* output);
void llm_cleanup(void);

// 工具函数声明
void matrix_multiply(const float* a, const float* b, float* c,
                    size_t m, size_t k, size_t n);
void attention_forward(const float* query, const float* key, const float* value,
                      float* output, size_t seq_len, size_t head_dim, float scale);
void ffn_forward(const float* input, const float* weight1, const float* weight2,
                const float* bias1, const float* bias2, float* output,
                size_t hidden_size, size_t ffn_size);

// 内存管理函数声明
int init_weight_cache(void);
int read_weight_data(void* dest, size_t offset, size_t size);
int write_weight_data(const void* src, size_t offset, size_t size);

// 张量操作函数
Tensor* tensor_create(size_t* shape, size_t ndim, QuantType quant_type);
void tensor_free(Tensor* tensor);
Tensor* tensor_view(Tensor* tensor, size_t* new_shape, size_t new_ndim);
int tensor_quantize(Tensor* tensor, QuantConfig* config);
int tensor_dequantize(Tensor* tensor);

#endif // LOW_MEMORY_LLM_H 