#ifndef LOW_MEMORY_LLM_H
#define LOW_MEMORY_LLM_H

#include <stdint.h>
#include <stdlib.h>

// 下载进度回调函数类型
typedef void (*DownloadProgressCallback)(size_t downloaded, size_t total);

// 下载配置结构
typedef struct {
    const char* url;              // 模型权重的URL
    const char* save_path;        // 保存路径
    int verify_ssl;              // 是否验证SSL证书
    const char* proxy;           // 代理服务器地址（可选）
    int timeout_seconds;         // 超时时间（秒）
    DownloadProgressCallback progress_callback;  // 进度回调
} DownloadConfig;

// Hugging Face 下载配置
typedef struct {
    const char* repo_id;          // 例如 "facebook/opt-125m"
    const char* filename;         // 例如 "pytorch_model.bin"
    const char* save_path;        // 保存路径
    const char* token;           // HF API token (可选)
    int timeout_seconds;         // 超时时间
    DownloadProgressCallback progress_callback;
} HFDownloadConfig;

// 量化类型枚举
typedef enum {
    QUANT_NONE = 0,    // 无量化，FP32
    QUANT_INT8,        // INT8量化
    QUANT_INT4,        // INT4量化
    QUANT_INT2,        // INT2量化
} QuantType;

// 激活函数类型
typedef enum {
    ACT_NONE = 0,
    ACT_RELU,
    ACT_GELU,
    ACT_SILU,
    ACT_SWISH,
} ActivationType;

// 量化配置
typedef struct {
    QuantType quant_type;           // 量化类型
    float scale;                    // 量化比例
    float zero_point;               // 零点
    int symmetric;                  // 是否对称量化
    int per_channel;                // 是否按通道量化
} QuantConfig;

// 模型配置结构
typedef struct {
    size_t vocab_size;             // 词汇表大小
    size_t hidden_size;            // 隐藏层大小
    size_t num_layers;             // 层数
    size_t max_seq_length;         // 最大序列长度
    size_t batch_size;             // 批次大小
    QuantConfig quant_config;      // 量化配置
    ActivationType act_type;       // 激活函数类型
    char* model_path;              // 模型文件路径
    int use_cache;                 // 是否使用KV缓存
} LLMConfig;

// 内存管理结构
typedef struct {
    size_t available_ram;          // 可用内存大小
    size_t page_size;              // 页面大小
    int use_disk_offload;          // 是否使用磁盘卸载
    char* swap_file_path;          // 交换文件路径
    size_t prefetch_size;          // 预取大小
    int use_memory_map;            // 是否使用内存映射
} MemoryManager;

// 张量结构
typedef struct {
    void* data;                    // 数据指针
    size_t* shape;                 // 形状数组
    size_t ndim;                   // 维度数
    size_t size;                   // 总元素数
    QuantType quant_type;          // 量化类型
    float* scales;                 // 量化比例（按通道）
    float* zero_points;            // 零点（按通道）
    int is_view;                   // 是否是视图
} Tensor;

// 注意力缓存
typedef struct {
    Tensor* key_cache;             // Key缓存
    Tensor* value_cache;           // Value缓存
    size_t current_length;         // 当前长度
} AttentionCache;

// 模型状态结构
typedef struct {
    Tensor** weights;              // 模型权重数组
    Tensor* activations;           // 激活值
    AttentionCache* cache;         // KV缓存
    size_t current_position;       // 当前处理位置
    int is_initialized;            // 初始化标志
} LLMState;

// 内存块结构
typedef struct {
    void* data;                    // 数据指针
    size_t size;                   // 块大小
    int is_in_memory;             // 是否在内存中
    int64_t last_access;          // 最后访问时间
    int is_dirty;                 // 是否被修改
    char* swap_path;              // 交换文件路径
} MemoryBlock;

// 初始化函数
int llm_init(LLMConfig* config, MemoryManager* mem_manager);
void llm_cleanup(void);

// 推理相关函数
int llm_load_weights(const char* weights_file);
int llm_forward(const int* input_tokens, size_t input_length, float* output);
int llm_generate(const int* prompt_tokens, size_t prompt_length, 
                int* output_tokens, size_t max_length,
                float temperature, float top_p);

// 内存管理函数
MemoryBlock* memory_block_create(size_t size);
void memory_block_free(MemoryBlock* block);
int memory_block_load(MemoryBlock* block);
int memory_block_offload(MemoryBlock* block);
int memory_block_swap(MemoryBlock* block);

// 张量操作函数
Tensor* tensor_create(size_t* shape, size_t ndim, QuantType quant_type);
void tensor_free(Tensor* tensor);
int tensor_quantize(Tensor* tensor, QuantConfig* config);
int tensor_dequantize(Tensor* tensor);
Tensor* tensor_view(Tensor* tensor, size_t* new_shape, size_t new_ndim);

// 数学运算函数
int matrix_multiply(Tensor* a, Tensor* b, Tensor* c, MemoryManager* mem_manager);
int layer_norm(Tensor* input, Tensor* weight, Tensor* bias, float eps);
void activate(Tensor* input, ActivationType act_type);

// 注意力计算函数
int self_attention(Tensor* query, Tensor* key, Tensor* value, 
                  Tensor* output, AttentionCache* cache,
                  MemoryManager* mem_manager);

// 实用函数
void llm_get_memory_stats(size_t* used_memory, size_t* peak_memory);
int llm_memory_defrag(void);
const char* llm_get_error(void);

// 下载相关函数声明
int llm_download_weights(const DownloadConfig* config);
int llm_verify_weights(const char* weights_file, const char* expected_hash);
const char* llm_get_download_error(void);
int llm_download_from_hf(const HFDownloadConfig* config);

#endif // LOW_MEMORY_LLM_H 