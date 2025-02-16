#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdint.h>
#include <stddef.h>

// 分词器类型
typedef enum {
    TOKENIZER_BPE,           // Byte-Pair Encoding
    TOKENIZER_WORDPIECE,     // WordPiece
    TOKENIZER_UNIGRAM,       // Unigram
    TOKENIZER_SENTENCEPIECE  // SentencePiece
} TokenizerType;

// 分词器配置
typedef struct {
    TokenizerType type;           // 分词器类型
    const char* vocab_file;       // 词汇表文件路径
    const char* merges_file;      // 合并规则文件路径（BPE专用）
    int add_special_tokens;       // 是否添加特殊令牌
    int add_prefix_space;        // 是否添加前缀空格
    const char* unk_token;       // 未知令牌
    const char* pad_token;       // 填充令牌
    const char* bos_token;       // 句子开始令牌
    const char* eos_token;       // 句子结束令牌
    const char* mask_token;      // 掩码令牌
} TokenizerConfig;

// 分词器结构
typedef struct {
    TokenizerConfig config;      // 配置
    void* vocab;                 // 词汇表（实现细节隐藏）
    void* merges;               // 合并规则（BPE专用）
    size_t vocab_size;          // 词汇表大小
    int is_initialized;         // 是否已初始化
} Tokenizer;

// 初始化分词器
int tokenizer_init(Tokenizer** tokenizer, const TokenizerConfig* config);

// 释放分词器
void tokenizer_free(Tokenizer* tokenizer);

// 编码文本
int tokenizer_encode(Tokenizer* tokenizer,
                    const char* text,
                    int32_t* tokens,
                    size_t* num_tokens,
                    size_t max_length);

// 解码令牌
int tokenizer_decode(Tokenizer* tokenizer,
                    const int32_t* tokens,
                    size_t num_tokens,
                    char* text,
                    size_t* text_length);

// 获取词汇表大小
size_t tokenizer_get_vocab_size(const Tokenizer* tokenizer);

// 从令牌ID获取文本
const char* tokenizer_id_to_token(const Tokenizer* tokenizer, int32_t token_id);

// 从文本获取令牌ID
int32_t tokenizer_token_to_id(const Tokenizer* tokenizer, const char* token);

// 添加特殊令牌
int tokenizer_add_special_tokens(Tokenizer* tokenizer,
                               const char** tokens,
                               size_t num_tokens);

// 保存分词器
int tokenizer_save(const Tokenizer* tokenizer, const char* path);

// 加载分词器
int tokenizer_load(Tokenizer** tokenizer, const char* path);

#endif // TOKENIZER_H 