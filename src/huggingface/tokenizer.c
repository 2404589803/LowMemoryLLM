#include "tokenizer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

// 词汇表节点
typedef struct VocabNode {
    char* token;
    int32_t id;
    struct VocabNode* next;
} VocabNode;

// 词汇表哈希表
typedef struct {
    VocabNode** buckets;
    size_t size;
    size_t capacity;
} VocabHashTable;

// BPE合并规则
typedef struct {
    char* pair;
    int priority;
} BPEMerge;

// 创建词汇表哈希表
static VocabHashTable* create_vocab_table(size_t capacity) {
    VocabHashTable* table = (VocabHashTable*)malloc(sizeof(VocabHashTable));
    if (!table) return NULL;
    
    table->buckets = (VocabNode**)calloc(capacity, sizeof(VocabNode*));
    if (!table->buckets) {
        free(table);
        return NULL;
    }
    
    table->capacity = capacity;
    table->size = 0;
    return table;
}

// 计算字符串哈希值
static size_t hash_string(const char* str) {
    size_t hash = 5381;
    int c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash;
}

// 添加词汇到哈希表
static int add_to_vocab(VocabHashTable* table, const char* token, int32_t id) {
    size_t index = hash_string(token) % table->capacity;
    
    // 检查是否已存在
    VocabNode* node = table->buckets[index];
    while (node) {
        if (strcmp(node->token, token) == 0) {
            return -1;  // 已存在
        }
        node = node->next;
    }
    
    // 创建新节点
    node = (VocabNode*)malloc(sizeof(VocabNode));
    if (!node) return -1;
    
    node->token = strdup(token);
    if (!node->token) {
        free(node);
        return -1;
    }
    
    node->id = id;
    node->next = table->buckets[index];
    table->buckets[index] = node;
    table->size++;
    
    return 0;
}

// 从词汇表中查找令牌ID
static int32_t find_token_id(const VocabHashTable* table, const char* token) {
    size_t index = hash_string(token) % table->capacity;
    VocabNode* node = table->buckets[index];
    
    while (node) {
        if (strcmp(node->token, token) == 0) {
            return node->id;
        }
        node = node->next;
    }
    
    return -1;
}

// 从词汇表中查找令牌
static const char* find_token(const VocabHashTable* table, int32_t id) {
    for (size_t i = 0; i < table->capacity; i++) {
        VocabNode* node = table->buckets[i];
        while (node) {
            if (node->id == id) {
                return node->token;
            }
            node = node->next;
        }
    }
    
    return NULL;
}

// 释放词汇表
static void free_vocab_table(VocabHashTable* table) {
    if (!table) return;
    
    for (size_t i = 0; i < table->capacity; i++) {
        VocabNode* node = table->buckets[i];
        while (node) {
            VocabNode* next = node->next;
            free(node->token);
            free(node);
            node = next;
        }
    }
    
    free(table->buckets);
    free(table);
}

// 加载词汇表文件
static int load_vocab_file(VocabHashTable* table, const char* vocab_file) {
    FILE* fp = fopen(vocab_file, "r");
    if (!fp) return -1;
    
    char line[1024];
    int32_t id = 0;
    
    while (fgets(line, sizeof(line), fp)) {
        // 移除换行符
        size_t len = strlen(line);
        if (len > 0 && line[len-1] == '\n') {
            line[len-1] = '\0';
        }
        
        // 跳过空行
        if (len <= 1) continue;
        
        // 添加到词汇表
        if (add_to_vocab(table, line, id++) != 0) {
            fclose(fp);
            return -1;
        }
    }
    
    fclose(fp);
    return 0;
}

// 初始化分词器
int tokenizer_init(Tokenizer** tokenizer, const TokenizerConfig* config) {
    if (!tokenizer || !config || !config->vocab_file) return -1;
    
    *tokenizer = (Tokenizer*)malloc(sizeof(Tokenizer));
    if (!*tokenizer) return -1;
    
    // 复制配置
    memcpy(&(*tokenizer)->config, config, sizeof(TokenizerConfig));
    (*tokenizer)->config.vocab_file = strdup(config->vocab_file);
    if (config->merges_file)
        (*tokenizer)->config.merges_file = strdup(config->merges_file);
    if (config->unk_token)
        (*tokenizer)->config.unk_token = strdup(config->unk_token);
    if (config->pad_token)
        (*tokenizer)->config.pad_token = strdup(config->pad_token);
    if (config->bos_token)
        (*tokenizer)->config.bos_token = strdup(config->bos_token);
    if (config->eos_token)
        (*tokenizer)->config.eos_token = strdup(config->eos_token);
    if (config->mask_token)
        (*tokenizer)->config.mask_token = strdup(config->mask_token);
    
    // 创建词汇表
    (*tokenizer)->vocab = create_vocab_table(65536);  // 使用合适的初始容量
    if (!(*tokenizer)->vocab) {
        tokenizer_free(*tokenizer);
        return -1;
    }
    
    // 加载词汇表
    if (load_vocab_file((*tokenizer)->vocab, config->vocab_file) != 0) {
        tokenizer_free(*tokenizer);
        return -1;
    }
    
    // 加载BPE合并规则（如果需要）
    if (config->type == TOKENIZER_BPE && config->merges_file) {
        // TODO: 实现BPE合并规则加载
    }
    
    (*tokenizer)->vocab_size = ((VocabHashTable*)(*tokenizer)->vocab)->size;
    (*tokenizer)->is_initialized = 1;
    
    return 0;
}

// 释放分词器
void tokenizer_free(Tokenizer* tokenizer) {
    if (!tokenizer) return;
    
    // 释放配置字符串
    if (tokenizer->config.vocab_file)
        free((void*)tokenizer->config.vocab_file);
    if (tokenizer->config.merges_file)
        free((void*)tokenizer->config.merges_file);
    if (tokenizer->config.unk_token)
        free((void*)tokenizer->config.unk_token);
    if (tokenizer->config.pad_token)
        free((void*)tokenizer->config.pad_token);
    if (tokenizer->config.bos_token)
        free((void*)tokenizer->config.bos_token);
    if (tokenizer->config.eos_token)
        free((void*)tokenizer->config.eos_token);
    if (tokenizer->config.mask_token)
        free((void*)tokenizer->config.mask_token);
    
    // 释放词汇表
    if (tokenizer->vocab)
        free_vocab_table(tokenizer->vocab);
    
    // 释放BPE合并规则
    if (tokenizer->merges) {
        // TODO: 实现BPE合并规则释放
    }
    
    free(tokenizer);
}

// 编码文本
int tokenizer_encode(Tokenizer* tokenizer,
                    const char* text,
                    int32_t* tokens,
                    size_t* num_tokens,
                    size_t max_length) {
    if (!tokenizer || !text || !tokens || !num_tokens) return -1;
    
    *num_tokens = 0;
    
    // 添加BOS令牌
    if (tokenizer->config.add_special_tokens && tokenizer->config.bos_token) {
        int32_t bos_id = tokenizer_token_to_id(tokenizer, tokenizer->config.bos_token);
        if (bos_id >= 0 && *num_tokens < max_length) {
            tokens[(*num_tokens)++] = bos_id;
        }
    }
    
    // 根据分词器类型进行分词
    switch (tokenizer->config.type) {
        case TOKENIZER_BPE:
            // TODO: 实现BPE分词
            break;
            
        case TOKENIZER_WORDPIECE:
            // 简单的按空格分词示例
            char* text_copy = strdup(text);
            char* token = strtok(text_copy, " ");
            
            while (token && *num_tokens < max_length) {
                int32_t id = tokenizer_token_to_id(tokenizer, token);
                if (id >= 0) {
                    tokens[(*num_tokens)++] = id;
                } else if (tokenizer->config.unk_token) {
                    // 使用未知令牌
                    id = tokenizer_token_to_id(tokenizer, tokenizer->config.unk_token);
                    if (id >= 0) {
                        tokens[(*num_tokens)++] = id;
                    }
                }
                token = strtok(NULL, " ");
            }
            
            free(text_copy);
            break;
            
        case TOKENIZER_UNIGRAM:
            // TODO: 实现Unigram分词
            break;
            
        case TOKENIZER_SENTENCEPIECE:
            // TODO: 实现SentencePiece分词
            break;
    }
    
    // 添加EOS令牌
    if (tokenizer->config.add_special_tokens && tokenizer->config.eos_token) {
        int32_t eos_id = tokenizer_token_to_id(tokenizer, tokenizer->config.eos_token);
        if (eos_id >= 0 && *num_tokens < max_length) {
            tokens[(*num_tokens)++] = eos_id;
        }
    }
    
    return 0;
}

// 解码令牌
int tokenizer_decode(Tokenizer* tokenizer,
                    const int32_t* tokens,
                    size_t num_tokens,
                    char* text,
                    size_t* text_length) {
    if (!tokenizer || !tokens || !text || !text_length) return -1;
    
    size_t offset = 0;
    *text_length = 0;
    
    for (size_t i = 0; i < num_tokens; i++) {
        // 跳过特殊令牌
        if (tokenizer->config.add_special_tokens) {
            if (tokenizer->config.bos_token) {
                int32_t bos_id = tokenizer_token_to_id(tokenizer, tokenizer->config.bos_token);
                if (tokens[i] == bos_id) continue;
            }
            if (tokenizer->config.eos_token) {
                int32_t eos_id = tokenizer_token_to_id(tokenizer, tokenizer->config.eos_token);
                if (tokens[i] == eos_id) continue;
            }
        }
        
        // 获取令牌文本
        const char* token = tokenizer_id_to_token(tokenizer, tokens[i]);
        if (!token) continue;
        
        // 添加空格（如果需要）
        if (offset > 0 && tokenizer->config.add_prefix_space) {
            text[offset++] = ' ';
        }
        
        // 复制令牌文本
        size_t token_len = strlen(token);
        memcpy(text + offset, token, token_len);
        offset += token_len;
    }
    
    text[offset] = '\0';
    *text_length = offset;
    
    return 0;
}

// 获取词汇表大小
size_t tokenizer_get_vocab_size(const Tokenizer* tokenizer) {
    if (!tokenizer) return 0;
    return tokenizer->vocab_size;
}

// 从令牌ID获取文本
const char* tokenizer_id_to_token(const Tokenizer* tokenizer, int32_t token_id) {
    if (!tokenizer || !tokenizer->vocab) return NULL;
    return find_token(tokenizer->vocab, token_id);
}

// 从文本获取令牌ID
int32_t tokenizer_token_to_id(const Tokenizer* tokenizer, const char* token) {
    if (!tokenizer || !tokenizer->vocab || !token) return -1;
    return find_token_id(tokenizer->vocab, token);
}

// 添加特殊令牌
int tokenizer_add_special_tokens(Tokenizer* tokenizer,
                               const char** tokens,
                               size_t num_tokens) {
    if (!tokenizer || !tokens || !tokenizer->vocab) return -1;
    
    VocabHashTable* table = (VocabHashTable*)tokenizer->vocab;
    int32_t next_id = table->size;
    
    for (size_t i = 0; i < num_tokens; i++) {
        if (add_to_vocab(table, tokens[i], next_id++) != 0) {
            return -1;
        }
    }
    
    tokenizer->vocab_size = table->size;
    return 0;
}

// 保存分词器
int tokenizer_save(const Tokenizer* tokenizer, const char* path) {
    if (!tokenizer || !path) return -1;
    
    // 创建目录
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", path);
    system(cmd);
    
    // 保存配置
    char config_path[512];
    snprintf(config_path, sizeof(config_path), "%s/tokenizer_config.json", path);
    FILE* fp = fopen(config_path, "w");
    if (!fp) return -1;
    
    // TODO: 生成配置JSON
    
    fclose(fp);
    
    // 保存词汇表
    char vocab_path[512];
    snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.txt", path);
    fp = fopen(vocab_path, "w");
    if (!fp) return -1;
    
    VocabHashTable* table = (VocabHashTable*)tokenizer->vocab;
    for (size_t i = 0; i < table->capacity; i++) {
        VocabNode* node = table->buckets[i];
        while (node) {
            fprintf(fp, "%s\n", node->token);
            node = node->next;
        }
    }
    
    fclose(fp);
    
    // 保存BPE合并规则（如果有）
    if (tokenizer->config.type == TOKENIZER_BPE && tokenizer->merges) {
        char merges_path[512];
        snprintf(merges_path, sizeof(merges_path), "%s/merges.txt", path);
        fp = fopen(merges_path, "w");
        if (!fp) return -1;
        
        // TODO: 保存BPE合并规则
        
        fclose(fp);
    }
    
    return 0;
}

// 加载分词器
int tokenizer_load(Tokenizer** tokenizer, const char* path) {
    if (!tokenizer || !path) return -1;
    
    // 读取配置
    char config_path[512];
    snprintf(config_path, sizeof(config_path), "%s/tokenizer_config.json", path);
    
    FILE* fp = fopen(config_path, "r");
    if (!fp) return -1;
    
    // TODO: 解析配置JSON
    
    fclose(fp);
    
    // 创建配置
    TokenizerConfig config = {0};
    char vocab_path[512];
    snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.txt", path);
    config.vocab_file = vocab_path;
    
    // 初始化分词器
    return tokenizer_init(tokenizer, &config);
} 