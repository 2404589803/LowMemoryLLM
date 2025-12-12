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

// BPE合并规则表
typedef struct {
    BPEMerge* merges;
    size_t num_merges;
} BPEMergeTable;

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

// 加载BPE合并规则
static void* load_bpe_merges(const char* merges_file) {
    FILE* fp = fopen(merges_file, "r");
    if (!fp) return NULL;
    
    // 先数一遍行数以确定数组大小
    char line[1024];
    size_t num_merges = 0;
    
    while (fgets(line, sizeof(line), fp) != NULL) {
        // 跳过版本行和空行
        if (line[0] == '#' || line[0] == '\n') continue;
        num_merges++;
    }
    
    // 分配合并规则数组
    BPEMerge* merges = (BPEMerge*)malloc(sizeof(BPEMerge) * num_merges);
    if (!merges) {
        fclose(fp);
        return NULL;
    }
    
    // 重新从文件开始读取
    rewind(fp);
    size_t idx = 0;
    
    while (fgets(line, sizeof(line), fp) != NULL) {
        // 跳过版本行和空行
        if (line[0] == '#' || line[0] == '\n') continue;
        
        // 移除换行符
        size_t len = strlen(line);
        if (len > 0 && line[len-1] == '\n') {
            line[len-1] = '\0';
            len--;
        }
        
        // 复制合并对
        merges[idx].pair = strdup(line);
        merges[idx].priority = idx;  // 优先级按文件中的顺序
        idx++;
    }
    
    fclose(fp);
    
    // 创建BPE合并规则结构
    BPEMergeTable* table = (BPEMergeTable*)malloc(sizeof(BPEMergeTable));
    if (!table) {
        for (size_t i = 0; i < num_merges; i++) {
            free(merges[i].pair);
        }
        free(merges);
        return NULL;
    }
    
    table->merges = merges;
    table->num_merges = num_merges;
    
    return table;
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
        (*tokenizer)->merges = load_bpe_merges(config->merges_file);
        if (!(*tokenizer)->merges) {
            tokenizer_free(*tokenizer);
            return -1;
        }
    } else {
        (*tokenizer)->merges = NULL;
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
        BPEMergeTable* merges_table = (BPEMergeTable*)tokenizer->merges;
        for (size_t i = 0; i < merges_table->num_merges; i++) {
            free(merges_table->merges[i].pair);
        }
        free(merges_table->merges);
    }
    
    free(tokenizer);
}

// 定义一些BPE分词相关的辅助函数
static char** split_into_chars(const char* text, size_t* num_chars) {
    size_t text_len = strlen(text);
    char** chars = (char**)malloc(sizeof(char*) * text_len);
    if (!chars) return NULL;
    
    *num_chars = 0;
    
    for (size_t i = 0; i < text_len; ) {
        size_t char_len = 1;
        // 处理UTF-8多字节字符
        if ((text[i] & 0xE0) == 0xC0) char_len = 2;
        else if ((text[i] & 0xF0) == 0xE0) char_len = 3;
        else if ((text[i] & 0xF8) == 0xF0) char_len = 4;
        
        // 确保不会超出文本边界
        if (i + char_len > text_len) break;
        
        chars[*num_chars] = (char*)malloc(char_len + 1);
        if (!chars[*num_chars]) {
            for (size_t j = 0; j < *num_chars; j++) {
                free(chars[j]);
            }
            free(chars);
            return NULL;
        }
        
        memcpy(chars[*num_chars], text + i, char_len);
        chars[*num_chars][char_len] = '\0';
        (*num_chars)++;
        
        i += char_len;
    }
    
    return chars;
}

static void free_chars(char** chars, size_t num_chars) {
    for (size_t i = 0; i < num_chars; i++) {
        free(chars[i]);
    }
    free(chars);
}

static char* join_chars(char** chars, size_t start, size_t end) {
    if (start >= end) return NULL;
    
    // 计算需要的总长度
    size_t total_len = 0;
    for (size_t i = start; i < end; i++) {
        total_len += strlen(chars[i]);
    }
    
    char* result = (char*)malloc(total_len + 1);
    if (!result) return NULL;
    
    // 合并字符
    size_t offset = 0;
    for (size_t i = start; i < end; i++) {
        size_t len = strlen(chars[i]);
        memcpy(result + offset, chars[i], len);
        offset += len;
    }
    
    result[total_len] = '\0';
    return result;
}

static int find_best_merge(char** parts, size_t num_parts, BPEMergeTable* merges, size_t* best_i, size_t* best_j) {
    int best_priority = -1;
    *best_i = 0;
    *best_j = 0;
    
    // 检查所有相邻对
    for (size_t i = 0; i < num_parts - 1; i++) {
        // 构建合并对作为查询
        size_t len1 = strlen(parts[i]);
        size_t len2 = strlen(parts[i + 1]);
        char* pair = (char*)malloc(len1 + len2 + 2);
        if (!pair) return -1;
        
        sprintf(pair, "%s %s", parts[i], parts[i + 1]);
        
        // 在合并规则中查找
        for (size_t j = 0; j < merges->num_merges; j++) {
            if (strcmp(pair, merges->merges[j].pair) == 0) {
                if (best_priority == -1 || merges->merges[j].priority < best_priority) {
                    best_priority = merges->merges[j].priority;
                    *best_i = i;
                    *best_j = i + 1;
                }
                break;
            }
        }
        
        free(pair);
    }
    
    return best_priority;
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
            {
                // 1. 将文本分割成单个字符
                size_t num_chars;
                char** chars = split_into_chars(text, &num_chars);
                if (!chars) return -1;
                
                // 初始子词为单个字符
                char** parts = (char**)malloc(sizeof(char*) * num_chars);
                if (!parts) {
                    free_chars(chars, num_chars);
                    return -1;
                }
                
                size_t num_parts = num_chars;
                for (size_t i = 0; i < num_chars; i++) {
                    parts[i] = strdup(chars[i]);
                }
                
                // 释放原始字符数组
                free_chars(chars, num_chars);
                
                // 获取BPE合并规则
                BPEMergeTable* merges = (BPEMergeTable*)tokenizer->merges;
                if (merges) {
                    // 2. 迭代应用BPE合并规则
                    while (num_parts > 1) {
                        size_t best_i, best_j;
                        int priority = find_best_merge(parts, num_parts, merges, &best_i, &best_j);
                        
                        if (priority == -1) break;  // 没有找到适用的合并规则
                        
                        // 合并选中的对
                        char* merged = (char*)malloc(strlen(parts[best_i]) + strlen(parts[best_j]) + 1);
                        if (!merged) break;
                        
                        strcpy(merged, parts[best_i]);
                        strcat(merged, parts[best_j]);
                        
                        free(parts[best_i]);
                        free(parts[best_j]);
                        
                        // 更新parts数组
                        parts[best_i] = merged;
                        for (size_t i = best_j + 1; i < num_parts; i++) {
                            parts[i - 1] = parts[i];
                        }
                        num_parts--;
                    }
                }
                
                // 3. 将合并后的部分转换为令牌ID
                for (size_t i = 0; i < num_parts && *num_tokens < max_length; i++) {
                    int32_t id = tokenizer_token_to_id(tokenizer, parts[i]);
                    if (id >= 0) {
                        tokens[(*num_tokens)++] = id;
                    } else if (tokenizer->config.unk_token) {
                        // 使用未知令牌
                        id = tokenizer_token_to_id(tokenizer, tokenizer->config.unk_token);
                        if (id >= 0) {
                            tokens[(*num_tokens)++] = id;
                        }
                    }
                }
                
                // 清理
                for (size_t i = 0; i < num_parts; i++) {
                    free(parts[i]);
                }
                free(parts);
            }
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
            {
                // Unigram分词器（简化版本）
                // 这是一个贪婪搜索的简单实现，完整实现应该使用Viterbi算法
                
                // 将文本转换为UTF-8字符
                size_t num_chars;
                char** chars = split_into_chars(text, &num_chars);
                if (!chars) return -1;
                
                // 尝试从每个起始位置开始寻找最长匹配
                size_t pos = 0;
                while (pos < num_chars && *num_tokens < max_length) {
                    int32_t best_id = -1;
                    size_t best_len = 0;
                    
                    // 从当前位置尝试不同长度的子串
                    for (size_t len = 1; len <= num_chars - pos; len++) {
                        char* substr = join_chars(chars, pos, pos + len);
                        if (!substr) continue;
                        
                        int32_t id = tokenizer_token_to_id(tokenizer, substr);
                        if (id >= 0 && len > best_len) {
                            best_id = id;
                            best_len = len;
                        }
                        
                        free(substr);
                    }
                    
                    // 如果找到匹配项，添加到结果
                    if (best_id >= 0) {
                        tokens[(*num_tokens)++] = best_id;
                        pos += best_len;
                    } else {
                        // 没有找到匹配，使用未知令牌并前进一个字符
                        if (tokenizer->config.unk_token) {
                            int32_t unk_id = tokenizer_token_to_id(tokenizer, tokenizer->config.unk_token);
                            if (unk_id >= 0) {
                                tokens[(*num_tokens)++] = unk_id;
                            }
                        }
                        pos++;
                    }
                }
                
                // 清理
                free_chars(chars, num_chars);
            }
            break;
            
        case TOKENIZER_SENTENCEPIECE:
            // TODO: 实现SentencePiece分词
            {
                // SentencePiece分词器（简化版本）
                // SentencePiece通常可以使用BPE或Unigram算法，这里我们使用
                // 类似Unigram的贪婪搜索算法，但增加了SentencePiece的特殊处理
                
                // 预处理：增加开始标记（如果未提供特殊token则使用空格）
                char* preprocessed_text;
                if (tokenizer->config.add_prefix_space) {
                    preprocessed_text = (char*)malloc(strlen(text) + 2);
                    if (!preprocessed_text) return -1;
                    preprocessed_text[0] = ' ';
                    strcpy(preprocessed_text + 1, text);
                } else {
                    preprocessed_text = strdup(text);
                    if (!preprocessed_text) return -1;
                }
                
                // 将文本转换为UTF-8字符
                size_t num_chars;
                char** chars = split_into_chars(preprocessed_text, &num_chars);
                free(preprocessed_text);
                if (!chars) return -1;
                
                // 尝试从每个起始位置开始寻找最长匹配
                size_t pos = 0;
                while (pos < num_chars && *num_tokens < max_length) {
                    int32_t best_id = -1;
                    size_t best_len = 0;
                    
                    // 从当前位置尝试不同长度的子串
                    for (size_t len = 1; len <= num_chars - pos; len++) {
                        char* substr = join_chars(chars, pos, pos + len);
                        if (!substr) continue;
                        
                        // SentencePiece中，多数token以"▁"（Unicode U+2581）开头
                        // 这里我们简化处理，直接尝试查找token
                        int32_t id = tokenizer_token_to_id(tokenizer, substr);
                        if (id >= 0 && len > best_len) {
                            best_id = id;
                            best_len = len;
                        }
                        
                        free(substr);
                    }
                    
                    // 如果找到匹配项，添加到结果
                    if (best_id >= 0) {
                        tokens[(*num_tokens)++] = best_id;
                        pos += best_len;
                    } else {
                        // 没有找到匹配，使用未知令牌并前进一个字符
                        if (tokenizer->config.unk_token) {
                            int32_t unk_id = tokenizer_token_to_id(tokenizer, tokenizer->config.unk_token);
                            if (unk_id >= 0) {
                                tokens[(*num_tokens)++] = unk_id;
                            }
                        }
                        pos++;
                    }
                }
                
                // 清理
                free_chars(chars, num_chars);
            }
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
    fprintf(fp, "{\n");
    fprintf(fp, "  \"model_type\": \"%s\",\n", 
        tokenizer->config.type == TOKENIZER_BPE ? "bpe" :
        tokenizer->config.type == TOKENIZER_WORDPIECE ? "wordpiece" :
        tokenizer->config.type == TOKENIZER_UNIGRAM ? "unigram" : "sentencepiece");
    fprintf(fp, "  \"add_special_tokens\": %s,\n", 
        tokenizer->config.add_special_tokens ? "true" : "false");
    fprintf(fp, "  \"add_prefix_space\": %s,\n", 
        tokenizer->config.add_prefix_space ? "true" : "false");
    
    if (tokenizer->config.unk_token)
        fprintf(fp, "  \"unk_token\": \"%s\",\n", tokenizer->config.unk_token);
    if (tokenizer->config.pad_token)
        fprintf(fp, "  \"pad_token\": \"%s\",\n", tokenizer->config.pad_token);
    if (tokenizer->config.bos_token)
        fprintf(fp, "  \"bos_token\": \"%s\",\n", tokenizer->config.bos_token);
    if (tokenizer->config.eos_token)
        fprintf(fp, "  \"eos_token\": \"%s\",\n", tokenizer->config.eos_token);
    if (tokenizer->config.mask_token)
        fprintf(fp, "  \"mask_token\": \"%s\",\n", tokenizer->config.mask_token);
    
    fprintf(fp, "  \"vocab_size\": %zu\n", tokenizer->vocab_size);
    fprintf(fp, "}\n");
    
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
        BPEMergeTable* merges = (BPEMergeTable*)tokenizer->merges;
        
        // 写入版本信息
        fprintf(fp, "#version: 0.2\n");
        
        // 按优先级排序写入合并规则
        for (size_t i = 0; i < merges->num_merges; i++) {
            fprintf(fp, "%s\n", merges->merges[i].pair);
        }
        
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
    TokenizerConfig config = {0};
    char line[1024];
    char model_type[32] = {0};
    
    while (fgets(line, sizeof(line), fp)) {
        // 简单解析JSON，仅用于演示，实际项目请使用JSON库
        if (strstr(line, "\"model_type\"")) {
            sscanf(line, "  \"model_type\": \"%[^\"]\"", model_type);
            if (strcmp(model_type, "bpe") == 0) config.type = TOKENIZER_BPE;
            else if (strcmp(model_type, "wordpiece") == 0) config.type = TOKENIZER_WORDPIECE;
            else if (strcmp(model_type, "unigram") == 0) config.type = TOKENIZER_UNIGRAM;
            else if (strcmp(model_type, "sentencepiece") == 0) config.type = TOKENIZER_SENTENCEPIECE;
        } else if (strstr(line, "\"add_special_tokens\"")) {
            char value[8];
            sscanf(line, "  \"add_special_tokens\": %s", value);
            config.add_special_tokens = (strcmp(value, "true,") == 0);
        } else if (strstr(line, "\"add_prefix_space\"")) {
            char value[8];
            sscanf(line, "  \"add_prefix_space\": %s", value);
            config.add_prefix_space = (strcmp(value, "true,") == 0);
        } else if (strstr(line, "\"unk_token\"")) {
            char token[64];
            sscanf(line, "  \"unk_token\": \"%[^\"]\"", token);
            config.unk_token = strdup(token);
        } else if (strstr(line, "\"pad_token\"")) {
            char token[64];
            sscanf(line, "  \"pad_token\": \"%[^\"]\"", token);
            config.pad_token = strdup(token);
        } else if (strstr(line, "\"bos_token\"")) {
            char token[64];
            sscanf(line, "  \"bos_token\": \"%[^\"]\"", token);
            config.bos_token = strdup(token);
        } else if (strstr(line, "\"eos_token\"")) {
            char token[64];
            sscanf(line, "  \"eos_token\": \"%[^\"]\"", token);
            config.eos_token = strdup(token);
        } else if (strstr(line, "\"mask_token\"")) {
            char token[64];
            sscanf(line, "  \"mask_token\": \"%[^\"]\"", token);
            config.mask_token = strdup(token);
        }
    }
    
    fclose(fp);
    
    // 设置词汇表文件路径
    char vocab_path[512];
    snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.txt", path);
    config.vocab_file = strdup(vocab_path);
    
    // 设置BPE合并规则文件路径（如果需要）
    if (config.type == TOKENIZER_BPE) {
        char merges_path[512];
        snprintf(merges_path, sizeof(merges_path), "%s/merges.txt", path);
        config.merges_file = strdup(merges_path);
    }
    
    // 初始化分词器
    int result = tokenizer_init(tokenizer, &config);
    
    // 清理
    free((void*)config.vocab_file);
    if (config.merges_file) free((void*)config.merges_file);
    if (config.unk_token) free((void*)config.unk_token);
    if (config.pad_token) free((void*)config.pad_token);
    if (config.bos_token) free((void*)config.bos_token);
    if (config.eos_token) free((void*)config.eos_token);
    if (config.mask_token) free((void*)config.mask_token);
    
    return result;
} 