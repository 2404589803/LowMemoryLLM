#include "extreme_compress.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// 定义常量
#define MAX_PATTERN_LENGTH 1024
#define MIN_PATTERN_LENGTH 4
#define HASH_SIZE 65536
#define ROLLING_WINDOW 8192

// 哈希表节点
typedef struct HashNode {
    uint8_t* pattern;
    size_t length;
    uint32_t frequency;
    struct HashNode* next;
} HashNode;

// 模式匹配表
typedef struct {
    HashNode* table[HASH_SIZE];
    size_t total_patterns;
} PatternTable;

// 计算模式哈希值
static uint32_t pattern_hash(const uint8_t* data, size_t len) {
    uint32_t hash = 0;
    for (size_t i = 0; i < len; i++) {
        hash = hash * 31 + data[i];
    }
    return hash % HASH_SIZE;
}

// 查找重复模式
static HashNode* find_pattern(PatternTable* table, const uint8_t* data, size_t len) {
    uint32_t hash = pattern_hash(data, len);
    HashNode* node = table->table[hash];
    
    while (node) {
        if (node->length == len && memcmp(node->pattern, data, len) == 0) {
            return node;
        }
        node = node->next;
    }
    return NULL;
}

// 添加新模式
static void add_pattern(PatternTable* table, const uint8_t* data, size_t len) {
    uint32_t hash = pattern_hash(data, len);
    HashNode* node = malloc(sizeof(HashNode));
    node->pattern = malloc(len);
    memcpy(node->pattern, data, len);
    node->length = len;
    node->frequency = 1;
    node->next = table->table[hash];
    table->table[hash] = node;
    table->total_patterns++;
}

// 增量编码
static void delta_encode(const uint8_t* input, size_t size, uint8_t* output) {
    if (size == 0) return;
    
    output[0] = input[0];
    for (size_t i = 1; i < size; i++) {
        output[i] = input[i] - input[i-1];
    }
}

// 增量解码
static void delta_decode(const uint8_t* input, size_t size, uint8_t* output) {
    if (size == 0) return;
    
    output[0] = input[0];
    for (size_t i = 1; i < size; i++) {
        output[i] = input[i] + output[i-1];
    }
}

// 初始化压缩上下文
CompressContext* init_compress_context(const CompressConfig* config) {
    CompressContext* ctx = malloc(sizeof(CompressContext));
    if (!ctx) return NULL;
    
    ctx->dictionary = malloc(sizeof(PatternTable));
    if (!ctx->dictionary) {
        free(ctx);
        return NULL;
    }
    
    memset(ctx->dictionary, 0, sizeof(PatternTable));
    ctx->pattern_cache = malloc(config->block_size);
    ctx->delta_buffer = malloc(config->block_size);
    ctx->total_size = 0;
    ctx->compressed_size = 0;
    
    return ctx;
}

// 压缩数据
int compress_data(CompressContext* ctx, const uint8_t* input, size_t input_size, 
                 uint8_t* output, size_t* output_size) {
    PatternTable* table = (PatternTable*)ctx->dictionary;
    uint8_t* delta_buffer = ctx->delta_buffer;
    size_t out_pos = 0;
    
    // 第一步：增量编码
    delta_encode(input, input_size, delta_buffer);
    
    // 第二步：模式匹配和替换
    for (size_t i = 0; i < input_size;) {
        size_t best_len = 0;
        HashNode* best_match = NULL;
        
        // 查找最长匹配模式
        for (size_t len = MIN_PATTERN_LENGTH; 
             len <= MAX_PATTERN_LENGTH && i + len <= input_size; 
             len++) {
            HashNode* match = find_pattern(table, &delta_buffer[i], len);
            if (match && len > best_len) {
                best_len = len;
                best_match = match;
            }
        }
        
        if (best_match) {
            // 写入模式引用
            output[out_pos++] = 0xFF; // 标记为模式引用
            memcpy(&output[out_pos], &best_match->pattern, sizeof(void*));
            out_pos += sizeof(void*);
            memcpy(&output[out_pos], &best_match->length, sizeof(size_t));
            out_pos += sizeof(size_t);
            i += best_match->length;
            best_match->frequency++;
        } else {
            // 写入原始数据
            output[out_pos++] = delta_buffer[i];
            
            // 尝试添加新模式
            if (i + MIN_PATTERN_LENGTH <= input_size) {
                add_pattern(table, &delta_buffer[i], MIN_PATTERN_LENGTH);
            }
            i++;
        }
    }
    
    *output_size = out_pos;
    ctx->total_size += input_size;
    ctx->compressed_size += out_pos;
    
    return 0;
}

// 解压数据
int decompress_data(CompressContext* ctx, const uint8_t* input, size_t input_size,
                   uint8_t* output, size_t* output_size) {
    size_t in_pos = 0;
    size_t out_pos = 0;
    
    while (in_pos < input_size) {
        if (input[in_pos] == 0xFF) {
            // 读取模式引用
            in_pos++;
            HashNode* pattern;
            memcpy(&pattern, &input[in_pos], sizeof(void*));
            in_pos += sizeof(void*);
            size_t length;
            memcpy(&length, &input[in_pos], sizeof(size_t));
            in_pos += sizeof(size_t);
            
            // 复制模式数据
            memcpy(&output[out_pos], pattern->pattern, length);
            out_pos += length;
        } else {
            // 复制原始数据
            output[out_pos++] = input[in_pos++];
        }
    }
    
    // 增量解码
    delta_decode(output, out_pos, output);
    *output_size = out_pos;
    
    return 0;
}

// 释放压缩上下文
void free_compress_context(CompressContext* ctx) {
    if (!ctx) return;
    
    PatternTable* table = (PatternTable*)ctx->dictionary;
    if (table) {
        // 释放所有模式
        for (size_t i = 0; i < HASH_SIZE; i++) {
            HashNode* node = table->table[i];
            while (node) {
                HashNode* next = node->next;
                free(node->pattern);
                free(node);
                node = next;
            }
        }
        free(table);
    }
    
    free(ctx->pattern_cache);
    free(ctx->delta_buffer);
    free(ctx);
}

// 流式压缩相关实现
CompressStream* init_compress_stream(const CompressConfig* config) {
    CompressStream* stream = malloc(sizeof(CompressStream));
    if (!stream) return NULL;
    
    stream->ctx = init_compress_context(config);
    if (!stream->ctx) {
        free(stream);
        return NULL;
    }
    
    stream->buffer_size = config->block_size;
    stream->stream_buffer = malloc(stream->buffer_size);
    if (!stream->stream_buffer) {
        free_compress_context(stream->ctx);
        free(stream);
        return NULL;
    }
    
    return stream;
}

int write_compress_stream(CompressStream* stream, const uint8_t* data, size_t size) {
    size_t compressed_size = 0;
    return compress_data(stream->ctx, data, size, stream->stream_buffer, &compressed_size);
}

int read_compress_stream(CompressStream* stream, uint8_t* data, size_t* size) {
    return decompress_data(stream->ctx, stream->stream_buffer, stream->buffer_size, data, size);
}

void free_compress_stream(CompressStream* stream) {
    if (!stream) return;
    free_compress_context(stream->ctx);
    free(stream->stream_buffer);
    free(stream);
} 