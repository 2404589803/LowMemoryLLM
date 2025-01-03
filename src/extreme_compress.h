#ifndef EXTREME_COMPRESS_H
#define EXTREME_COMPRESS_H

#include <stdint.h>
#include <stdbool.h>

// 压缩配置
typedef struct {
    uint32_t block_size;          // 块大小
    uint32_t dictionary_size;     // 字典大小
    bool use_delta_encoding;      // 是否使用增量编码
    bool use_pattern_matching;    // 是否使用模式匹配
    float similarity_threshold;   // 相似度阈值
} CompressConfig;

// 压缩上下文
typedef struct {
    void* dictionary;             // 压缩字典
    void* pattern_cache;          // 模式缓存
    void* delta_buffer;          // 增量缓存
    uint64_t total_size;         // 总大小
    uint64_t compressed_size;    // 压缩后大小
} CompressContext;

// 初始化压缩上下文
CompressContext* init_compress_context(const CompressConfig* config);

// 释放压缩上下文
void free_compress_context(CompressContext* ctx);

// 压缩数据
int compress_data(CompressContext* ctx, const uint8_t* input, size_t input_size, 
                 uint8_t* output, size_t* output_size);

// 解压数据
int decompress_data(CompressContext* ctx, const uint8_t* input, size_t input_size,
                   uint8_t* output, size_t* output_size);

// 流式压缩
typedef struct {
    CompressContext* ctx;
    void* stream_buffer;
    size_t buffer_size;
} CompressStream;

// 初始化流式压缩
CompressStream* init_compress_stream(const CompressConfig* config);

// 写入流式压缩数据
int write_compress_stream(CompressStream* stream, const uint8_t* data, size_t size);

// 读取流式压缩数据
int read_compress_stream(CompressStream* stream, uint8_t* data, size_t* size);

// 释放流式压缩
void free_compress_stream(CompressStream* stream);

#endif // EXTREME_COMPRESS_H 