#include "extreme_kv_cache.h"
#include "extreme_compress.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// 临时文件前缀
#define TEMP_FILE_PREFIX "kv_cache_"

// 内部函数声明
static bool write_to_disk(const uint8_t* data, size_t size, const char* path);
static bool read_from_disk(uint8_t* data, size_t size, const char* path);
static void generate_temp_filename(char* buffer, uint32_t seq_pos, uint32_t layer_id, bool is_key);
static int find_least_used_block(ExtremeKVCache* cache);
static bool compress_and_store(ExtremeKVCache* cache, const float* data, uint32_t size, int block_idx);
static bool decompress_and_load(ExtremeKVCache* cache, float* output, int block_idx);

ExtremeKVCache* extreme_kv_cache_init(void) {
    ExtremeKVCache* cache = (ExtremeKVCache*)malloc(sizeof(ExtremeKVCache));
    if (!cache) return NULL;

    cache->blocks = (CacheBlockMeta*)malloc(MAX_BLOCKS * sizeof(CacheBlockMeta));
    cache->memory_buffer = (uint8_t*)malloc(CACHE_BLOCK_SIZE * MAX_BLOCKS);
    
    if (!cache->blocks || !cache->memory_buffer) {
        free(cache->blocks);
        free(cache->memory_buffer);
        free(cache);
        return NULL;
    }

    cache->total_blocks = MAX_BLOCKS;
    cache->active_blocks = 0;
    cache->total_tokens_processed = 0;
    cache->avg_compression_ratio = 1.0;

    // 初始化所有块的元数据
    memset(cache->blocks, 0, MAX_BLOCKS * sizeof(CacheBlockMeta));
    for (int i = 0; i < MAX_BLOCKS; i++) {
        cache->blocks[i].location = CACHE_MEMORY;
        cache->blocks[i].compression_ratio = 1;
    }

    return cache;
}

bool extreme_kv_cache_store(ExtremeKVCache* cache, const float* data,
                           uint32_t size, uint32_t seq_pos, uint32_t layer_id, bool is_key) {
    if (!cache || !data || size == 0) return false;
    
    // 查找可用块或最少使用的块
    int block_idx = find_least_used_block(cache);
    if (block_idx < 0) return false;

    // 如果当前块在磁盘上，需要清理
    if (cache->blocks[block_idx].location == CACHE_DISK) {
        remove(cache->blocks[block_idx].temp_file_path);
    }

    // 更新块元数据
    cache->blocks[block_idx].sequence_pos = seq_pos;
    cache->blocks[block_idx].layer_id = layer_id;
    cache->blocks[block_idx].is_key = is_key;
    
    // 尝试压缩存储
    if (!compress_and_store(cache, data, size, block_idx)) {
        // 如果压缩失败，尝试直接写入磁盘
        generate_temp_filename(cache->blocks[block_idx].temp_file_path, seq_pos, layer_id, is_key);
        if (!write_to_disk((const uint8_t*)data, size * sizeof(float), 
                          cache->blocks[block_idx].temp_file_path)) {
            return false;
        }
        cache->blocks[block_idx].location = CACHE_DISK;
    }

    cache->total_tokens_processed++;
    return true;
}

bool extreme_kv_cache_retrieve(ExtremeKVCache* cache, float* output,
                             uint32_t seq_pos, uint32_t layer_id, bool is_key) {
    if (!cache || !output) return false;

    // 查找对应的缓存块
    int block_idx = -1;
    for (uint32_t i = 0; i < cache->total_blocks; i++) {
        if (cache->blocks[i].sequence_pos == seq_pos &&
            cache->blocks[i].layer_id == layer_id &&
            cache->blocks[i].is_key == is_key) {
            block_idx = i;
            break;
        }
    }

    if (block_idx < 0) return false;

    // 根据位置进行相应的读取操作
    switch (cache->blocks[block_idx].location) {
        case CACHE_MEMORY:
            memcpy(output, cache->memory_buffer + block_idx * CACHE_BLOCK_SIZE,
                   cache->blocks[block_idx].block_size);
            return true;

        case CACHE_COMPRESSED:
            return decompress_and_load(cache, output, block_idx);

        case CACHE_DISK:
            return read_from_disk((uint8_t*)output, cache->blocks[block_idx].block_size,
                                cache->blocks[block_idx].temp_file_path);

        default:
            return false;
    }
}

void extreme_kv_cache_cleanup(ExtremeKVCache* cache) {
    if (!cache) return;

    // 清理所有磁盘上的临时文件
    for (uint32_t i = 0; i < cache->total_blocks; i++) {
        if (cache->blocks[i].location == CACHE_DISK) {
            remove(cache->blocks[i].temp_file_path);
        }
    }

    free(cache->memory_buffer);
    free(cache->blocks);
    free(cache);
}

// 内部函数实现
static bool write_to_disk(const uint8_t* data, size_t size, const char* path) {
    FILE* file = fopen(path, "wb");
    if (!file) return false;
    
    bool success = fwrite(data, 1, size, file) == size;
    fclose(file);
    return success;
}

static bool read_from_disk(uint8_t* data, size_t size, const char* path) {
    FILE* file = fopen(path, "rb");
    if (!file) return false;
    
    bool success = fread(data, 1, size, file) == size;
    fclose(file);
    return success;
}

static void generate_temp_filename(char* buffer, uint32_t seq_pos, uint32_t layer_id, bool is_key) {
    snprintf(buffer, 32, "%s%u_%u_%d.tmp", TEMP_FILE_PREFIX, seq_pos, layer_id, is_key);
}

static int find_least_used_block(ExtremeKVCache* cache) {
    // 如果有空闲块，直接返回
    if (cache->active_blocks < cache->total_blocks) {
        return cache->active_blocks++;
    }

    // 使用简单的LRU策略
    return (cache->total_tokens_processed % cache->total_blocks);
}

static bool compress_and_store(ExtremeKVCache* cache, const float* data, uint32_t size, int block_idx) {
    // 使用extreme_compress模块进行压缩
    uint8_t* compressed_data = cache->memory_buffer + block_idx * CACHE_BLOCK_SIZE;
    size_t compressed_size = CACHE_BLOCK_SIZE;
    
    if (extreme_compress((const uint8_t*)data, size * sizeof(float), 
                        compressed_data, &compressed_size)) {
        cache->blocks[block_idx].block_size = compressed_size;
        cache->blocks[block_idx].compression_ratio = (uint8_t)(size * sizeof(float) / compressed_size);
        cache->blocks[block_idx].location = CACHE_COMPRESSED;
        
        // 更新平均压缩率
        cache->avg_compression_ratio = (cache->avg_compression_ratio * cache->total_tokens_processed + 
                                      cache->blocks[block_idx].compression_ratio) / 
                                     (cache->total_tokens_processed + 1);
        return true;
    }
    return false;
}

static bool decompress_and_load(ExtremeKVCache* cache, float* output, int block_idx) {
    uint8_t* compressed_data = cache->memory_buffer + block_idx * CACHE_BLOCK_SIZE;
    size_t original_size = cache->blocks[block_idx].block_size * 
                          cache->blocks[block_idx].compression_ratio;
    
    return extreme_decompress(compressed_data, cache->blocks[block_idx].block_size,
                            (uint8_t*)output, &original_size);
} 