#ifndef EXTREME_KV_CACHE_H
#define EXTREME_KV_CACHE_H

#include <stdint.h>
#include <stdbool.h>

// 定义缓存块大小（以字节为单位）
#define CACHE_BLOCK_SIZE 256  // 每个缓存块使用256字节
#define MAX_BLOCKS 4         // 最多4个缓存块，总共1KB

// 缓存位置枚举
typedef enum {
    CACHE_MEMORY,    // 内存中
    CACHE_DISK,      // 磁盘中
    CACHE_COMPRESSED // 压缩状态
} CacheLocation;

// 缓存块元数据
typedef struct {
    uint32_t sequence_pos;   // 序列位置
    uint32_t layer_id;       // 层ID
    uint16_t block_size;     // 实际数据大小
    uint8_t compression_ratio; // 压缩比率
    CacheLocation location;   // 当前位置
    bool is_key;             // 是K还是V
    char temp_file_path[32]; // 临时文件路径
} CacheBlockMeta;

// KV缓存管理器
typedef struct {
    CacheBlockMeta* blocks;
    uint8_t* memory_buffer;
    uint32_t total_blocks;
    uint32_t active_blocks;
    uint64_t total_tokens_processed;
    double avg_compression_ratio;
} ExtremeKVCache;

// 初始化KV缓存
ExtremeKVCache* extreme_kv_cache_init(void);

// 存储KV数据
bool extreme_kv_cache_store(ExtremeKVCache* cache, 
                          const float* data,
                          uint32_t size,
                          uint32_t seq_pos,
                          uint32_t layer_id,
                          bool is_key);

// 读取KV数据
bool extreme_kv_cache_retrieve(ExtremeKVCache* cache,
                             float* output,
                             uint32_t seq_pos,
                             uint32_t layer_id,
                             bool is_key);

// 清理缓存
void extreme_kv_cache_cleanup(ExtremeKVCache* cache);

#endif // EXTREME_KV_CACHE_H 