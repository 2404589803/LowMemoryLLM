#ifndef KV_CACHE_H
#define KV_CACHE_H

#include <stdint.h>
#include <stddef.h>

// KV缓存配置
typedef struct {
    size_t max_seq_length;      // 最大序列长度
    size_t num_layers;          // Transformer层数
    size_t num_heads;           // 注意力头数
    size_t head_dim;            // 每个头的维度
    size_t batch_size;          // 批次大小
    int use_disk_offload;       // 是否使用磁盘卸载
} KVCacheConfig;

// KV缓存项
typedef struct {
    void* key_cache;           // Key缓存
    void* value_cache;         // Value缓存
    size_t current_length;     // 当前缓存的序列长度
    size_t* token_positions;   // 令牌位置映射
} KVCacheItem;

// KV缓存管理器
typedef struct {
    KVCacheConfig config;      // 缓存配置
    KVCacheItem** items;       // 每层的缓存项
    size_t num_items;          // 缓存项数量
    void* device;              // 设备指针
} KVCacheManager;

// 初始化KV缓存管理器
int kv_cache_init(KVCacheManager** manager, const KVCacheConfig* config, void* device);

// 释放KV缓存管理器
void kv_cache_cleanup(KVCacheManager* manager);

// 重置缓存
void kv_cache_reset(KVCacheManager* manager);

// 添加KV到缓存
int kv_cache_append(KVCacheManager* manager, 
                   size_t layer_idx,
                   const void* key, 
                   const void* value,
                   size_t seq_idx);

// 从缓存获取KV
int kv_cache_lookup(KVCacheManager* manager,
                   size_t layer_idx,
                   void* key_out,
                   void* value_out,
                   const size_t* positions,
                   size_t num_positions);

// 缓存旋转（用于滑动窗口）
int kv_cache_rotate(KVCacheManager* manager,
                   size_t layer_idx,
                   size_t rotation_offset);

// 缓存压缩（移除无用位置）
int kv_cache_compact(KVCacheManager* manager,
                    size_t layer_idx);

// 磁盘卸载
int kv_cache_offload(KVCacheManager* manager,
                    size_t layer_idx,
                    const char* cache_dir);

// 从磁盘加载
int kv_cache_load(KVCacheManager* manager,
                 size_t layer_idx,
                 const char* cache_dir);

#endif // KV_CACHE_H 