#include "kv_cache.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// 计算缓存大小
static size_t calculate_cache_size(const KVCacheConfig* config) {
    return config->batch_size * config->max_seq_length * 
           config->num_heads * config->head_dim * sizeof(float);
}

// 初始化KV缓存管理器
int kv_cache_init(KVCacheManager** manager, const KVCacheConfig* config, void* device) {
    if (!manager || !config || !device) return -1;
    
    *manager = (KVCacheManager*)malloc(sizeof(KVCacheManager));
    if (!*manager) return -1;
    
    // 复制配置
    memcpy(&(*manager)->config, config, sizeof(KVCacheConfig));
    (*manager)->device = device;
    
    // 分配每层的缓存项
    (*manager)->num_items = config->num_layers;
    (*manager)->items = (KVCacheItem**)malloc(sizeof(KVCacheItem*) * config->num_layers);
    if (!(*manager)->items) {
        free(*manager);
        return -1;
    }
    
    // 初始化每层的缓存
    size_t cache_size = calculate_cache_size(config);
    for (size_t i = 0; i < config->num_layers; i++) {
        (*manager)->items[i] = (KVCacheItem*)malloc(sizeof(KVCacheItem));
        if (!(*manager)->items[i]) goto cleanup;
        
        // 分配key和value缓存
        (*manager)->items[i]->key_cache = ((HAL_Device*)device)->allocate_memory(cache_size);
        (*manager)->items[i]->value_cache = ((HAL_Device*)device)->allocate_memory(cache_size);
        if (!(*manager)->items[i]->key_cache || !(*manager)->items[i]->value_cache) goto cleanup;
        
        // 分配位置映射
        (*manager)->items[i]->token_positions = (size_t*)malloc(sizeof(size_t) * config->max_seq_length);
        if (!(*manager)->items[i]->token_positions) goto cleanup;
        
        (*manager)->items[i]->current_length = 0;
    }
    
    return 0;
    
cleanup:
    kv_cache_cleanup(*manager);
    *manager = NULL;
    return -1;
}

// 释放KV缓存管理器
void kv_cache_cleanup(KVCacheManager* manager) {
    if (!manager) return;
    
    if (manager->items) {
        for (size_t i = 0; i < manager->num_items; i++) {
            if (manager->items[i]) {
                HAL_Device* device = (HAL_Device*)manager->device;
                if (manager->items[i]->key_cache)
                    device->free_memory(manager->items[i]->key_cache);
                if (manager->items[i]->value_cache)
                    device->free_memory(manager->items[i]->value_cache);
                if (manager->items[i]->token_positions)
                    free(manager->items[i]->token_positions);
                free(manager->items[i]);
            }
        }
        free(manager->items);
    }
    
    free(manager);
}

// 重置缓存
void kv_cache_reset(KVCacheManager* manager) {
    if (!manager) return;
    
    for (size_t i = 0; i < manager->num_items; i++) {
        if (manager->items[i]) {
            manager->items[i]->current_length = 0;
        }
    }
}

// 添加KV到缓存
int kv_cache_append(KVCacheManager* manager, 
                   size_t layer_idx,
                   const void* key, 
                   const void* value,
                   size_t seq_idx) {
    if (!manager || layer_idx >= manager->num_items || !key || !value) return -1;
    
    KVCacheItem* item = manager->items[layer_idx];
    if (!item || item->current_length >= manager->config.max_seq_length) return -1;
    
    HAL_Device* device = (HAL_Device*)manager->device;
    size_t offset = item->current_length * manager->config.num_heads * manager->config.head_dim;
    size_t size = manager->config.num_heads * manager->config.head_dim * sizeof(float);
    
    // 复制key和value到缓存
    device->memcpy_to_device((char*)item->key_cache + offset * sizeof(float), key, size);
    device->memcpy_to_device((char*)item->value_cache + offset * sizeof(float), value, size);
    
    // 更新位置映射
    item->token_positions[item->current_length] = seq_idx;
    item->current_length++;
    
    return 0;
}

// 从缓存获取KV
int kv_cache_lookup(KVCacheManager* manager,
                   size_t layer_idx,
                   void* key_out,
                   void* value_out,
                   const size_t* positions,
                   size_t num_positions) {
    if (!manager || layer_idx >= manager->num_items || !key_out || !value_out || !positions) return -1;
    
    KVCacheItem* item = manager->items[layer_idx];
    if (!item) return -1;
    
    HAL_Device* device = (HAL_Device*)manager->device;
    size_t head_size = manager->config.num_heads * manager->config.head_dim * sizeof(float);
    
    // 收集请求的位置的KV
    for (size_t i = 0; i < num_positions; i++) {
        if (positions[i] >= item->current_length) return -1;
        
        size_t offset = positions[i] * head_size;
        device->memcpy_from_device((char*)key_out + i * head_size,
                                 (char*)item->key_cache + offset, head_size);
        device->memcpy_from_device((char*)value_out + i * head_size,
                                 (char*)item->value_cache + offset, head_size);
    }
    
    return 0;
}

// 缓存旋转
int kv_cache_rotate(KVCacheManager* manager,
                   size_t layer_idx,
                   size_t rotation_offset) {
    if (!manager || layer_idx >= manager->num_items) return -1;
    
    KVCacheItem* item = manager->items[layer_idx];
    if (!item || rotation_offset >= item->current_length) return -1;
    
    HAL_Device* device = (HAL_Device*)manager->device;
    size_t head_size = manager->config.num_heads * manager->config.head_dim * sizeof(float);
    size_t move_size = (item->current_length - rotation_offset) * head_size;
    
    // 移动key缓存
    void* temp = malloc(move_size);
    if (!temp) return -1;
    
    device->memcpy_from_device(temp, (char*)item->key_cache + rotation_offset * head_size, move_size);
    device->memcpy_to_device(item->key_cache, temp, move_size);
    
    // 移动value缓存
    device->memcpy_from_device(temp, (char*)item->value_cache + rotation_offset * head_size, move_size);
    device->memcpy_to_device(item->value_cache, temp, move_size);
    
    free(temp);
    
    // 更新位置映射
    memmove(item->token_positions, 
            item->token_positions + rotation_offset,
            (item->current_length - rotation_offset) * sizeof(size_t));
    item->current_length -= rotation_offset;
    
    return 0;
}

// 缓存压缩
int kv_cache_compact(KVCacheManager* manager, size_t layer_idx) {
    if (!manager || layer_idx >= manager->num_items) return -1;
    
    KVCacheItem* item = manager->items[layer_idx];
    if (!item) return -1;
    
    // 标记有效位置
    char* valid = (char*)calloc(item->current_length, sizeof(char));
    if (!valid) return -1;
    
    size_t valid_count = 0;
    for (size_t i = 0; i < item->current_length; i++) {
        if (item->token_positions[i] != (size_t)-1) {
            valid[i] = 1;
            valid_count++;
        }
    }
    
    if (valid_count == item->current_length) {
        free(valid);
        return 0;  // 无需压缩
    }
    
    // 创建新的缓存
    HAL_Device* device = (HAL_Device*)manager->device;
    size_t head_size = manager->config.num_heads * manager->config.head_dim * sizeof(float);
    void* new_key_cache = device->allocate_memory(valid_count * head_size);
    void* new_value_cache = device->allocate_memory(valid_count * head_size);
    if (!new_key_cache || !new_value_cache) {
        if (new_key_cache) device->free_memory(new_key_cache);
        if (new_value_cache) device->free_memory(new_value_cache);
        free(valid);
        return -1;
    }
    
    // 复制有效数据
    size_t new_idx = 0;
    for (size_t i = 0; i < item->current_length; i++) {
        if (valid[i]) {
            device->memcpy_to_device((char*)new_key_cache + new_idx * head_size,
                                   (char*)item->key_cache + i * head_size, head_size);
            device->memcpy_to_device((char*)new_value_cache + new_idx * head_size,
                                   (char*)item->value_cache + i * head_size, head_size);
            item->token_positions[new_idx] = item->token_positions[i];
            new_idx++;
        }
    }
    
    // 更新缓存
    device->free_memory(item->key_cache);
    device->free_memory(item->value_cache);
    item->key_cache = new_key_cache;
    item->value_cache = new_value_cache;
    item->current_length = valid_count;
    
    free(valid);
    return 0;
}

// 磁盘卸载
int kv_cache_offload(KVCacheManager* manager,
                    size_t layer_idx,
                    const char* cache_dir) {
    if (!manager || layer_idx >= manager->num_items || !cache_dir) return -1;
    
    KVCacheItem* item = manager->items[layer_idx];
    if (!item) return -1;
    
    char filename[256];
    snprintf(filename, sizeof(filename), "%s/layer_%zu_kv_cache.bin", cache_dir, layer_idx);
    
    FILE* fp = fopen(filename, "wb");
    if (!fp) return -1;
    
    // 写入元数据
    fwrite(&item->current_length, sizeof(size_t), 1, fp);
    fwrite(item->token_positions, sizeof(size_t), item->current_length, fp);
    
    // 写入缓存数据
    HAL_Device* device = (HAL_Device*)manager->device;
    size_t cache_size = item->current_length * manager->config.num_heads * 
                       manager->config.head_dim * sizeof(float);
    
    void* temp = malloc(cache_size);
    if (!temp) {
        fclose(fp);
        return -1;
    }
    
    // 写入key缓存
    device->memcpy_from_device(temp, item->key_cache, cache_size);
    fwrite(temp, 1, cache_size, fp);
    
    // 写入value缓存
    device->memcpy_from_device(temp, item->value_cache, cache_size);
    fwrite(temp, 1, cache_size, fp);
    
    free(temp);
    fclose(fp);
    
    // 释放设备内存
    device->free_memory(item->key_cache);
    device->free_memory(item->value_cache);
    item->key_cache = NULL;
    item->value_cache = NULL;
    
    return 0;
}

// 从磁盘加载
int kv_cache_load(KVCacheManager* manager,
                 size_t layer_idx,
                 const char* cache_dir) {
    if (!manager || layer_idx >= manager->num_items || !cache_dir) return -1;
    
    KVCacheItem* item = manager->items[layer_idx];
    if (!item) return -1;
    
    char filename[256];
    snprintf(filename, sizeof(filename), "%s/layer_%zu_kv_cache.bin", cache_dir, layer_idx);
    
    FILE* fp = fopen(filename, "rb");
    if (!fp) return -1;
    
    // 读取元数据
    size_t length;
    if (fread(&length, sizeof(size_t), 1, fp) != 1) {
        fclose(fp);
        return -1;
    }
    
    if (length > manager->config.max_seq_length) {
        fclose(fp);
        return -1;
    }
    
    if (fread(item->token_positions, sizeof(size_t), length, fp) != length) {
        fclose(fp);
        return -1;
    }
    
    // 分配设备内存
    HAL_Device* device = (HAL_Device*)manager->device;
    size_t cache_size = length * manager->config.num_heads * 
                       manager->config.head_dim * sizeof(float);
    
    item->key_cache = device->allocate_memory(cache_size);
    item->value_cache = device->allocate_memory(cache_size);
    if (!item->key_cache || !item->value_cache) {
        if (item->key_cache) device->free_memory(item->key_cache);
        if (item->value_cache) device->free_memory(item->value_cache);
        fclose(fp);
        return -1;
    }
    
    // 读取缓存数据
    void* temp = malloc(cache_size);
    if (!temp) {
        device->free_memory(item->key_cache);
        device->free_memory(item->value_cache);
        fclose(fp);
        return -1;
    }
    
    // 读取并上传key缓存
    if (fread(temp, 1, cache_size, fp) != cache_size) {
        free(temp);
        device->free_memory(item->key_cache);
        device->free_memory(item->value_cache);
        fclose(fp);
        return -1;
    }
    device->memcpy_to_device(item->key_cache, temp, cache_size);
    
    // 读取并上传value缓存
    if (fread(temp, 1, cache_size, fp) != cache_size) {
        free(temp);
        device->free_memory(item->key_cache);
        device->free_memory(item->value_cache);
        fclose(fp);
        return -1;
    }
    device->memcpy_to_device(item->value_cache, temp, cache_size);
    
    item->current_length = length;
    
    free(temp);
    fclose(fp);
    return 0;
} 