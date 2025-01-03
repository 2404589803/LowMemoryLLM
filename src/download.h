#ifndef DOWNLOAD_H
#define DOWNLOAD_H

#include <stddef.h>
#include <stdbool.h>

// 下载状态枚举
typedef enum {
    DOWNLOAD_SUCCESS = 0,
    DOWNLOAD_FAILED = -1,
    DOWNLOAD_INCOMPLETE = 1,
    DOWNLOAD_CORRUPTED = -2,
    DOWNLOAD_MEMORY_ERROR = -3
} DownloadStatus;

// 下载配置
typedef struct {
    size_t chunk_size;         // 分块大小
    size_t memory_limit;       // 内存限制
    size_t retry_count;        // 重试次数
    bool enable_resume;        // 启用断点续传
    bool verify_hash;          // 启用哈希验证
    const char* proxy_url;     // 代理服务器
    const char** mirror_urls;  // 镜像站点列表
    size_t mirror_count;       // 镜像站点数量
    const char* cache_dir;     // 缓存目录
} DownloadConfig;

// 下载进度回调
typedef void (*progress_callback)(size_t downloaded, size_t total, void* user_data);

// 初始化下载配置
DownloadConfig* download_config_init(void);

// 释放下载配置
void download_config_cleanup(DownloadConfig* config);

// 下载模型文件（新版本）
DownloadStatus download_model_ex(
    const char* url,                  // 主URL
    const char* save_path,            // 保存路径
    const char* token,                // API token（可选）
    const DownloadConfig* config,     // 下载配置
    progress_callback progress,       // 进度回调（可选）
    void* user_data                   // 用户数据（可选）
);

// 验证文件完整性（增强版）
DownloadStatus verify_file_ex(
    const char* file_path,            // 文件路径
    const char* expected_hash,        // 预期哈希值
    const char* hash_type            // 哈希类型（如"sha256"、"md5"）
);

// 解压文件（增强版）
DownloadStatus decompress_file_ex(
    const char* input_path,           // 输入文件
    const char* output_path,          // 输出路径
    const DownloadConfig* config,     // 配置（用于内存限制）
    progress_callback progress,       // 进度回调（可选）
    void* user_data                   // 用户数据（可选）
);

// 获取模型信息
typedef struct {
    size_t total_size;                // 总大小
    char hash[65];                    // 哈希值
    char hash_type[10];              // 哈希类型
    char version[32];                // 版本号
    bool requires_auth;              // 是否需要认证
} ModelInfo;

// 获取模型信息
DownloadStatus get_model_info(
    const char* url,                  // 模型URL
    const char* token,                // API token（可选）
    ModelInfo* info                   // 输出信息
);

// 检查更新
bool check_model_update(
    const char* local_path,           // 本地模型路径
    const char* url,                  // 远程URL
    const char* token                 // API token（可选）
);

#endif // DOWNLOAD_H 