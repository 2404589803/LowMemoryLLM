#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <direct.h>
#include <errno.h>
#include "../src/download.h"

// 进度回调函数
static void progress_callback(size_t downloaded, size_t total, void* user_data) {
    static int last_percent = -1;
    int percent = (int)((double)downloaded / total * 100);
    
    if (percent != last_percent) {
        printf("\r下载进度: [");
        for (int i = 0; i < 50; i++) {
            if (i < percent / 2) {
                printf("=");
            } else if (i == percent / 2) {
                printf(">");
            } else {
                printf(" ");
            }
        }
        printf("] %d%%", percent);
        fflush(stdout);
        last_percent = percent;
    }
}

int main() {
    // MiniCPM-V-2_6 模型下载URL
    const char* model_url = "https://huggingface.co/openbmb/MiniCPM-V-2_6/resolve/main/pytorch_model.bin";
    const char* save_path = "models/MiniCPM-V-2_6/pytorch_model.bin";
    const char* token = "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";
    
    printf("=== LowMemoryLLM 模型下载测试 ===\n\n");
    
    // 创建保存目录
    printf("1. 创建目录结构...\n");
    if (_mkdir("models") != 0 && errno != EEXIST) {
        printf("创建models目录失败: %s\n", strerror(errno));
        return -1;
    }
    
    char model_dir[256];
    snprintf(model_dir, sizeof(model_dir), "models/MiniCPM-V-2_6");
    if (_mkdir(model_dir) != 0 && errno != EEXIST) {
        printf("创建模型目录失败: %s\n", strerror(errno));
        return -1;
    }
    printf("目录创建成功！\n\n");
    
    // 初始化下载配置
    printf("2. 配置下载参数...\n");
    DownloadConfig* config = download_config_init();
    if (!config) {
        printf("配置初始化失败！\n");
        return -1;
    }
    
    // 设置极限配置
    config->chunk_size = 1024;        // 1KB 块大小
    config->memory_limit = 1024;      // 1KB 内存限制
    config->retry_count = 5;          // 最多重试5次
    config->enable_resume = true;     // 启用断点续传
    config->verify_hash = true;       // 启用哈希验证
    
    printf("块大小: %zu bytes\n", config->chunk_size);
    printf("内存限制: %zu bytes\n", config->memory_limit);
    printf("重试次数: %zu\n", config->retry_count);
    printf("断点续传: %s\n", config->enable_resume ? "启用" : "禁用");
    printf("哈希验证: %s\n\n", config->verify_hash ? "启用" : "禁用");
    
    // 获取模型信息
    printf("3. 获取模型信息...\n");
    ModelInfo info;
    if (get_model_info(model_url, token, &info) == DOWNLOAD_SUCCESS) {
        printf("模型大小: %zu bytes\n", info.total_size);
        printf("哈希类型: %s\n", info.hash_type);
        printf("哈希值: %s\n", info.hash);
        printf("版本: %s\n", info.version);
        printf("需要认证: %s\n\n", info.requires_auth ? "是" : "否");
    } else {
        printf("获取模型信息失败！\n\n");
    }
    
    // 开始下载
    printf("4. 开始下载模型...\n");
    printf("URL: %s\n", model_url);
    printf("保存路径: %s\n\n", save_path);
    
    DownloadStatus status = download_model_ex(
        model_url,
        save_path,
        token,
        config,
        progress_callback,
        NULL
    );
    
    printf("\n\n5. 下载结果: ");
    switch (status) {
        case DOWNLOAD_SUCCESS:
            printf("成功！\n");
            break;
        case DOWNLOAD_FAILED:
            printf("失败！\n");
            break;
        case DOWNLOAD_INCOMPLETE:
            printf("下载不完整！\n");
            break;
        case DOWNLOAD_CORRUPTED:
            printf("文件损坏！\n");
            break;
        case DOWNLOAD_MEMORY_ERROR:
            printf("内存错误！\n");
            break;
        default:
            printf("未知错误！\n");
    }
    
    // 验证文件
    if (status == DOWNLOAD_SUCCESS) {
        printf("\n6. 验证文件完整性...\n");
        status = verify_file_ex(save_path, info.hash, info.hash_type);
        
        if (status == DOWNLOAD_SUCCESS) {
            printf("文件验证成功！\n");
        } else {
            printf("文件验证失败！\n");
        }
    }
    
    // 清理资源
    download_config_cleanup(config);
    printf("\n=== 测试完成 ===\n");
    
    return (status == DOWNLOAD_SUCCESS) ? 0 : -1;
} 