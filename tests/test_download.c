#include "low_memory_llm.h"
#include <stdio.h>
#include <locale.h>
#include <windows.h>

void print_progress(size_t downloaded, size_t total) {
    if (total > 0) {
        float percentage = (float)downloaded / total * 100;
        printf("\r下载进度：%.1f%%", percentage);
        fflush(stdout);
    }
}

int main() {
    // 设置控制台代码页为UTF-8
    SetConsoleOutputCP(65001);
    
    // 测试从 Hugging Face 下载小型模型
    HFDownloadConfig config = {
        .repo_id = "openbmb/MiniCPM3-4B",
        .filename = "config.json",
        .save_path = "config.json",
        .token = NULL,
        .timeout_seconds = 3600,
        .progress_callback = print_progress
    };

    printf("开始从 Hugging Face 下载...\n");
    int result = llm_download_from_hf(&config);
    
    if (result != 0) {
        printf("\n下载失败，错误代码 %d：%s\n", result, llm_get_download_error());
        return 1;
    }

    printf("\n下载成功完成！\n");
    return 0;
} 