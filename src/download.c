#include "low_memory_llm.h"
#include <curl/curl.h>
#include <stdio.h>
#include <string.h>

// 用于写入数据的结构
typedef struct {
    FILE* fp;
    size_t total_bytes;
    size_t downloaded_bytes;
    DownloadProgressCallback progress_cb;
} WriteData;

// CURL写入回调
static size_t write_callback(void* ptr, size_t size, size_t nmemb, void* userdata) {
    WriteData* wd = (WriteData*)userdata;
    size_t written = fwrite(ptr, size, nmemb, wd->fp);
    
    wd->downloaded_bytes += written;
    if (wd->progress_cb) {
        wd->progress_cb(wd->downloaded_bytes, wd->total_bytes);
    }
    
    return written;
}

int llm_download_weights(const DownloadConfig* config) {
    CURL* curl;
    CURLcode res;
    int ret = 0;
    static char error_buffer[CURL_ERROR_SIZE];
    
    // 初始化CURL
    curl = curl_easy_init();
    if (!curl) {
        printf("CURL初始化失败\n");
        return -1;
    }
    
    FILE* fp = fopen(config->save_path, "wb");
    if (!fp) {
        printf("无法打开文件：%s\n", config->save_path);
        curl_easy_cleanup(curl);
        return -2;
    }
    
    WriteData wd = {
        .fp = fp,
        .total_bytes = 0,
        .downloaded_bytes = 0,
        .progress_cb = config->progress_callback
    };
    
    // 设置CURL选项
    printf("正在设置CURL选项...\n");
    
    if (curl_easy_setopt(curl, CURLOPT_URL, config->url) != CURLE_OK) {
        printf("设置URL失败\n");
        ret = -3;
        goto cleanup;
    }
    
    if (curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback) != CURLE_OK) {
        printf("设置写入回调失败\n");
        ret = -3;
        goto cleanup;
    }
    
    if (curl_easy_setopt(curl, CURLOPT_WRITEDATA, &wd) != CURLE_OK) {
        printf("设置写入数据失败\n");
        ret = -3;
        goto cleanup;
    }
    
    curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, error_buffer);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, config->timeout_seconds);
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);  // 添加详细输出
    
    if (!config->verify_ssl) {
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
    }
    
    if (config->proxy) {
        curl_easy_setopt(curl, CURLOPT_PROXY, config->proxy);
    }
    
    // 执行下载
    printf("开始下载...\n");
    res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        printf("CURL错误：%s\n", curl_easy_strerror(res));
        ret = -3;
    }
    
cleanup:
    // 清理
    fclose(fp);
    curl_easy_cleanup(curl);
    
    return ret;
}

const char* llm_get_download_error(void) {
    return curl_easy_strerror(curl_easy_perform(NULL));
}

// 构建 Hugging Face URL
static char* build_hf_url(const char* repo_id, const char* filename, int use_mirror) {
    const char* base_url = use_mirror ? "https://hf-mirror.com/" : "https://huggingface.co/";
    size_t url_len = strlen(base_url) + strlen(repo_id) + strlen("/resolve/main/") + strlen(filename) + 1;
    char* url = (char*)malloc(url_len);
    
    if (url) {
        snprintf(url, url_len, "%s%s/resolve/main/%s", base_url, repo_id, filename);
    }
    return url;
}

// 测试URL连接
static int test_url_connection(const char* url) {
    CURL* curl = curl_easy_init();
    if (!curl) return 0;

    CURLcode res;
    long response_code;
    
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);  // 只检查头部
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);  // 5秒超时
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
    
    res = curl_easy_perform(curl);
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
    curl_easy_cleanup(curl);
    
    return (res == CURLE_OK && response_code == 200);
}

int llm_download_from_hf(const HFDownloadConfig* config) {
    if (!config || !config->repo_id || !config->filename || !config->save_path) {
        printf("配置参数无效\n");
        return -1;
    }

    // 首先尝试原始URL
    char* original_url = build_hf_url(config->repo_id, config->filename, 0);
    if (!original_url) {
        printf("构建URL失败\n");
        return -2;
    }

    printf("正在测试连接...\n");
    int use_mirror = !test_url_connection(original_url);
    free(original_url);

    // 根据测试结果选择URL
    char* url = build_hf_url(config->repo_id, config->filename, use_mirror);
    if (!url) {
        printf("构建URL失败\n");
        return -2;
    }

    printf("尝试从以下URL下载：%s\n", url);
    printf("使用%s\n", use_mirror ? "国内镜像" : "原始网址");

    DownloadConfig dl_config = {
        .url = url,
        .save_path = config->save_path,
        .verify_ssl = 0,
        .proxy = NULL,
        .timeout_seconds = config->timeout_seconds,
        .progress_callback = config->progress_callback
    };

    int result = llm_download_weights(&dl_config);
    
    if (result != 0) {
        printf("下载失败，URL：%s\n", url);
    }

    free(url);
    return result;
} 