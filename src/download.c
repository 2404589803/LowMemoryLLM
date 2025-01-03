#include "download.h"
#include <curl/curl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/sha.h>
#include <openssl/md5.h>
#include <sys/stat.h>
#include <errno.h>
#include <zlib.h>
#include <cJSON/cJSON.h>

#define DEFAULT_CHUNK_SIZE (1024 * 1024)  // 1MB
#define DEFAULT_RETRY_COUNT 3
#define TEMP_SUFFIX ".part"
#define HASH_BUFFER_SIZE (8 * 1024)  // 8KB for hash calculation
#define DECOMPRESS_BUFFER_SIZE (16 * 1024)  // 16KB for decompression

// 响应数据结构
typedef struct {
    char* data;
    size_t size;
} ResponseData;

// 文件块信息
typedef struct {
    FILE* fp;
    size_t offset;
    size_t size;
    char* buffer;
    size_t buffer_size;
    progress_callback progress;
    void* user_data;
    size_t total_size;
    size_t downloaded;
} ChunkInfo;

// 默认配置
static const DownloadConfig default_config = {
    .chunk_size = DEFAULT_CHUNK_SIZE,
    .memory_limit = 1024 * 1024 * 10,  // 10MB
    .retry_count = DEFAULT_RETRY_COUNT,
    .enable_resume = true,
    .verify_hash = true,
    .proxy_url = NULL,
    .mirror_urls = NULL,
    .mirror_count = 0,
    .cache_dir = NULL
};

// CURL写回调 - 用于JSON响应
static size_t write_response_callback(void* ptr, size_t size, size_t nmemb, void* userdata) {
    ResponseData* response = (ResponseData*)userdata;
    size_t new_size = response->size + size * nmemb;
    
    char* new_data = realloc(response->data, new_size + 1);
    if (!new_data) return 0;
    
    response->data = new_data;
    memcpy(response->data + response->size, ptr, size * nmemb);
    response->size = new_size;
    response->data[new_size] = '\0';
    
    return size * nmemb;
}

// CURL写回调 - 用于文件下载
static size_t write_callback(void* ptr, size_t size, size_t nmemb, void* userdata) {
    ChunkInfo* chunk = (ChunkInfo*)userdata;
    size_t total = size * nmemb;
    
    if (chunk->downloaded + total > chunk->total_size) {
        total = chunk->total_size - chunk->downloaded;
    }
    
    if (total == 0) return 0;
    
    if (fwrite(ptr, 1, total, chunk->fp) != total) {
        return 0;
    }
    
    chunk->downloaded += total;
    
    if (chunk->progress) {
        chunk->progress(chunk->downloaded, chunk->total_size, chunk->user_data);
    }
    
    return total;
}

// 初始化CURL会话
static CURL* init_curl_session(const char* url, const char* token, 
                             const DownloadConfig* config, void* data,
                             curl_write_callback write_cb) {
    CURL* curl = curl_easy_init();
    if (!curl) return NULL;
    
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, data);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 5L);
    curl_easy_setopt(curl, CURLOPT_BUFFERSIZE, config->chunk_size);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
    
    // 设置代理
    if (config->proxy_url) {
        curl_easy_setopt(curl, CURLOPT_PROXY, config->proxy_url);
    }
    
    // 设置认证
    if (token) {
        char auth[1024];
        snprintf(auth, sizeof(auth), "Bearer %s", token);
        struct curl_slist* headers = NULL;
        headers = curl_slist_append(headers, auth);
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    }
    
    return curl;
}

// 计算文件哈希
static bool calculate_file_hash(const char* file_path, const char* hash_type,
                              char* hash_output, size_t output_size) {
    FILE* file = fopen(file_path, "rb");
    if (!file) return false;
    
    unsigned char buffer[HASH_BUFFER_SIZE];
    unsigned char hash[SHA512_DIGEST_LENGTH];  // 最大哈希长度
    
    if (strcmp(hash_type, "sha256") == 0) {
        SHA256_CTX sha256;
        SHA256_Init(&sha256);
        size_t bytes;
        while ((bytes = fread(buffer, 1, HASH_BUFFER_SIZE, file)) > 0) {
            SHA256_Update(&sha256, buffer, bytes);
        }
        SHA256_Final(hash, &sha256);
    } else if (strcmp(hash_type, "md5") == 0) {
        MD5_CTX md5;
        MD5_Init(&md5);
        size_t bytes;
        while ((bytes = fread(buffer, 1, HASH_BUFFER_SIZE, file)) > 0) {
            MD5_Update(&md5, buffer, bytes);
        }
        MD5_Final(hash, &md5);
    } else {
        fclose(file);
        return false;
    }
    
    fclose(file);
    
    // 转换为十六进制字符串
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        snprintf(hash_output + (i * 2), output_size - (i * 2), "%02x", hash[i]);
    }
    
    return true;
}

DownloadConfig* download_config_init(void) {
    DownloadConfig* config = (DownloadConfig*)malloc(sizeof(DownloadConfig));
    if (!config) return NULL;
    
    memcpy(config, &default_config, sizeof(DownloadConfig));
    return config;
}

void download_config_cleanup(DownloadConfig* config) {
    if (!config) return;
    free(config);
}

// 解压缩功能实现
DownloadStatus decompress_file_ex(const char* input_path, const char* output_path,
                                const DownloadConfig* config,
                                progress_callback progress, void* user_data) {
    if (!input_path || !output_path) return DOWNLOAD_FAILED;
    
    FILE* source = fopen(input_path, "rb");
    FILE* dest = fopen(output_path, "wb");
    if (!source || !dest) {
        if (source) fclose(source);
        if (dest) fclose(dest);
        return DOWNLOAD_FAILED;
    }
    
    // 获取源文件大小
    fseek(source, 0, SEEK_END);
    size_t source_size = ftell(source);
    fseek(source, 0, SEEK_SET);
    
    z_stream strm = {0};
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    
    if (inflateInit(&strm) != Z_OK) {
        fclose(source);
        fclose(dest);
        return DOWNLOAD_FAILED;
    }
    
    unsigned char in[DECOMPRESS_BUFFER_SIZE];
    unsigned char out[DECOMPRESS_BUFFER_SIZE];
    size_t total_processed = 0;
    int ret;
    
    do {
        strm.avail_in = fread(in, 1, DECOMPRESS_BUFFER_SIZE, source);
        total_processed += strm.avail_in;
        
        if (ferror(source)) {
            inflateEnd(&strm);
            fclose(source);
            fclose(dest);
            return DOWNLOAD_FAILED;
        }
        
        if (strm.avail_in == 0) break;
        strm.next_in = in;
        
        do {
            strm.avail_out = DECOMPRESS_BUFFER_SIZE;
            strm.next_out = out;
            
            ret = inflate(&strm, Z_NO_FLUSH);
            if (ret != Z_OK && ret != Z_STREAM_END) {
                inflateEnd(&strm);
                fclose(source);
                fclose(dest);
                return DOWNLOAD_FAILED;
            }
            
            size_t have = DECOMPRESS_BUFFER_SIZE - strm.avail_out;
            if (fwrite(out, 1, have, dest) != have || ferror(dest)) {
                inflateEnd(&strm);
                fclose(source);
                fclose(dest);
                return DOWNLOAD_FAILED;
            }
            
            if (progress) {
                progress(total_processed, source_size, user_data);
            }
            
        } while (strm.avail_out == 0);
        
    } while (ret != Z_STREAM_END);
    
    inflateEnd(&strm);
    fclose(source);
    fclose(dest);
    
    return (ret == Z_STREAM_END) ? DOWNLOAD_SUCCESS : DOWNLOAD_FAILED;
}

DownloadStatus download_model_ex(const char* url, const char* save_path,
                               const char* token, const DownloadConfig* config,
                               progress_callback progress, void* user_data) {
    if (!url || !save_path) return DOWNLOAD_FAILED;
    
    // 使用默认配置如果未提供
    DownloadConfig local_config;
    if (!config) {
        memcpy(&local_config, &default_config, sizeof(DownloadConfig));
        config = &local_config;
    }
    
    // 创建临时文件路径
    char temp_path[1024];
    snprintf(temp_path, sizeof(temp_path), "%s%s", save_path, TEMP_SUFFIX);
    
    // 初始化下载信息
    ChunkInfo chunk = {
        .fp = NULL,
        .offset = 0,
        .buffer = NULL,
        .progress = progress,
        .user_data = user_data,
        .downloaded = 0
    };
    
    // 获取文件大小
    CURL* curl = curl_easy_init();
    if (!curl) return DOWNLOAD_FAILED;
    
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);
    curl_easy_setopt(curl, CURLOPT_HEADER, 0L);
    
    if (curl_easy_perform(curl) == CURLE_OK) {
        double cl;
        curl_easy_getinfo(curl, CURLINFO_CONTENT_LENGTH_DOWNLOAD, &cl);
        chunk.total_size = (size_t)cl;
    }
    curl_easy_cleanup(curl);
    
    // 检查是否支持断点续传
    struct stat st;
    if (config->enable_resume && stat(temp_path, &st) == 0) {
        chunk.downloaded = st.st_size;
        chunk.fp = fopen(temp_path, "ab");
    } else {
        chunk.fp = fopen(temp_path, "wb");
    }
    
    if (!chunk.fp) return DOWNLOAD_FAILED;
    
    // 开始下载
    DownloadStatus status = DOWNLOAD_SUCCESS;
    size_t retry_count = 0;
    
    while (chunk.downloaded < chunk.total_size && retry_count <= config->retry_count) {
        curl = init_curl_session(url, token, config, &chunk, write_callback);
        if (!curl) {
            status = DOWNLOAD_FAILED;
            break;
        }
        
        // 设置断点续传
        if (chunk.downloaded > 0) {
            char range[64];
            snprintf(range, sizeof(range), "%zu-", chunk.downloaded);
            curl_easy_setopt(curl, CURLOPT_RANGE, range);
        }
        
        CURLcode res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);
        
        if (res != CURLE_OK) {
            retry_count++;
            continue;
        }
        
        break;
    }
    
    fclose(chunk.fp);
    
    // 验证下载是否完整
    if (status == DOWNLOAD_SUCCESS && chunk.downloaded == chunk.total_size) {
        // 如果需要验证哈希
        if (config->verify_hash) {
            ModelInfo info;
            if (get_model_info(url, token, &info) == DOWNLOAD_SUCCESS) {
                char calculated_hash[65];
                if (calculate_file_hash(temp_path, info.hash_type, calculated_hash, sizeof(calculated_hash))) {
                    if (strcmp(calculated_hash, info.hash) != 0) {
                        status = DOWNLOAD_CORRUPTED;
                    }
                }
            }
        }
        
        // 如果验证通过，重命名文件
        if (status == DOWNLOAD_SUCCESS) {
            if (rename(temp_path, save_path) != 0) {
                status = DOWNLOAD_FAILED;
            }
        }
    } else {
        status = DOWNLOAD_INCOMPLETE;
    }
    
    return status;
}

DownloadStatus verify_file_ex(const char* file_path, const char* expected_hash,
                            const char* hash_type) {
    if (!file_path || !expected_hash || !hash_type) return DOWNLOAD_FAILED;
    
    char calculated_hash[65];
    if (!calculate_file_hash(file_path, hash_type, calculated_hash, sizeof(calculated_hash))) {
        return DOWNLOAD_FAILED;
    }
    
    return (strcmp(calculated_hash, expected_hash) == 0) ? 
           DOWNLOAD_SUCCESS : DOWNLOAD_CORRUPTED;
}

DownloadStatus get_model_info(const char* url, const char* token, ModelInfo* info) {
    if (!url || !info) return DOWNLOAD_FAILED;
    
    // 构建API URL
    char api_url[2048];
    snprintf(api_url, sizeof(api_url), "%s/info", url);
    
    // 初始化响应数据
    ResponseData response = {0};
    response.data = malloc(1);
    response.size = 0;
    
    CURL* curl = init_curl_session(api_url, token, &default_config, &response, write_response_callback);
    if (!curl) {
        free(response.data);
        return DOWNLOAD_FAILED;
    }
    
    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    
    if (res != CURLE_OK) {
        free(response.data);
        return DOWNLOAD_FAILED;
    }
    
    // 解析JSON响应
    cJSON* root = cJSON_Parse(response.data);
    free(response.data);
    
    if (!root) {
        return DOWNLOAD_FAILED;
    }
    
    // 解析各个字段
    cJSON* size = cJSON_GetObjectItem(root, "size");
    cJSON* hash = cJSON_GetObjectItem(root, "hash");
    cJSON* hash_type = cJSON_GetObjectItem(root, "hash_type");
    cJSON* version = cJSON_GetObjectItem(root, "version");
    cJSON* auth = cJSON_GetObjectItem(root, "requires_auth");
    
    if (size && size->valueint) info->total_size = size->valueint;
    if (hash && hash->valuestring) strncpy(info->hash, hash->valuestring, sizeof(info->hash) - 1);
    if (hash_type && hash_type->valuestring) strncpy(info->hash_type, hash_type->valuestring, sizeof(info->hash_type) - 1);
    if (version && version->valuestring) strncpy(info->version, version->valuestring, sizeof(info->version) - 1);
    if (auth && auth->type == cJSON_True) info->requires_auth = true;
    
    cJSON_Delete(root);
    return DOWNLOAD_SUCCESS;
}

bool check_model_update(const char* local_path, const char* url, const char* token) {
    ModelInfo local_info = {0};
    ModelInfo remote_info = {0};
    
    if (get_model_info(url, token, &remote_info) != DOWNLOAD_SUCCESS) {
        return false;
    }
    
    // 读取本地版本信息
    char version_file[1024];
    snprintf(version_file, sizeof(version_file), "%s.version", local_path);
    FILE* fp = fopen(version_file, "r");
    if (!fp) return true;  // 如果没有版本文件，假设需要更新
    
    char local_version[32];
    if (fgets(local_version, sizeof(local_version), fp)) {
        fclose(fp);
        return strcmp(local_version, remote_info.version) != 0;
    }
    
    fclose(fp);
    return true;
} 