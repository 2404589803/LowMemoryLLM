#include "model_loader.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <curl/curl.h>
#include <json-c/json.h>

// CURL写入回调
static size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t realsize = size * nmemb;
    char** response_ptr = (char**)userp;
    
    *response_ptr = realloc(*response_ptr, realsize + 1);
    if (*response_ptr == NULL) return 0;
    
    memcpy(*response_ptr, contents, realsize);
    (*response_ptr)[realsize] = 0;
    return realsize;
}

// 初始化模型加载器
int hf_model_init(void) {
    curl_global_init(CURL_GLOBAL_DEFAULT);
    return 0;
}

// 清理模型加载器
void hf_model_cleanup(void) {
    curl_global_cleanup();
}

// 从Hugging Face Hub下载模型配置
static char* download_model_config(const ModelConfig* config) {
    CURL* curl = curl_easy_init();
    if (!curl) return NULL;
    
    char url[512];
    snprintf(url, sizeof(url), 
             "https://huggingface.co/%s/resolve/%s/config.json",
             config->model_name,
             config->revision ? config->revision : "main");
    
    char* response = NULL;
    
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    
    if (config->use_auth_token) {
        char auth_header[256];
        snprintf(auth_header, sizeof(auth_header), 
                "Authorization: Bearer %s", config->auth_token);
        struct curl_slist* headers = NULL;
        headers = curl_slist_append(headers, auth_header);
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    }
    
    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    
    if (res != CURLE_OK) {
        free(response);
        return NULL;
    }
    
    return response;
}

// 解析模型配置
static int parse_model_config(const char* config_json, HFModel* model) {
    json_object* root = json_tokener_parse(config_json);
    if (!root) return -1;
    
    // 解析基本信息
    json_object* hidden_size_obj;
    json_object* vocab_size_obj;
    json_object* num_attention_heads_obj;
    json_object* intermediate_size_obj;
    json_object* max_position_embeddings_obj;
    json_object* model_type_obj;
    size_t hidden_size = 0;
    size_t vocab_size = 0;
    size_t num_attention_heads = 0;
    size_t intermediate_size = 0;
    size_t max_position_embeddings = 0;
    const char* model_type = NULL;
    
    // 获取各种参数
    if (json_object_object_get_ex(root, "hidden_size", &hidden_size_obj)) {
        hidden_size = json_object_get_int(hidden_size_obj);
    }
    
    if (json_object_object_get_ex(root, "vocab_size", &vocab_size_obj)) {
        vocab_size = json_object_get_int(vocab_size_obj);
    }
    
    if (json_object_object_get_ex(root, "num_attention_heads", &num_attention_heads_obj)) {
        num_attention_heads = json_object_get_int(num_attention_heads_obj);
    }
    
    if (json_object_object_get_ex(root, "intermediate_size", &intermediate_size_obj)) {
        intermediate_size = json_object_get_int(intermediate_size_obj);
    }
    
    if (json_object_object_get_ex(root, "max_position_embeddings", &max_position_embeddings_obj)) {
        max_position_embeddings = json_object_get_int(max_position_embeddings_obj);
    }
    
    if (json_object_object_get_ex(root, "model_type", &model_type_obj)) {
        model_type = json_object_get_string(model_type_obj);
    }
    
    // 设置模型参数
    // 这里我们在模型结构中添加额外字段来存储这些参数
    // 注意：这需要扩展HFModel结构
    if (hidden_size > 0) model->hidden_size = hidden_size;
    if (vocab_size > 0) model->vocab_size = vocab_size;
    if (num_attention_heads > 0) model->num_attention_heads = num_attention_heads;
    if (intermediate_size > 0) model->intermediate_size = intermediate_size;
    if (max_position_embeddings > 0) model->max_position_embeddings = max_position_embeddings;
    if (model_type) model->model_type = strdup(model_type);
    
    // 解析层配置
    json_object* num_layers_obj;
    if (json_object_object_get_ex(root, "num_hidden_layers", &num_layers_obj)) {
        model->num_layers = json_object_get_int(num_layers_obj);
        model->layers = (LayerParams**)malloc(sizeof(LayerParams*) * model->num_layers);
        if (!model->layers) {
            json_object_put(root);
            return -1;
        }
        
        // 为每层分配结构
        for (size_t i = 0; i < model->num_layers; i++) {
            model->layers[i] = (LayerParams*)malloc(sizeof(LayerParams));
            if (!model->layers[i]) {
                for (size_t j = 0; j < i; j++) {
                    free(model->layers[j]);
                }
                free(model->layers);
                model->layers = NULL;
                json_object_put(root);
                return -1;
            }
            memset(model->layers[i], 0, sizeof(LayerParams));
        }
    }
    
    json_object_put(root);
    return 0;
}

// 下载文件
static int download_file(const char* url, const char* output_path, const char* auth_token) {
    CURL* curl = curl_easy_init();
    if (!curl) return -1;
    
    FILE* fp = fopen(output_path, "wb");
    if (!fp) {
        curl_easy_cleanup(curl);
        return -1;
    }
    
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, NULL);  // 使用默认回调
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
    
    if (auth_token) {
        char auth_header[256];
        snprintf(auth_header, sizeof(auth_header), 
                "Authorization: Bearer %s", auth_token);
        struct curl_slist* headers = NULL;
        headers = curl_slist_append(headers, auth_header);
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    }
    
    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    fclose(fp);
    
    return (res == CURLE_OK) ? 0 : -1;
}

// 下载模型权重
static int download_model_weights(const ModelConfig* config, HFModel* model) {
    // 创建缓存目录
    if (config->cache_dir) {
        char cmd[512];
        snprintf(cmd, sizeof(cmd), "mkdir -p %s", config->cache_dir);
        system(cmd);
    }
    
    // 创建模型目录
    char model_dir[512];
    snprintf(model_dir, sizeof(model_dir), "%s/%s",
             config->cache_dir ? config->cache_dir : ".",
             config->model_name);
    
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", model_dir);
    system(cmd);
    
    // 下载权重索引文件
    char index_url[512];
    snprintf(index_url, sizeof(index_url),
            "https://huggingface.co/%s/resolve/%s/pytorch_model.bin.index.json",
            config->model_name,
            config->revision ? config->revision : "main");
    
    char index_path[512];
    snprintf(index_path, sizeof(index_path), "%s/pytorch_model.bin.index.json", model_dir);
    
    if (download_file(index_url, index_path, config->auth_token) != 0) {
        // 如果没有索引文件，假设模型权重在单个文件中
        char weights_url[512];
        snprintf(weights_url, sizeof(weights_url),
                "https://huggingface.co/%s/resolve/%s/pytorch_model.bin",
                config->model_name,
                config->revision ? config->revision : "main");
        
        char weights_path[512];
        snprintf(weights_path, sizeof(weights_path), "%s/pytorch_model.bin", model_dir);
        
        return download_file(weights_url, weights_path, config->auth_token);
    } else {
        // 解析索引文件以获取分片信息
        FILE* fp = fopen(index_path, "r");
        if (!fp) return -1;
        
        // 读取文件内容
        fseek(fp, 0, SEEK_END);
        long file_size = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        
        char* buffer = (char*)malloc(file_size + 1);
        if (!buffer) {
            fclose(fp);
            return -1;
        }
        
        size_t read_size = fread(buffer, 1, file_size, fp);
        buffer[read_size] = '\0';
        fclose(fp);
        
        // 解析JSON
        json_object* root = json_tokener_parse(buffer);
        free(buffer);
        
        if (!root) return -1;
        
        // 获取权重文件列表
        json_object* weight_map;
        if (!json_object_object_get_ex(root, "weight_map", &weight_map)) {
            json_object_put(root);
            return -1;
        }
        
        // 创建已下载的文件集合，防止重复下载
        char** downloaded_files = NULL;
        size_t num_downloaded = 0;
        
        // 遍历权重映射
        json_object_object_foreach(weight_map, key, val) {
            const char* file_name = json_object_get_string(val);
            
            // 检查是否已下载
            int already_downloaded = 0;
            for (size_t i = 0; i < num_downloaded; i++) {
                if (strcmp(downloaded_files[i], file_name) == 0) {
                    already_downloaded = 1;
                    break;
                }
            }
            
            if (!already_downloaded) {
                // 下载权重文件
                char weights_url[512];
                snprintf(weights_url, sizeof(weights_url),
                        "https://huggingface.co/%s/resolve/%s/%s",
                        config->model_name,
                        config->revision ? config->revision : "main",
                        file_name);
                
                char weights_path[512];
                snprintf(weights_path, sizeof(weights_path), "%s/%s", model_dir, file_name);
                
                if (download_file(weights_url, weights_path, config->auth_token) != 0) {
                    // 下载失败
                    for (size_t i = 0; i < num_downloaded; i++) {
                        free(downloaded_files[i]);
                    }
                    free(downloaded_files);
                    json_object_put(root);
                    return -1;
                }
                
                // 添加到已下载列表
                downloaded_files = (char**)realloc(downloaded_files, 
                                                  (num_downloaded + 1) * sizeof(char*));
                downloaded_files[num_downloaded] = strdup(file_name);
                num_downloaded++;
            }
        }
        
        // 清理
        for (size_t i = 0; i < num_downloaded; i++) {
            free(downloaded_files[i]);
        }
        free(downloaded_files);
        json_object_put(root);
    }
    
    return 0;
}

// 从Hugging Face Hub下载模型
int hf_model_download(const ModelConfig* config) {
    if (!config || !config->model_name) return -1;
    
    // 下载模型配置
    char* config_json = download_model_config(config);
    if (!config_json) return -1;
    
    // 创建模型结构
    HFModel* model = (HFModel*)malloc(sizeof(HFModel));
    if (!model) {
        free(config_json);
        return -1;
    }
    memset(model, 0, sizeof(HFModel));
    
    // 复制配置
    memcpy(&model->config, config, sizeof(ModelConfig));
    model->config.model_name = strdup(config->model_name);
    if (config->revision)
        model->config.revision = strdup(config->revision);
    if (config->cache_dir)
        model->config.cache_dir = strdup(config->cache_dir);
    if (config->auth_token)
        model->config.auth_token = strdup(config->auth_token);
    
    // 解析配置
    if (parse_model_config(config_json, model) != 0) {
        free(config_json);
        hf_model_free(model);
        return -1;
    }
    
    free(config_json);
    
    // 下载权重
    if (download_model_weights(config, model) != 0) {
        hf_model_free(model);
        return -1;
    }
    
    return 0;
}

// 加载模型
int hf_model_load(HFModel** model, const ModelConfig* config, void* device) {
    if (!model || !config || !device) return -1;
    
    // 如果模型未下载，先下载
    if (hf_model_download(config) != 0) return -1;
    
    // 创建模型结构
    *model = (HFModel*)malloc(sizeof(HFModel));
    if (!*model) return -1;
    memset(*model, 0, sizeof(HFModel));
    
    // 设置设备
    (*model)->device = device;
    
    // 加载本地模型
    char model_path[512];
    snprintf(model_path, sizeof(model_path), "%s/%s",
             config->cache_dir ? config->cache_dir : ".",
             config->model_name);
    
    return hf_model_load_local(model, model_path, device);
}

// 释放模型
void hf_model_free(HFModel* model) {
    if (!model) return;
    
    // 释放配置字符串
    if (model->config.model_name)
        free((void*)model->config.model_name);
    if (model->config.revision)
        free((void*)model->config.revision);
    if (model->config.cache_dir)
        free((void*)model->config.cache_dir);
    if (model->config.auth_token)
        free((void*)model->config.auth_token);
    
    // 释放层参数
    if (model->layers) {
        for (size_t i = 0; i < model->num_layers; i++) {
            if (model->layers[i]) {
                if (model->layers[i]->weights)
                    ((HAL_Device*)model->device)->free_memory(model->layers[i]->weights);
                if (model->layers[i]->bias)
                    ((HAL_Device*)model->device)->free_memory(model->layers[i]->bias);
                if (model->layers[i]->shape)
                    free(model->layers[i]->shape);
                free(model->layers[i]);
            }
        }
        free(model->layers);
    }
    
    // 释放分词器
    if (model->tokenizer) {
        // TODO: 实现分词器释放
        extern void tokenizer_free(void* tokenizer);  // 来自tokenizer.h
        tokenizer_free(model->tokenizer);
        model->tokenizer = NULL;
    }
    
    // 释放模型类型
    if (model->model_type) {
        free(model->model_type);
    }
    
    free(model);
}

// 转换模型格式
int hf_model_convert(HFModel* model, WeightFormat target_format) {
    if (!model || !model->layers) return -1;
    
    for (size_t i = 0; i < model->num_layers; i++) {
        LayerParams* layer = model->layers[i];
        if (!layer || !layer->weights) continue;
        
        // 计算权重大小
        size_t total_size = 1;
        for (size_t j = 0; j < layer->num_dims; j++) {
            total_size *= layer->shape[j];
        }
        
        // 根据目标格式分配新内存
        size_t new_size;
        switch (target_format) {
            case WEIGHT_FORMAT_FP32:
                new_size = total_size * sizeof(float);
                break;
            case WEIGHT_FORMAT_FP16:
                new_size = total_size * sizeof(uint16_t);
                break;
            case WEIGHT_FORMAT_INT8:
                new_size = total_size * sizeof(int8_t);
                break;
            case WEIGHT_FORMAT_INT4:
                new_size = (total_size + 1) / 2;  // 4位压缩
                break;
            default:
                return -1;
        }
        
        void* new_weights = ((HAL_Device*)model->device)->allocate_memory(new_size);
        if (!new_weights) return -1;
        
        // 转换权重格式
        // TODO: 实现不同格式间的转换
        
        // 更新层参数
        ((HAL_Device*)model->device)->free_memory(layer->weights);
        layer->weights = new_weights;
        layer->format = target_format;
    }
    
    return 0;
}

// 保存模型到本地
int hf_model_save(const HFModel* model, const char* path) {
    if (!model || !path) return -1;
    
    // 创建目录
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", path);
    system(cmd);
    
    // 保存配置
    char config_path[512];
    snprintf(config_path, sizeof(config_path), "%s/config.json", path);
    FILE* fp = fopen(config_path, "w");
    if (!fp) return -1;
    
    // TODO: 生成配置JSON
    fprintf(fp, "{\n");
    
    // 基本信息
    if (model->model_type) {
        fprintf(fp, "  \"model_type\": \"%s\",\n", model->model_type);
    }
    
    fprintf(fp, "  \"hidden_size\": %zu,\n", model->hidden_size);
    fprintf(fp, "  \"num_hidden_layers\": %zu,\n", model->num_layers);
    fprintf(fp, "  \"num_attention_heads\": %zu,\n", model->num_attention_heads);
    
    if (model->intermediate_size > 0) {
        fprintf(fp, "  \"intermediate_size\": %zu,\n", model->intermediate_size);
    }
    
    fprintf(fp, "  \"vocab_size\": %zu,\n", model->vocab_size);
    
    if (model->max_position_embeddings > 0) {
        fprintf(fp, "  \"max_position_embeddings\": %zu,\n", model->max_position_embeddings);
    }
    
    // 添加默认值
    fprintf(fp, "  \"layer_norm_eps\": 1e-12,\n");
    fprintf(fp, "  \"initializer_range\": 0.02,\n");
    fprintf(fp, "  \"architectures\": [\"AutoModel\"],\n");
    fprintf(fp, "  \"output_past\": true,\n");
    fprintf(fp, "  \"output_hidden_states\": false,\n");
    fprintf(fp, "  \"output_attentions\": false,\n");
    fprintf(fp, "  \"tie_word_embeddings\": true,\n");
    fprintf(fp, "  \"pruned_heads\": {},\n");
    fprintf(fp, "  \"is_encoder_decoder\": false,\n");
    fprintf(fp, "  \"torchscript\": false,\n");
    fprintf(fp, "  \"use_bfloat16\": false,\n");
    fprintf(fp, "  \"is_encoder_decoder\": false,\n");
    fprintf(fp, "  \"model_format\": \"pytorch\"\n");
    
    fprintf(fp, "}\n");
    
    fclose(fp);
    
    // 保存权重
    for (size_t i = 0; i < model->num_layers; i++) {
        char weight_path[512];
        snprintf(weight_path, sizeof(weight_path), 
                "%s/pytorch_model-%d-of-%d.bin",
                path, (int)i + 1, (int)model->num_layers);
        
        fp = fopen(weight_path, "wb");
        if (!fp) return -1;
        
        LayerParams* layer = model->layers[i];
        if (layer && layer->weights) {
            // 计算权重大小
            size_t total_size = 1;
            for (size_t j = 0; j < layer->num_dims; j++) {
                total_size *= layer->shape[j];
            }
            
            // 分配临时缓冲区
            void* temp = malloc(total_size * sizeof(float));
            if (!temp) {
                fclose(fp);
                return -1;
            }
            
            // 从设备复制数据
            ((HAL_Device*)model->device)->memcpy_from_device(
                temp, layer->weights, total_size * sizeof(float));
            
            // 写入文件
            fwrite(temp, sizeof(float), total_size, fp);
            free(temp);
        }
        
        fclose(fp);
    }
    
    return 0;
}

// 加载本地模型
int hf_model_load_local(HFModel** model, const char* path, void* device) {
    if (!model || !path || !device) return -1;
    
    // 读取配置
    char config_path[512];
    snprintf(config_path, sizeof(config_path), "%s/config.json", path);
    
    FILE* fp = fopen(config_path, "r");
    if (!fp) return -1;
    
    // 读取配置文件内容
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    char* config_json = (char*)malloc(fsize + 1);
    if (!config_json) {
        fclose(fp);
        return -1;
    }
    
    fread(config_json, 1, fsize, fp);
    config_json[fsize] = 0;
    fclose(fp);
    
    // 创建模型结构
    *model = (HFModel*)malloc(sizeof(HFModel));
    if (!*model) {
        free(config_json);
        return -1;
    }
    memset(*model, 0, sizeof(HFModel));
    
    // 设置设备
    (*model)->device = device;
    
    // 解析配置
    if (parse_model_config(config_json, *model) != 0) {
        free(config_json);
        hf_model_free(*model);
        return -1;
    }
    
    free(config_json);
    
    // 加载权重
    for (size_t i = 0; i < (*model)->num_layers; i++) {
        char weight_path[512];
        snprintf(weight_path, sizeof(weight_path),
                "%s/pytorch_model-%d-of-%d.bin",
                path, (int)i + 1, (int)(*model)->num_layers);
        
        fp = fopen(weight_path, "rb");
        if (!fp) {
            hf_model_free(*model);
            return -1;
        }
        
        // 获取文件大小
        fseek(fp, 0, SEEK_END);
        size_t fsize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        
        // 分配临时缓冲区
        void* temp = malloc(fsize);
        if (!temp) {
            fclose(fp);
            hf_model_free(*model);
            return -1;
        }
        
        // 读取权重
        if (fread(temp, 1, fsize, fp) != fsize) {
            free(temp);
            fclose(fp);
            hf_model_free(*model);
            return -1;
        }
        
        fclose(fp);
        
        // 分配设备内存并上传权重
        void* device_mem = ((HAL_Device*)device)->allocate_memory(fsize);
        if (!device_mem) {
            free(temp);
            hf_model_free(*model);
            return -1;
        }
        
        ((HAL_Device*)device)->memcpy_to_device(device_mem, temp, fsize);
        free(temp);
        
        // 设置层参数
        (*model)->layers[i] = (LayerParams*)malloc(sizeof(LayerParams));
        if (!(*model)->layers[i]) {
            ((HAL_Device*)device)->free_memory(device_mem);
            hf_model_free(*model);
            return -1;
        }
        
        (*model)->layers[i]->weights = device_mem;
        (*model)->layers[i]->format = WEIGHT_FORMAT_FP32;
        // TODO: 设置其他层参数
    }
    
    return 0;
}

// 获取模型信息
const char* hf_model_get_info(const HFModel* model) {
    if (!model) return NULL;
    
    // TODO: 生成模型信息字符串
    return "Model information not implemented";
}

// 验证模型完整性
int hf_model_verify(const HFModel* model) {
    if (!model || !model->layers) return -1;
    
    // 检查每一层
    for (size_t i = 0; i < model->num_layers; i++) {
        LayerParams* layer = model->layers[i];
        if (!layer || !layer->weights) return -1;
        
        // TODO: 实现更详细的验证
    }
    
    return 0;
}

// 获取层参数
const LayerParams* hf_model_get_layer(const HFModel* model, size_t layer_idx) {
    if (!model || !model->layers || layer_idx >= model->num_layers) return NULL;
    return model->layers[layer_idx];
}

// 获取分词器
void* hf_model_get_tokenizer(const HFModel* model) {
    if (!model) return NULL;
    return model->tokenizer;
} 