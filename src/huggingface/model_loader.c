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
    if (json_object_object_get_ex(root, "hidden_size", &hidden_size_obj)) {
        size_t hidden_size = json_object_get_int(hidden_size_obj);
        // TODO: 设置模型参数
    }
    
    // 解析层配置
    json_object* num_layers_obj;
    if (json_object_object_get_ex(root, "num_hidden_layers", &num_layers_obj)) {
        model->num_layers = json_object_get_int(num_layers_obj);
        model->layers = (LayerParams**)malloc(sizeof(LayerParams*) * model->num_layers);
        if (!model->layers) {
            json_object_put(root);
            return -1;
        }
    }
    
    json_object_put(root);
    return 0;
}

// 下载模型权重
static int download_model_weights(const ModelConfig* config, HFModel* model) {
    // 创建缓存目录
    if (config->cache_dir) {
        char cmd[512];
        snprintf(cmd, sizeof(cmd), "mkdir -p %s", config->cache_dir);
        system(cmd);
    }
    
    // 下载每一层的权重
    for (size_t i = 0; i < model->num_layers; i++) {
        char url[512];
        snprintf(url, sizeof(url),
                "https://huggingface.co/%s/resolve/%s/pytorch_model-%d-of-%d.bin",
                config->model_name,
                config->revision ? config->revision : "main",
                (int)i + 1, (int)model->num_layers);
        
        // TODO: 实现权重下载和加载
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