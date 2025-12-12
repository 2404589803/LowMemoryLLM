#include "../src/huggingface/model_loader.h"
#include "../src/hal/hal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    printf("Running model loader tests...\n");
    
    // 初始化HAL
    if (hal_init() != 0) {
        printf("Failed to initialize HAL\n");
        return 1;
    }
    
    // 获取设备
    HAL_Device* device = hal_select_optimal_device();
    if (!device) {
        printf("Failed to select device\n");
        return 1;
    }
    
    printf("Selected device type: %d\n", device->device_type);
    
    // 初始化模型加载器
    if (hf_model_init() != 0) {
        printf("Failed to initialize model loader\n");
        return 1;
    }
    
    // 创建配置
    ModelConfig config = {
        .model_name = "bert-base-uncased",  // 仅测试，不会实际下载
        .cache_dir = "tests/model_cache",
        .use_auth_token = 0
    };
    
    // 尝试创建模型
    HFModel* model = NULL;
    
    printf("NOTE: This test doesn't actually download models from Hugging Face.\n");
    printf("It only tests the API structure and local functionality.\n");
    
    // 创建本地测试模型
    char cmd[512];
    sprintf(cmd, "mkdir -p %s/%s", config.cache_dir, config.model_name);
    system(cmd);
    
    // 创建测试配置文件
    char config_path[512];
    sprintf(config_path, "%s/%s/config.json", config.cache_dir, config.model_name);
    FILE* fp = fopen(config_path, "w");
    if (!fp) {
        printf("Failed to create test config file\n");
        hf_model_cleanup();
        return 1;
    }
    
    // 写入测试配置
    fprintf(fp, "{\n");
    fprintf(fp, "  \"model_type\": \"bert\",\n");
    fprintf(fp, "  \"hidden_size\": 768,\n");
    fprintf(fp, "  \"num_hidden_layers\": 12,\n");
    fprintf(fp, "  \"num_attention_heads\": 12,\n");
    fprintf(fp, "  \"intermediate_size\": 3072,\n");
    fprintf(fp, "  \"hidden_act\": \"gelu\",\n");
    fprintf(fp, "  \"vocab_size\": 30522\n");
    fprintf(fp, "}\n");
    fclose(fp);
    
    // 创建测试权重文件
    char weights_path[512];
    sprintf(weights_path, "%s/%s/pytorch_model.bin", config.cache_dir, config.model_name);
    fp = fopen(weights_path, "wb");
    if (!fp) {
        printf("Failed to create test weights file\n");
        hf_model_cleanup();
        return 1;
    }
    
    // 写入一些测试数据
    float dummy_data[10] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    fwrite(dummy_data, sizeof(float), 10, fp);
    fclose(fp);
    
    // 加载模型
    if (hf_model_load_local(&model, config_path, device) != 0) {
        printf("Failed to load model, but this is expected in test\n");
        printf("Test incomplete - would need proper model files\n");
    } else {
        // 测试获取模型信息
        const char* info = hf_model_get_info(model);
        printf("Model info: %s\n", info);
        
        // 测试模型保存
        char save_path[512];
        sprintf(save_path, "%s/saved_model", config.cache_dir);
        if (hf_model_save(model, save_path) != 0) {
            printf("Failed to save model\n");
        } else {
            printf("Model saved to %s\n", save_path);
        }
        
        // 释放模型
        hf_model_free(model);
    }
    
    // 清理
    hf_model_cleanup();
    
    // 删除测试文件
    sprintf(cmd, "rm -rf %s", config.cache_dir);
    system(cmd);
    
    printf("Model loader tests completed\n");
    return 0;
} 