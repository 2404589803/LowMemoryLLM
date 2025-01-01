#include "low_memory_llm.h"
#include <stdio.h>
#include <windows.h>

int main() {
    // 设置控制台编码
    SetConsoleOutputCP(65001);
    
    // 配置模型
    LLMConfig config = {
        .vocab_size = 100000,         // MiniCPM3词表大小
        .hidden_size = 4096,          // 隐藏层大小
        .num_layers = 32,             // 层数
        .max_seq_length = 2048,       // 最大序列长度
        .batch_size = 1,              // 批次大小
        .quant_config = {
            .quant_type = QUANT_INT8, // 使用INT8量化
            .symmetric = 1,           // 对称量化
            .per_channel = 1,         // 按通道量化
        },
        .act_type = ACT_SILU,        // 使用SiLU激活函数
        .model_path = "model",        // 模型目录
        .use_cache = 1               // 使用KV缓存
    };
    
    // 配置内存管理
    MemoryManager mem_manager = {
        .available_ram = 2ULL * 1024 * 1024 * 1024,  // 限制使用2GB内存
        .page_size = 4096,                           // 4KB页面大小
        .use_disk_offload = 1,                       // 启用磁盘交换
        .swap_file_path = "model/swap",              // 交换文件路径
        .prefetch_size = 32 * 1024 * 1024,          // 32MB预取大小
        .use_memory_map = 1                          // 使用内存映射
    };
    
    printf("初始化模型...\n");
    if (!llm_init(&config, &mem_manager)) {
        printf("初始化失败：%s\n", llm_get_error());
        return 1;
    }
    
    // 加载模型权重
    printf("加载模型权重...\n");
    if (!llm_load_weights("model/pytorch_model.bin")) {
        printf("加载权重失败：%s\n", llm_get_error());
        llm_cleanup();
        return 1;
    }
    
    // 准备输入
    const int input_tokens[] = {1, 2025, 2};  // <s>你好</s>
    size_t input_length = sizeof(input_tokens) / sizeof(input_tokens[0]);
    
    // 分配输出缓冲区
    int output_tokens[2048];  // 最大输出长度
    size_t max_length = 100;  // 生成100个token
    
    // 生成文本
    printf("开始生成...\n");
    if (!llm_generate(input_tokens, input_length, output_tokens, max_length, 0.7f, 0.9f)) {
        printf("生成失败：%s\n", llm_get_error());
        llm_cleanup();
        return 1;
    }
    
    // 清理资源
    llm_cleanup();
    printf("测试完成\n");
    
    return 0;
} 