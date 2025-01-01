# LowMemoryLLM

[English Documentation](./README.md)

## 项目概述
LowMemoryLLM 是一个专为内存受限环境设计的轻量级大语言模型推理实现。通过多种优化技术，提供高效的模型推理能力，同时保持最小的内存占用。

## 主要特性
- 🚀 多种量化选项（INT8、INT4、INT2）
- 💾 智能内存管理，支持磁盘卸载
- 🔄 高效的注意力缓存机制
- 📦 集成 Hugging Face 模型支持
- 🛠️ 灵活的内存管理和优化
- 🌐 内置下载管理器，支持代理设置

## 技术特性
- 通过磁盘卸载和内存映射实现内存优化
- 可配置的量化支持，包括按通道量化
- KV缓存机制提升推理效率
- 支持多种激活函数（ReLU、GELU、SILU、SWISH）
- 针对低内存环境优化的矩阵运算
- 全面的张量操作和管理

## 系统要求
- 支持 C11 的 C 编译器
- CMake 构建系统
- 足够的磁盘空间用于模型权重和交换文件
- 网络连接用于模型下载

## 安装方法
```bash
git clone https://github.com/2404589803/LowMemoryLLM.git
cd LowMemoryLLM
mkdir build && cd build
cmake ..
make
```

## 使用方法
1. 配置模型设置：
```c
LLMConfig config = {
    .vocab_size = 50257,
    .hidden_size = 768,
    .num_layers = 12,
    .max_seq_length = 1024,
    .batch_size = 1,
    .use_cache = 1
};
```

2. 初始化内存管理器：
```c
MemoryManager mem_manager = {
    .use_disk_offload = 1,
    .use_memory_map = 1,
    .prefetch_size = 1024 * 1024
};
```

3. 初始化并运行推理：
```c
llm_init(&config, &mem_manager);
llm_load_weights("path/to/weights");
// 运行推理...
```

## 开源协议
[在此添加开源协议信息]

## 贡献指南
欢迎提交 Pull Request 来帮助改进项目！

## 联系方式
[在此添加联系方式] 