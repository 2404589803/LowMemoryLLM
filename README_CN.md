# LowMemoryLLM

[English Documentation](./README.md)

## 项目概述
LowMemoryLLM 是一个专为内存受限环境设计的轻量级大语言模型推理和训练实现。通过多种优化技术，提供高效的模型推理和训练能力，同时保持最小的内存占用。

## 主要特性
- 🚀 多种量化选项（INT8、INT4、INT2）
- 💾 智能内存管理，支持磁盘卸载
- 🔄 高效的注意力缓存机制
- 📦 集成 Hugging Face 模型支持
- 🛠️ 灵活的内存管理和优化
- 🌐 内置下载管理器，支持代理设置
- 🎯 硬件无关的训练支持
- 🔋 汇编优化的计算内核

## 技术特性
- 通过磁盘卸载和内存映射实现内存优化
- 可配置的量化支持，包括按通道量化
- KV缓存机制提升推理效率
- 支持多种激活函数（ReLU、GELU、SILU、SWISH）
- 针对低内存环境优化的矩阵运算
- 全面的张量操作和管理
- 跨平台训练支持，硬件加速优化
- 多种优化器实现（SGD、Adam、AdamW、RMSprop）
- 混合精度训练支持
- 梯度裁剪和归一化

## 系统要求
- 支持 C11 的 C 编译器
- CMake 构建系统
- 足够的磁盘空间用于模型权重和交换文件
- 网络连接用于模型下载
- （可选）支持 AVX2/NEON 的硬件加速

## 安装方法
```bash
git clone https://github.com/2404589803/LowMemoryLLM.git
cd LowMemoryLLM
mkdir build && cd build
cmake ..
make
```

## 使用方法

### 推理
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

### 训练
1. 配置训练参数：
```c
TrainingConfig train_config = {
    .batch_size = 32,
    .num_epochs = 100,
    .loss_type = LOSS_CROSS_ENTROPY,
    .optimizer = {
        .type = OPTIMIZER_ADAM,
        .learning_rate = 0.001f,
        .beta1 = 0.9f,
        .beta2 = 0.999f,
        .epsilon = 1e-8f
    },
    .gradient_clip_norm = 1.0f,
    .enable_mixed_precision = 1
};
```

2. 初始化训练系统：
```c
TrainingExtension extension = {
    .backward_matrix_multiply = backward_matrix_multiply_asm,
    .backward_vector_add = backward_vector_add_asm,
    // 设置其他函数指针...
};

training_init(device, &extension);
training_configure(&train_config);
```

3. 训练循环：
```c
TrainingState state = {0};
TrainingCallbacks callbacks = {
    .on_epoch_begin = my_epoch_begin_callback,
    .on_batch_end = my_batch_end_callback
};

for (size_t epoch = 0; epoch < train_config.num_epochs; epoch++) {
    for (size_t batch = 0; batch < num_batches; batch++) {
        void* inputs = prepare_batch_inputs(batch);
        void* targets = prepare_batch_targets(batch);
        training_step(model, inputs, targets, &state, &callbacks);
    }
    
    float metrics[2];
    training_evaluate(model, val_inputs, val_targets, metrics, 2);
}
```

## 硬件支持
- x86_64 架构，支持 AVX/AVX2 优化
- ARM64 架构，支持 NEON 优化
- 通用 CPU 实现
- 可扩展的设备抽象层

## 开源协议
本项目采用 MIT 开源协议 - 详见 [LICENSE](LICENSE) 文件。

## 贡献指南
欢迎提交 Pull Request 来帮助改进项目！

## 联系方式
[在此添加联系方式] 