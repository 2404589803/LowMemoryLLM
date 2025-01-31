# LowMemoryLLM

[中文文档](./README_CN.md)

## Overview
LowMemoryLLM is a lightweight inference and training implementation for Large Language Models (LLMs) designed specifically for memory-constrained environments. It provides efficient model inference and training with minimal memory footprint through various optimization techniques.

## Key Features
- 🚀 Multiple quantization options (INT8, INT4, INT2)
- 💾 Smart memory management with disk offloading
- 🔄 Efficient attention caching mechanism
- 📦 Hugging Face model integration
- 🛠️ Flexible memory management and optimization
- 🌐 Built-in download manager with proxy support
- 🎯 Hardware-agnostic training support
- 🔋 Assembly-optimized computation kernels

## Technical Features
- Memory optimization through disk offloading and memory mapping
- Configurable quantization with per-channel support
- KV-cache for efficient inference
- Support for various activation functions (ReLU, GELU, SILU, SWISH)
- Matrix operations optimized for low memory environments
- Comprehensive tensor operations and management
- Cross-platform training support with hardware acceleration
- Multiple optimizer implementations (SGD, Adam, AdamW, RMSprop)
- Mixed precision training support
- Gradient clipping and normalization

## Requirements
- C compiler with C11 support
- CMake for build system
- Sufficient disk space for model weights and swap files
- Network connection for model downloads
- (Optional) AVX2/NEON support for hardware acceleration

## Installation
```bash
git clone https://github.com/2404589803/LowMemoryLLM.git
cd LowMemoryLLM
mkdir build && cd build
cmake ..
make
```

## Usage

### Inference
1. Configure model settings:
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

2. Initialize memory manager:
```c
MemoryManager mem_manager = {
    .use_disk_offload = 1,
    .use_memory_map = 1,
    .prefetch_size = 1024 * 1024
};
```

3. Initialize and run inference:
```c
llm_init(&config, &mem_manager);
llm_load_weights("path/to/weights");
// Run inference...
```

### Training
1. Configure training settings:
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

2. Initialize training system:
```c
TrainingExtension extension = {
    .backward_matrix_multiply = backward_matrix_multiply_asm,
    .backward_vector_add = backward_vector_add_asm,
    // Set other function pointers...
};

training_init(device, &extension);
training_configure(&train_config);
```

3. Training loop:
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

## Hardware Support
- x86_64 with AVX/AVX2 optimization
- ARM64 with NEON optimization
- Generic CPU fallback implementation
- Extensible device abstraction layer

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Contact
[Add your contact information here] 