# LowMemoryLLM

[中文文档](./README_CN.md)

## Overview
LowMemoryLLM is a lightweight inference implementation for Large Language Models (LLMs) designed specifically for memory-constrained environments. It provides efficient model inference with minimal memory footprint through various optimization techniques.

## Key Features
- 🚀 Multiple quantization options (INT8, INT4, INT2)
- 💾 Smart memory management with disk offloading
- 🔄 Efficient attention caching mechanism
- 📦 Hugging Face model integration
- 🛠️ Flexible memory management and optimization
- 🌐 Built-in download manager with proxy support

## Technical Features
- Memory optimization through disk offloading and memory mapping
- Configurable quantization with per-channel support
- KV-cache for efficient inference
- Support for various activation functions (ReLU, GELU, SILU, SWISH)
- Matrix operations optimized for low memory environments
- Comprehensive tensor operations and management

## Requirements
- C compiler with C11 support
- CMake for build system
- Sufficient disk space for model weights and swap files
- Network connection for model downloads

## Installation
```bash
git clone https://github.com/yourusername/LowMemoryLLM.git
cd LowMemoryLLM
mkdir build && cd build
cmake ..
make
```

## Usage
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

## License
[Add your license information here]

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Contact
[Add your contact information here] 