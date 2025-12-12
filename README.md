# LowMemoryLLM

A framework for running large language models on resource-constrained devices with full precision and high speed.

[中文文档](./README_CN.md)

---

## Overview

LowMemoryLLM aims to enable large language models (70B+ parameters) to run on low-end hardware (old smartphones, outdated GPUs, limited memory devices) while maintaining:

- **Full precision**: Bit-exact output matching FP32/FP16 inference
- **High speed**: Near-native inference performance
- **Minimal memory**: Run models that exceed available RAM/VRAM

This requires developing novel algorithms that go beyond existing optimization techniques.

---

## Problem Statement

| Constraint | Current Solutions | Limitation |
|------------|------------------|------------|
| Memory | Quantization (INT4/INT8) | Precision loss |
| Memory | Model pruning | Accuracy degradation |
| Speed | Smaller models | Reduced capability |
| Speed | Distillation | Not running original model |

**Our goal**: Achieve all three (speed, precision, low memory) simultaneously through new algorithmic approaches.

---

## Research Directions

### 1. Neural Subspace Projection

Discover low-dimensional structure in model weights and dynamically synthesize weights on-demand rather than storing them entirely.

- Hypothesis: Model knowledge exists in a compressed manifold
- Approach: Decompose weights into compact basis + input-dependent coefficients
- Target: 10x memory reduction with zero precision loss

### 2. Input-Adaptive Dynamic Computation

Not all inputs require full model capacity. Automatically determine computation complexity based on input difficulty.

- Simple queries → Minimal computation path
- Complex queries → Full model engagement
- No retraining required for existing models

### 3. Predictive Asynchronous Pipeline

Predict future computation paths based on attention patterns and overlap I/O with computation completely.

- Analyze attention patterns to predict weight access
- Prefetch weights before they are needed
- Goal: Zero effective memory latency

### 4. Cross-Request Computation Reuse

Build semantic computation cache to identify similarity between inputs and reuse intermediate computations.

- Cache KV states and intermediate activations
- Semantic similarity matching across requests
- Amortize computation cost over similar queries

### 5. Lossless Adaptive Precision

Dynamic precision selection per token and per layer with mathematical guarantees for precision preservation.

- Not quantization, but intelligent computation scheduling
- Higher precision for critical computations
- Lower precision where mathematically safe

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LowMemoryLLM Framework                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Inference  │  │  Training   │  │  Model Adaptation   │  │
│  │   Engine    │  │   Engine    │  │      Engine         │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
│         │                │                     │             │
│  ┌──────┴────────────────┴─────────────────────┴──────────┐  │
│  │              Core Algorithm Layer                       │  │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────────────┐  │  │
│  │  │ Subspace   │ │  Dynamic   │ │    Predictive      │  │  │
│  │  │ Projection │ │ Computation│ │    Pipeline        │  │  │
│  │  └────────────┘ └────────────┘ └────────────────────┘  │  │
│  └─────────────────────────┬───────────────────────────────┘  │
│                            │                                  │
│  ┌─────────────────────────┴───────────────────────────────┐  │
│  │           Hardware Abstraction Layer (HAL)              │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐   │  │
│  │  │   CPU   │ │   GPU   │ │   NPU   │ │   Storage   │   │  │
│  │  │(x86/ARM)│ │(OpenCL) │ │(Future) │ │  (Disk I/O) │   │  │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────────┘   │  │
│  └─────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              Model Integration Layer                     │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │  │
│  │  │ HuggingFace  │  │   Custom     │  │    GGUF      │   │  │
│  │  │   Loader     │  │   Formats    │  │   Support    │   │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘   │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Target Performance

| Model | Device | Active Memory | Target Speed | Precision |
|-------|--------|---------------|--------------|-----------|
| LLaMA-2 7B | Phone (4GB RAM) | <2GB | 10+ tokens/s | Lossless |
| LLaMA-2 70B | Laptop (16GB RAM) | <8GB | 2+ tokens/s | Lossless |
| Mixtral 8x7B | PC (8GB RAM) | <4GB | 5+ tokens/s | Lossless |
| 100B+ Models | Desktop (32GB RAM) | <16GB | 1+ tokens/s | Lossless |

---

## Implementation Status

### Completed

- **Hardware Abstraction Layer (HAL)**
  - CPU support with AVX2/NEON optimization
  - GPU support via OpenCL
  - Unified memory management interface

- **Quantization Infrastructure**
  - INT8/INT4/FP16/FP8 quantization
  - Per-channel quantization
  - Dynamic quantization framework
  - Quantization-Aware Training (QAT)

- **KV-Cache Management**
  - Disk offloading
  - Sliding window rotation
  - Cache compression

- **Mixed Precision Training**
  - Dynamic loss scaling
  - FP32 weight backup
  - Overflow detection

- **Model Integration**
  - HuggingFace model loader
  - Tokenizers (BPE, WordPiece, Unigram, SentencePiece)

### In Development

- **Core Algorithms**
  - Neural Subspace Projection
  - Input-Adaptive Dynamic Computation
  - Predictive Asynchronous Pipeline
  - Cross-Request Computation Reuse

- **Transformer Implementation**
  - Self-Attention mechanism
  - Feed-Forward Networks
  - Layer Normalization
  - Complete inference pipeline

- **Memory Management**
  - Layer-wise weight streaming
  - Predictive prefetching
  - Memory pool with LRU eviction

---

## Publication Targets

- Algorithm design: NeurIPS, ICML, ICLR
- Systems optimization: MLSys, OSDI, SOSP

---

## Installation

```bash
git clone https://github.com/2404589803/LowMemoryLLM.git
cd LowMemoryLLM
mkdir build && cd build
cmake ..
make
```

## Requirements

- C compiler with C11 support
- CMake 3.10+
- (Optional) OpenCL for GPU support
- (Optional) AVX2/NEON capable CPU

---

## Contributing

Contributions welcome in the following areas:
- Algorithm research and implementation
- Performance optimization
- Hardware backend support
- Theoretical analysis

---

## License

MIT License - see [LICENSE](LICENSE) file.
