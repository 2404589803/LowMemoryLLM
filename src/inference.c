#include "low_memory_llm.h"
#include <stdio.h>
#include <string.h>
#include <windows.h>
#include <zlib.h>
#include <direct.h>
#include <time.h>
#include <math.h>
#include <float.h>

#define WEIGHT_CHUNK_SIZE (4 * 1024)  // 4KB权重块
#define VM_PAGE_SIZE (4 * 1024)       // 4KB虚拟内存页
#define MAX_ACTIVE_PAGES 1024
#define SWAP_FILE_PREFIX "weight_page"
#define CACHE_DIR "weight_cache"
#define PAGE_SIZE (4 * 1024 * 1024)  // 4MB per page

static LLMState* g_model_state = NULL;
static WeightCache g_weight_cache = {0};

// 初始化控制台编码
static void init_console_encoding(void) {
    // 设置控制台输出代码页为UTF-8
    SetConsoleOutputCP(65001);
    // 设置控制台输入代码页为UTF-8
    SetConsoleCP(65001);
    
    // 获取标准输出句柄
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut != INVALID_HANDLE_VALUE) {
        DWORD dwMode = 0;
        GetConsoleMode(hOut, &dwMode);
        // 启用虚拟终端序列
        dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
        SetConsoleMode(hOut, dwMode);
    }
}

// 打印UTF-8编码的消息
static void print_utf8(const char* message) {
    // 获取当前代码页
    UINT oldcp = GetConsoleOutputCP();
    // 临时切换到UTF-8
    SetConsoleOutputCP(65001);
    printf("%s", message);
    // 恢复原来的代码页
    SetConsoleOutputCP(oldcp);
}

// 初始化权重缓存
static int init_weight_cache(void) {
    memset(&g_weight_cache, 0, sizeof(WeightCache));
    snprintf(g_weight_cache.cache_dir, sizeof(g_weight_cache.cache_dir), "%s", CACHE_DIR);
    _mkdir(g_weight_cache.cache_dir);
    return 0;
}

// 生成交换文件名
static void generate_swap_filename(char* filename, size_t size, uint64_t page_id) {
    snprintf(filename, size, "%s/%s%llu.bin", 
             g_weight_cache.cache_dir, SWAP_FILE_PREFIX, page_id);
}

// 将页面写入交换文件
static int write_page_to_swap(VMPage* page) {
    if (!page->is_dirty) return 0;

    char filename[256];
    generate_swap_filename(filename, sizeof(filename), page->page_id);

    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        print_utf8("错误：无法创建交换文件\n");
        return -1;
    }

    // 压缩数据
    uLong compressed_size = compressBound(page->size);
    unsigned char* compressed_data = (unsigned char*)malloc(compressed_size);
    if (!compressed_data) {
        fclose(fp);
        return -1;
    }

    if (compress2(compressed_data, &compressed_size, page->data, page->size, 
                 Z_BEST_COMPRESSION) != Z_OK) {
        free(compressed_data);
        fclose(fp);
        return -1;
    }

    size_t written = fwrite(compressed_data, 1, compressed_size, fp);
    free(compressed_data);
    fclose(fp);

    return (written == compressed_size) ? 0 : -1;
}

// 从交换文件读取页面
static int read_page_from_swap(VMPage* page) {
    char filename[256];
    generate_swap_filename(filename, sizeof(filename), page->page_id);

    FILE* fp = fopen(filename, "rb");
    if (!fp) return -1;

    // 获取压缩数据大小
    fseek(fp, 0, SEEK_END);
    size_t compressed_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    unsigned char* compressed_data = (unsigned char*)malloc(compressed_size);
    if (!compressed_data) {
        fclose(fp);
        return -1;
    }

    if (fread(compressed_data, 1, compressed_size, fp) != compressed_size) {
        free(compressed_data);
        fclose(fp);
        return -1;
    }
    fclose(fp);

    // 解压数据
    uLong decompressed_size = page->size;
    if (uncompress(page->data, &decompressed_size, compressed_data, compressed_size) != Z_OK) {
        free(compressed_data);
        return -1;
    }

    free(compressed_data);
    return 0;
}

// 查找最旧的页面
static VMPage* find_oldest_page(void) {
    if (g_weight_cache.active_pages == 0) return NULL;

    VMPage* oldest = &g_weight_cache.pages[0];
    for (size_t i = 1; i < g_weight_cache.active_pages; i++) {
        if (g_weight_cache.pages[i].last_access < oldest->last_access) {
            oldest = &g_weight_cache.pages[i];
        }
    }
    return oldest;
}

// 获取或创建页面
static VMPage* get_or_create_page(uint64_t page_id, size_t size) {
    // 检查是否已存在
    for (size_t i = 0; i < g_weight_cache.active_pages; i++) {
        if (g_weight_cache.pages[i].page_id == page_id) {
            g_weight_cache.pages[i].last_access = time(NULL);
            return &g_weight_cache.pages[i];
        }
    }

    // 如果达到最大页面数，交换出最旧的页面
    if (g_weight_cache.active_pages >= MAX_ACTIVE_PAGES) {
        VMPage* oldest = find_oldest_page();
        if (oldest) {
            if (oldest->is_dirty) {
                write_page_to_swap(oldest);
            }
            if (oldest->data) {
                free(oldest->data);
            }
            oldest->data = malloc(size);
            if (!oldest->data) return NULL;
            oldest->size = size;
            oldest->page_id = page_id;
            oldest->is_dirty = 0;
            oldest->last_access = time(NULL);
            read_page_from_swap(oldest);
            return oldest;
        }
        return NULL;
    }

    // 创建新页面
    VMPage* page = &g_weight_cache.pages[g_weight_cache.active_pages++];
    page->data = malloc(size);
    if (!page->data) return NULL;
    page->size = size;
    page->page_id = page_id;
    page->is_dirty = 0;
    page->last_access = time(NULL);
    return page;
}

// 读取权重数据
static int read_weight_data(void* dest, size_t offset, size_t size) {
    uint64_t page_id = offset / VM_PAGE_SIZE;
    size_t page_offset = offset % VM_PAGE_SIZE;
    size_t remaining = size;
    char* dest_ptr = (char*)dest;

    while (remaining > 0) {
        size_t chunk_size = (remaining < VM_PAGE_SIZE - page_offset) ? 
                           remaining : (VM_PAGE_SIZE - page_offset);

        VMPage* page = get_or_create_page(page_id, VM_PAGE_SIZE);
        if (!page) return -1;

        memcpy(dest_ptr, (char*)page->data + page_offset, chunk_size);
        
        dest_ptr += chunk_size;
        remaining -= chunk_size;
        page_offset = 0;
        page_id++;
    }

    return 0;
}

// 写入权重数据
static int write_weight_data(const void* src, size_t offset, size_t size) {
    uint64_t page_id = offset / VM_PAGE_SIZE;
    size_t page_offset = offset % VM_PAGE_SIZE;
    size_t remaining = size;
    const char* src_ptr = (const char*)src;

    while (remaining > 0) {
        size_t chunk_size = (remaining < VM_PAGE_SIZE - page_offset) ? 
                           remaining : (VM_PAGE_SIZE - page_offset);

        VMPage* page = get_or_create_page(page_id, VM_PAGE_SIZE);
        if (!page) return -1;

        memcpy((char*)page->data + page_offset, src_ptr, chunk_size);
        page->is_dirty = 1;
        
        src_ptr += chunk_size;
        remaining -= chunk_size;
        page_offset = 0;
        page_id++;
    }

    return 0;
}

// 流式矩阵乘法 - 使用极小内存块进行计算
static int stream_matrix_multiply(const Tensor* a, const Tensor* b, Tensor* c) {
    if (!a || !b || !c) return -1;
    
    size_t M = a->shape[0];
    size_t K = a->shape[1];
    size_t N = b->shape[1];
    
    // 验证维度
    if (b->shape[0] != K || c->shape[0] != M || c->shape[1] != N) {
        return -1;
    }
    
    // 使用1KB的缓冲区
    #define BLOCK_SIZE 32  // 32个float = 128字节
    float row_buffer[BLOCK_SIZE];
    float col_buffer[BLOCK_SIZE];
    float result_buffer[BLOCK_SIZE];
    
    // 分块计算
    for (size_t i = 0; i < M; i++) {
        // 每次处理一行
        for (size_t j = 0; j < N; j++) {
            float sum = 0.0f;
            
            // 分块加载和计算
            for (size_t k = 0; k < K; k += BLOCK_SIZE) {
                size_t block_size = (k + BLOCK_SIZE > K) ? (K - k) : BLOCK_SIZE;
                
                // 加载A矩阵的一块
                if (read_weight_data(row_buffer, 
                    (i * K + k) * sizeof(float), 
                    block_size * sizeof(float)) != 0) {
                    return -1;
                }
                
                // 加载B矩阵的一块
                if (read_weight_data(col_buffer,
                    (k * N + j) * sizeof(float),
                    block_size * sizeof(float)) != 0) {
                    return -1;
                }
                
                // 计算点积
                for (size_t b = 0; b < block_size; b++) {
                    sum += row_buffer[b] * col_buffer[b];
                }
            }
            
            // 写入结果
            if (write_weight_data(&sum, 
                (i * N + j) * sizeof(float),
                sizeof(float)) != 0) {
                return -1;
            }
        }
    }
    
    return 0;
}

// 流式注意力计算 - 使用极小内存块
static int stream_attention(const Tensor* query, const Tensor* key, const Tensor* value,
                          Tensor* output, float scale) {
    if (!query || !key || !value || !output) return -1;
    
    size_t seq_len = query->shape[0];
    size_t head_dim = query->shape[1];
    
    // 验证维度
    if (key->shape[0] != seq_len || key->shape[1] != head_dim ||
        value->shape[0] != seq_len || value->shape[1] != head_dim ||
        output->shape[0] != seq_len || output->shape[1] != head_dim) {
        return -1;
    }
    
    // 使用1KB的缓冲区
    #define ATT_BLOCK_SIZE 16  // 16个float = 64字节
    float q_buffer[ATT_BLOCK_SIZE];
    float k_buffer[ATT_BLOCK_SIZE];
    float v_buffer[ATT_BLOCK_SIZE];
    float scores[ATT_BLOCK_SIZE];
    float max_score = -FLT_MAX;
    float sum_exp = 0.0f;
    
    // 分块计算注意力
    for (size_t i = 0; i < seq_len; i++) {
        // 计算当前query与所有key的得分
        for (size_t j = 0; j < seq_len; j++) {
            float score = 0.0f;
            
            // 分块计算点积
            for (size_t k = 0; k < head_dim; k += ATT_BLOCK_SIZE) {
                size_t block_size = (k + ATT_BLOCK_SIZE > head_dim) ? 
                                  (head_dim - k) : ATT_BLOCK_SIZE;
                
                // 加载query块
                if (read_weight_data(q_buffer,
                    (i * head_dim + k) * sizeof(float),
                    block_size * sizeof(float)) != 0) {
                    return -1;
                }
                
                // 加载key块
                if (read_weight_data(k_buffer,
                    (j * head_dim + k) * sizeof(float),
                    block_size * sizeof(float)) != 0) {
                    return -1;
                }
                
                // 计算点积
                for (size_t b = 0; b < block_size; b++) {
                    score += q_buffer[b] * k_buffer[b];
                }
            }
            
            // 应用scale
            score *= scale;
            
            // 保存分数
            scores[j % ATT_BLOCK_SIZE] = score;
            
            // 更新最大分数
            if (score > max_score) {
                max_score = score;
            }
            
            // 如果缓冲区满或是最后一个，计算softmax
            if ((j + 1) % ATT_BLOCK_SIZE == 0 || j == seq_len - 1) {
                size_t block_size = ((j + 1) % ATT_BLOCK_SIZE) ? 
                                  ((j + 1) % ATT_BLOCK_SIZE) : ATT_BLOCK_SIZE;
                
                // 计算exp并累加
                for (size_t b = 0; b < block_size; b++) {
                    scores[b] = expf(scores[b] - max_score);
                    sum_exp += scores[b];
                }
                
                // 加载value块并计算加权和
                for (size_t k = 0; k < head_dim; k += ATT_BLOCK_SIZE) {
                    size_t dim_block_size = (k + ATT_BLOCK_SIZE > head_dim) ?
                                          (head_dim - k) : ATT_BLOCK_SIZE;
                    
                    // 初始化输出缓冲区
                    memset(v_buffer, 0, dim_block_size * sizeof(float));
                    
                    // 对每个value进行加权
                    for (size_t b = 0; b < block_size; b++) {
                        size_t v_idx = (j - block_size + b + 1) * head_dim + k;
                        
                        // 加载value
                        if (read_weight_data(v_buffer,
                            v_idx * sizeof(float),
                            dim_block_size * sizeof(float)) != 0) {
                            return -1;
                        }
                        
                        // 加权累加
                        float weight = scores[b] / sum_exp;
                        for (size_t d = 0; d < dim_block_size; d++) {
                            v_buffer[d] *= weight;
                        }
                        
                        // 写回结果
                        if (write_weight_data(v_buffer,
                            (i * head_dim + k) * sizeof(float),
                            dim_block_size * sizeof(float)) != 0) {
                            return -1;
                        }
                    }
                }
            }
        }
    }
    
    return 0;
}

// 激活函数 - GELU
static float gelu(float x) {
    // GELU近似实现
    return 0.5f * x * (1.0f + tanhf(0.797884f * (x + 0.044715f * x * x * x)));
}

// 流式前馈网络层
static int stream_ffn(const Tensor* input, const Tensor* weights1, const Tensor* weights2,
                     const Tensor* bias1, const Tensor* bias2, Tensor* output) {
    if (!input || !weights1 || !weights2 || !bias1 || !bias2 || !output) return -1;
    
    size_t seq_len = input->shape[0];
    size_t input_dim = input->shape[1];
    size_t hidden_dim = weights1->shape[1];
    size_t output_dim = weights2->shape[1];
    
    // 验证维度
    if (weights1->shape[0] != input_dim || weights2->shape[0] != hidden_dim ||
        bias1->shape[0] != hidden_dim || bias2->shape[0] != output_dim ||
        output->shape[0] != seq_len || output->shape[1] != output_dim) {
        return -1;
    }
    
    // 使用1KB缓冲区
    #define FFN_BLOCK_SIZE 32
    float input_buffer[FFN_BLOCK_SIZE];
    float weight_buffer[FFN_BLOCK_SIZE];
    float hidden_buffer[FFN_BLOCK_SIZE];
    float output_buffer[FFN_BLOCK_SIZE];
    float bias_buffer[FFN_BLOCK_SIZE];
    
    // 分块计算
    for (size_t i = 0; i < seq_len; i++) {
        // 第一层变换
        for (size_t j = 0; j < hidden_dim; j += FFN_BLOCK_SIZE) {
            size_t block_size = (j + FFN_BLOCK_SIZE > hidden_dim) ?
                               (hidden_dim - j) : FFN_BLOCK_SIZE;
            
            // 加载偏置
            if (read_weight_data(bias_buffer,
                j * sizeof(float),
                block_size * sizeof(float)) != 0) {
                return -1;
            }
            
            // 初始化隐藏状态为偏置
            memcpy(hidden_buffer, bias_buffer, block_size * sizeof(float));
            
            // 计算隐藏状态
            for (size_t k = 0; k < input_dim; k += FFN_BLOCK_SIZE) {
                size_t input_block_size = (k + FFN_BLOCK_SIZE > input_dim) ?
                                        (input_dim - k) : FFN_BLOCK_SIZE;
                
                // 加载输入
                if (read_weight_data(input_buffer,
                    (i * input_dim + k) * sizeof(float),
                    input_block_size * sizeof(float)) != 0) {
                    return -1;
                }
                
                // 加载权重
                if (read_weight_data(weight_buffer,
                    (k * hidden_dim + j) * sizeof(float),
                    block_size * sizeof(float)) != 0) {
                    return -1;
                }
                
                // 计算矩阵乘法
                for (size_t b = 0; b < input_block_size; b++) {
                    for (size_t h = 0; h < block_size; h++) {
                        hidden_buffer[h] += input_buffer[b] * weight_buffer[b * block_size + h];
                    }
                }
            }
            
            // 应用GELU激活函数
            for (size_t h = 0; h < block_size; h++) {
                hidden_buffer[h] = gelu(hidden_buffer[h]);
            }
            
            // 写回隐藏状态
            if (write_weight_data(hidden_buffer,
                (i * hidden_dim + j) * sizeof(float),
                block_size * sizeof(float)) != 0) {
                return -1;
            }
        }
        
        // 第二层变换
        for (size_t j = 0; j < output_dim; j += FFN_BLOCK_SIZE) {
            size_t block_size = (j + FFN_BLOCK_SIZE > output_dim) ?
                               (output_dim - j) : FFN_BLOCK_SIZE;
            
            // 加载偏置
            if (read_weight_data(bias_buffer,
                j * sizeof(float),
                block_size * sizeof(float)) != 0) {
                return -1;
            }
            
            // 初始化输出为偏置
            memcpy(output_buffer, bias_buffer, block_size * sizeof(float));
            
            // 计算输出
            for (size_t k = 0; k < hidden_dim; k += FFN_BLOCK_SIZE) {
                size_t hidden_block_size = (k + FFN_BLOCK_SIZE > hidden_dim) ?
                                         (hidden_dim - k) : FFN_BLOCK_SIZE;
                
                // 加载隐藏状态
                if (read_weight_data(hidden_buffer,
                    (i * hidden_dim + k) * sizeof(float),
                    hidden_block_size * sizeof(float)) != 0) {
                    return -1;
                }
                
                // 加载权重
                if (read_weight_data(weight_buffer,
                    (k * output_dim + j) * sizeof(float),
                    block_size * sizeof(float)) != 0) {
                    return -1;
                }
                
                // 计算矩阵乘法
                for (size_t h = 0; h < hidden_block_size; h++) {
                    for (size_t o = 0; o < block_size; o++) {
                        output_buffer[o] += hidden_buffer[h] * weight_buffer[h * block_size + o];
                    }
                }
            }
            
            // 写回输出
            if (write_weight_data(output_buffer,
                (i * output_dim + j) * sizeof(float),
                block_size * sizeof(float)) != 0) {
                return -1;
            }
        }
    }
    
    return 0;
}

// 初始化推理环境
int llm_init(LLMConfig* config, MemoryManager* mem_manager) {
    if (!config || !mem_manager) return -1;
    
    // 初始化控制台编码
    init_console_encoding();
    
    // 初始化权重缓存系统
    init_weight_cache();
    
    print_utf8("正在初始化模型...\n");
    
    // 初始化模型状态
    g_model_state = (LLMState*)calloc(1, sizeof(LLMState));
    if (!g_model_state) return -1;
    
    // 复制配置
    memcpy(&g_model_state->config, config, sizeof(LLMConfig));
    
    // 初始化激活值
    g_model_state->activations.data = NULL;
    g_model_state->activations.ndim = 2;
    g_model_state->activations.shape[0] = 1;
    g_model_state->activations.shape[1] = config->hidden_size;
    g_model_state->activations.quant_type = QUANT_NONE;
    
    g_model_state->is_initialized = 1;
    return 0;
}

// 主推理函数
int llm_forward(const int* input_tokens, size_t input_length, float* output) {
    if (!g_model_state || !input_tokens || !output) {
        print_utf8("错误：模型未初始化或参数无效\n");
        return -1;
    }
    
    print_utf8("开始推理计算...\n");
    
    // 分配临时缓冲区
    float* temp_buffer = malloc(g_model_state->config.hidden_size * sizeof(float));
    if (!temp_buffer) {
        print_utf8("错误：内存分配失败\n");
        return -1;
    }
    
    // 分配激活值缓冲区（如果未初始化）
    if (!g_model_state->activations.data) {
        g_model_state->activations.data = malloc(g_model_state->config.hidden_size * sizeof(float));
        if (!g_model_state->activations.data) {
            free(temp_buffer);
            return -1;
        }
    }
    
    for (size_t pos = 0; pos < input_length; pos++) {
        // 更新当前位置
        g_model_state->current_position = pos;
        
        // 1. 词嵌入层
        size_t token_offset = (size_t)input_tokens[pos] * g_model_state->config.hidden_size;
        if (read_weight_data(g_model_state->activations.data,
                           token_offset * sizeof(float),
                           g_model_state->config.hidden_size * sizeof(float)) != 0) {
            free(temp_buffer);
            return -1;
        }
        
        // 2. Transformer层循环
        for (size_t layer = 0; layer < g_model_state->config.num_layers; layer++) {
            TransformerWeights* layer_weights = &g_model_state->weights[layer];
            
            // 2.1 自注意力层
            float* query = malloc(g_model_state->config.hidden_size * sizeof(float));
            float* key = malloc(g_model_state->config.hidden_size * sizeof(float));
            float* value = malloc(g_model_state->config.hidden_size * sizeof(float));
            
            if (!query || !key || !value) {
                if (query) free(query);
                if (key) free(key);
                if (value) free(value);
                free(temp_buffer);
                return -1;
            }
            
            // QKV投影
            matrix_multiply(g_model_state->activations.data,
                          layer_weights->query_weight.data,
                          query,
                          1,
                          g_model_state->config.hidden_size,
                          g_model_state->config.hidden_size);
                          
            matrix_multiply(g_model_state->activations.data,
                          layer_weights->key_weight.data,
                          key,
                          1,
                          g_model_state->config.hidden_size,
                          g_model_state->config.hidden_size);
                          
            matrix_multiply(g_model_state->activations.data,
                          layer_weights->value_weight.data,
                          value,
                          1,
                          g_model_state->config.hidden_size,
                          g_model_state->config.hidden_size);
            
            // 更新KV缓存
            if (g_model_state->cache) {
                size_t cache_offset = pos * g_model_state->config.hidden_size;
                float* key_cache_ptr = g_model_state->cache->key_cache + cache_offset;
                float* value_cache_ptr = g_model_state->cache->value_cache + cache_offset;
                memcpy(key_cache_ptr, key, g_model_state->config.hidden_size * sizeof(float));
                memcpy(value_cache_ptr, value, g_model_state->config.hidden_size * sizeof(float));
            }
            
            // 注意力计算
            float scale = 1.0f / sqrtf((float)g_model_state->config.hidden_size);
            size_t head_dim = g_model_state->config.hidden_size / g_model_state->config.num_heads;
            
            if (g_model_state->cache) {
                attention_forward(query,
                               g_model_state->cache->key_cache,
                               g_model_state->cache->value_cache,
                               temp_buffer,
                               pos + 1,
                               head_dim,
                               scale);
            } else {
                attention_forward(query, key, value, temp_buffer,
                               1,
                               head_dim,
                               scale);
            }
            
            // 2.2 前馈网络
            ffn_forward(temp_buffer,
                       layer_weights->ffn_weight1.data,
                       layer_weights->ffn_weight2.data,
                       layer_weights->ffn_bias1.data,
                       layer_weights->ffn_bias2.data,
                       g_model_state->activations.data,
                       g_model_state->config.hidden_size,
                       g_model_state->config.ffn_hidden_size);
            
            // 清理临时内存
            free(query);
            free(key);
            free(value);
        }
        
        // 3. 输出层
        size_t output_offset = pos * g_model_state->config.hidden_size;
        memcpy(&output[output_offset],
               g_model_state->activations.data,
               g_model_state->config.hidden_size * sizeof(float));
    }
    
    free(temp_buffer);
    return 0;
}

// 清理函数
void llm_cleanup(void) {
    if (!g_model_state) return;
    
    print_utf8("正在清理资源...\n");
    
    // 清理权重缓存
    for (size_t i = 0; i < g_weight_cache.active_pages; i++) {
        if (g_weight_cache.pages[i].is_dirty) {
            write_page_to_swap(&g_weight_cache.pages[i]);
        }
        if (g_weight_cache.pages[i].data) {
            free(g_weight_cache.pages[i].data);
        }
    }
    
    // 删除所有交换文件
    WIN32_FIND_DATA fd;
    HANDLE hFind = FindFirstFile(CACHE_DIR "/*.*", &fd);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            char path[512];
            snprintf(path, sizeof(path), "%s/%s", CACHE_DIR, fd.cFileName);
            DeleteFile(path);
        } while (FindNextFile(hFind, &fd));
        FindClose(hFind);
    }
    _rmdir(CACHE_DIR);
    
    // 清理模型状态
    if (g_model_state->activations.data) {
        free(g_model_state->activations.data);
    }
    
    free(g_model_state);
    g_model_state = NULL;
    
    print_utf8("清理完成\n");
} 