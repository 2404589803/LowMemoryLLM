#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

// 生成随机权重
static float random_weight(void) {
    return (float)rand() / RAND_MAX * 2.0f - 1.0f;
}

int main() {
    const char* output_file = "../build/model/pytorch_model.bin";
    const uint32_t magic = 0x4D4C4C4D;  // "MLLM"
    const size_t num_layers = 32;        // MiniCPM3-4B的层数
    const size_t hidden_size = 4096;     // 隐藏层大小
    const size_t num_weights_per_layer = 12;  // 每层的权重矩阵数量
    
    // 创建模型目录
    system("mkdir -p ../build/model");
    
    FILE* fp = fopen(output_file, "wb");
    if (!fp) {
        printf("无法创建输出文件\n");
        return 1;
    }
    
    // 写入魔数
    fwrite(&magic, sizeof(uint32_t), 1, fp);
    
    // 为每一层生成权重
    for (size_t layer = 0; layer < num_layers; layer++) {
        for (size_t w = 0; w < num_weights_per_layer; w++) {
            // 确定矩阵维度
            uint32_t ndim = 2;
            uint32_t dim1, dim2;
            
            switch (w) {
                case 0:  // Query权重
                case 1:  // Key权重
                case 2:  // Value权重
                    dim1 = hidden_size;
                    dim2 = hidden_size;
                    break;
                case 3:  // Query偏置
                case 4:  // Key偏置
                case 5:  // Value偏置
                    dim1 = 1;
                    dim2 = hidden_size;
                    break;
                case 6:  // 输出投影权重
                    dim1 = hidden_size;
                    dim2 = hidden_size;
                    break;
                case 7:  // 输出投影偏置
                    dim1 = 1;
                    dim2 = hidden_size;
                    break;
                case 8:  // FFN第一层权重
                    dim1 = hidden_size;
                    dim2 = hidden_size * 4;
                    break;
                case 9:  // FFN第一层偏置
                    dim1 = 1;
                    dim2 = hidden_size * 4;
                    break;
                case 10: // FFN第二层权重
                    dim1 = hidden_size * 4;
                    dim2 = hidden_size;
                    break;
                case 11: // FFN第二层偏置
                    dim1 = 1;
                    dim2 = hidden_size;
                    break;
            }
            
            // 写入维度信息
            fwrite(&ndim, sizeof(uint32_t), 1, fp);
            fwrite(&dim1, sizeof(uint32_t), 1, fp);
            fwrite(&dim2, sizeof(uint32_t), 1, fp);
            
            // 生成并写入随机权重
            size_t num_elements = dim1 * dim2;
            float* data = (float*)malloc(num_elements * sizeof(float));
            if (!data) {
                printf("内存分配失败\n");
                fclose(fp);
                return 1;
            }
            
            // 使用Xavier初始化
            float scale = sqrtf(6.0f / (dim1 + dim2));
            for (size_t i = 0; i < num_elements; i++) {
                data[i] = random_weight() * scale;
            }
            
            fwrite(data, sizeof(float), num_elements, fp);
            free(data);
            
            printf("已生成第 %zu 层的第 %zu 个权重矩阵 (%u x %u)\n", 
                   layer + 1, w + 1, dim1, dim2);
        }
    }
    
    fclose(fp);
    printf("权重文件生成完成：%s\n", output_file);
    
    return 0;
} 