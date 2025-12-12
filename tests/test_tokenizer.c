#include "../src/huggingface/tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    printf("Running tokenizer tests...\n");
    
    // 创建测试配置
    TokenizerConfig config = {
        .type = TOKENIZER_WORDPIECE,
        .vocab_file = "tests/test_vocab.txt",
        .add_special_tokens = 1,
        .add_prefix_space = 0,
        .unk_token = "[UNK]",
        .pad_token = "[PAD]",
        .bos_token = "[BOS]",
        .eos_token = "[EOS]"
    };
    
    // 创建测试词汇表
    FILE* fp = fopen(config.vocab_file, "w");
    if (!fp) {
        printf("Failed to create test vocab file\n");
        return 1;
    }
    
    // 写入一些测试词汇
    fprintf(fp, "[PAD]\n");
    fprintf(fp, "[UNK]\n");
    fprintf(fp, "[BOS]\n");
    fprintf(fp, "[EOS]\n");
    fprintf(fp, "hello\n");
    fprintf(fp, "world\n");
    fprintf(fp, "test\n");
    fprintf(fp, "token\n");
    fprintf(fp, "##izer\n");
    fclose(fp);
    
    // 初始化分词器
    Tokenizer* tokenizer = NULL;
    if (tokenizer_init(&tokenizer, &config) != 0) {
        printf("Failed to initialize tokenizer\n");
        return 1;
    }
    
    // 测试编码
    const char* test_text = "hello world test";
    int32_t tokens[10];
    size_t num_tokens = 0;
    
    if (tokenizer_encode(tokenizer, test_text, tokens, &num_tokens, 10) != 0) {
        printf("Failed to encode text\n");
        tokenizer_free(tokenizer);
        return 1;
    }
    
    printf("Encoded %zu tokens:\n", num_tokens);
    for (size_t i = 0; i < num_tokens; i++) {
        const char* token_text = tokenizer_id_to_token(tokenizer, tokens[i]);
        printf("%zu: %d (%s)\n", i, tokens[i], token_text ? token_text : "NULL");
    }
    
    // 测试解码
    char decoded_text[100];
    size_t text_length = 0;
    
    if (tokenizer_decode(tokenizer, tokens, num_tokens, decoded_text, &text_length) != 0) {
        printf("Failed to decode tokens\n");
        tokenizer_free(tokenizer);
        return 1;
    }
    
    printf("Decoded text: %s\n", decoded_text);
    
    // 清理
    tokenizer_free(tokenizer);
    remove(config.vocab_file);
    
    printf("Tokenizer tests completed\n");
    return 0;
} 