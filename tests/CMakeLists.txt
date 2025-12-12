# 测试目录的CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

# 添加基本测试执行目标
add_executable(test_tokenizer test_tokenizer.c)
target_link_libraries(test_tokenizer lowmemory_llm)

add_executable(test_model_loader test_model_loader.c)
target_link_libraries(test_model_loader lowmemory_llm)

add_executable(test_training test_training.c)
target_link_libraries(test_training lowmemory_llm)

# 添加测试
add_test(NAME TokenizerTest COMMAND test_tokenizer)
add_test(NAME ModelLoaderTest COMMAND test_model_loader)
add_test(NAME TrainingTest COMMAND test_training) 