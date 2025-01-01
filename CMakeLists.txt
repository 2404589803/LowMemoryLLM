cmake_minimum_required(VERSION 3.10)
project(LowMemoryLLM C)

# 查找CURL包
find_package(CURL REQUIRED)

# 包含目录
include_directories(${CURL_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

# 添加源文件
add_executable(test_download 
    tests/test_download.c
    src/download.c
)

# 添加推理测试可执行文件
add_executable(test_inference
    tests/test_inference.c
    src/inference.c
    src/download.c
)

# 添加权重生成工具
add_executable(generate_weights
    tools/generate_weights.c
)

# 链接CURL库
target_link_libraries(test_download ${CURL_LIBRARIES})
target_link_libraries(test_inference ${CURL_LIBRARIES})

# 添加运行时依赖
if(WIN32)
    file(COPY 
        "C:/msys64/mingw64/bin/libcurl-4.dll"
        DESTINATION ${CMAKE_BINARY_DIR}
    )
endif() 