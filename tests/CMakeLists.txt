# 测试目录的CMake配置
cmake_minimum_required(VERSION 3.10)

# 添加测试可执行文件
add_executable(test_download test_download.c)

# 链接主库和依赖
target_link_libraries(test_download
    PRIVATE low_memory_llm
    PRIVATE ${CURL_LIBRARIES}
    PRIVATE ${OPENSSL_LIBRARIES}
    PRIVATE ${ZLIB_LIBRARIES}
    PRIVATE ${CJSON_LIBRARIES}
)

# 设置包含目录
target_include_directories(test_download
    PRIVATE ${CMAKE_SOURCE_DIR}/include
    PRIVATE ${CMAKE_SOURCE_DIR}/src
)

# 设置输出目录
set_target_properties(test_download
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
) 