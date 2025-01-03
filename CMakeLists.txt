cmake_minimum_required(VERSION 3.10)
project(LowMemoryLLM C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# 添加依赖包
find_package(CURL REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(ZLIB REQUIRED)
find_package(cJSON REQUIRED)

# 添加子目录
add_subdirectory(src)
add_subdirectory(tests)

# 设置包含目录
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CURL_INCLUDE_DIRS}
    ${OPENSSL_INCLUDE_DIR}
    ${ZLIB_INCLUDE_DIRS}
    ${CJSON_INCLUDE_DIRS}
)

# 设置链接库
set(COMMON_LIBS
    ${CURL_LIBRARIES}
    ${OPENSSL_LIBRARIES}
    ${ZLIB_LIBRARIES}
    ${CJSON_LIBRARIES}
)

# 导出链接库变量给子目录使用
set(COMMON_LIBS ${COMMON_LIBS} PARENT_SCOPE) 