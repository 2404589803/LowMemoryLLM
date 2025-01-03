# 源文件列表
set(LIB_SOURCES
    download.c
    extreme_compress.c
    extreme_kv_cache.c
    tensor.c
    inference.c
)

# 创建库
add_library(low_memory_llm STATIC ${LIB_SOURCES})

# 设置包含目录
target_include_directories(low_memory_llm
    PUBLIC ${CMAKE_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

# 链接依赖库
target_link_libraries(low_memory_llm
    PRIVATE ${CURL_LIBRARIES}
    PRIVATE ${OPENSSL_LIBRARIES}
    PRIVATE ${ZLIB_LIBRARIES}
    PRIVATE ${CJSON_LIBRARIES}
)

# 安装配置
install(TARGETS low_memory_llm
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
) 