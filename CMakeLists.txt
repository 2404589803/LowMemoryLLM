cmake_minimum_required(VERSION 3.10)
project(LowMemoryLLM C ASM)

# 设置C标准
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# 检测操作系统和架构
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(OS_LINUX 1)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(OS_WINDOWS 1)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(OS_MACOS 1)
endif()

# 检测CPU架构
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(ARCH_X86_64 1)
    set(ASM_SOURCE src/hal/x86_64/cpu_impl.asm)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
    set(ARCH_ARM64 1)
    set(ASM_SOURCE src/hal/arm/cpu_impl.asm)
endif()

# 添加编译选项
add_compile_options(-Wall -Wextra)

# 添加源文件
set(SOURCES
    src/hal/hal.c
    src/hal/device_manager.c
    ${ASM_SOURCE}
)

# 创建库目标
add_library(lowmemory_llm STATIC ${SOURCES})

# 设置包含目录
target_include_directories(lowmemory_llm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hal
)

# 根据平台设置特定编译选项
if(OS_LINUX)
    target_compile_definitions(lowmemory_llm PUBLIC OS_LINUX)
elseif(OS_WINDOWS)
    target_compile_definitions(lowmemory_llm PUBLIC OS_WINDOWS)
elseif(OS_MACOS)
    target_compile_definitions(lowmemory_llm PUBLIC OS_MACOS)
endif()

# 添加测试目标
enable_testing()
add_subdirectory(tests)

# 安装规则
install(TARGETS lowmemory_llm
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES
    src/hal/hal.h
    src/hal/device_manager.h
    DESTINATION include/lowmemory_llm
) 