cmake_minimum_required(VERSION 3.15)
project(FiberAnalyzer VERSION 1.0.0 LANGUAGES CXX)

# C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置默认构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 打印构建信息
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Project directory: ${CMAKE_SOURCE_DIR}")

# 查找Python（必须）
find_package(Python 3.11 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Found Python: ${Python_VERSION}")
message(STATUS "Python include: ${Python_INCLUDE_DIRS}")

# 添加pybind11（使用项目内置的版本）
add_subdirectory(src/cpp/extern/pybind11)
message(STATUS "Using bundled pybind11")

# 指定输出目录为根目录的lib文件夹
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

# 对于多配置生成器（如Visual Studio），确保所有配置都输出到同一目录
foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${CONFIG} CONFIG_UPPER)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_SOURCE_DIR}/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_SOURCE_DIR}/lib)
endforeach()

# ============================================
# 创建Python扩展模块
# ============================================

# 源文件列表（注意：从根目录开始的相对路径）
set(SOURCES
    ${CMAKE_SOURCE_DIR}/src/cpp/measure_tool.cpp
    # 如果有更多cpp文件，在这里添加
)

# 验证源文件是否存在
foreach(SOURCE_FILE ${SOURCES})
    if(NOT EXISTS ${SOURCE_FILE})
        message(FATAL_ERROR "Source file not found: ${SOURCE_FILE}")
    else()
        message(STATUS "Found source file: ${SOURCE_FILE}")
    endif()
endforeach()

# 创建Python模块
pybind11_add_module(measure_tool ${SOURCES})

# 如果有头文件目录，添加include路径
# target_include_directories(measure_tool PRIVATE src/cpp/include)

# ============================================
# 编译优化选项
# ============================================

if(MSVC)
    # Windows: Visual Studio / MSVC
    target_compile_options(measure_tool PRIVATE
        /O2                 # 优化
        /W3                 # 警告级别
        /utf-8              # UTF-8源文件编码
        /bigobj             # 支持大对象文件
    )
    # 禁用一些不必要的警告
    target_compile_definitions(measure_tool PRIVATE
        _CRT_SECURE_NO_WARNINGS
        _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
    )
else()
    # Linux / macOS: GCC / Clang
    target_compile_options(measure_tool PRIVATE
        -O3                 # 最高优化
        -Wall               # 所有警告
        -Wextra             # 额外警告
        -march=native       # 针对当前CPU优化
    )
endif()

# ============================================
# 打印最终信息
# ============================================

message(STATUS "")
message(STATUS "===========================================")
message(STATUS "Configuration Summary:")
message(STATUS "  Module name: measure_tool")  
message(STATUS "  Output directory: ${CMAKE_SOURCE_DIR}/lib")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Python: ${Python_VERSION}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "===========================================")
message(STATUS "")