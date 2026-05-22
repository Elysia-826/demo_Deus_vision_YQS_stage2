cmake_minimum_required(VERSION 3.15)
project(armor_detect VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================
# 开关选项
# ============================================================
option(ENABLE_JUDGE "Enable judge logging module" OFF)
option(ENABLE_EVALUATE "Enable evaluate module (requires ENABLE_JUDGE)" OFF)

# ============================================================
# 依赖项查找与配置
# ============================================================
# 1. 查找 OpenCV
find_package(OpenCV 4.5 REQUIRED)

# 2. 设置 ONNX Runtime 路径
set(ONNXRUNTIME_DIR "C:\\onnxruntime-win-x64-gpu-1.19.0")

# 3. 统一包含头文件目录
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_DIR}/include
)

# 4. 统一配置库搜索目录
link_directories(
    ${ONNXRUNTIME_DIR}/lib
)

# ============================================================
# 源文件收集
# ============================================================
set(SOURCES
    src/main.cpp
    src/detector.cpp
    src/target_selector.cpp
    src/visualizer.cpp
)

# 根据开关动态追加裁判与评估模块
if(ENABLE_JUDGE)
    list(APPEND SOURCES src/judge.cpp)
    add_definitions(-DENABLE_JUDGE)
    message(STATUS "Judge module: ENABLED")
else()
    message(STATUS "Judge module: DISABLED")
endif()

if(ENABLE_EVALUATE)
    list(APPEND SOURCES src/evaluate.cpp)
    add_definitions(-DENABLE_EVALUATE)
    message(STATUS "Evaluate module: ENABLED")
else()
    message(STATUS "Evaluate module: DISABLED")
endif()

# ============================================================
# 创建可执行目标 (全文件仅保留这一次 add_executable)
# ============================================================
add_executable(${PROJECT_NAME} ${SOURCES})

# ============================================================
# 编译器选项配置
# ============================================================
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /utf-8)
endif()

# ============================================================
# 链接库配置
# ============================================================
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    ${OpenCV_LIBS}
    "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib"
)

# ============================================================
# POST_BUILD 自动化黑科技：自动复制 ONNX Runtime 的 GPU 动态库到 exe 目录
# ============================================================
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
    $<TARGET_FILE_DIR:${PROJECT_NAME}>
)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${ONNXRUNTIME_DIR}/lib/onnxruntime_providers_cuda.dll"
    $<TARGET_FILE_DIR:${PROJECT_NAME}>
)

# ============================================================
# 状态打印
# ============================================================
message(STATUS "OpenCV version: ${OpenCV_VERSION}")