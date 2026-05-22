cmake_minimum_required(VERSION 3.15)
project(armor_detect VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================
# 开关
# ============================================================
option(ENABLE_JUDGE "Enable judge logging module" OFF)
option(ENABLE_EVALUATE "Enable evaluate module (requires ENABLE_JUDGE)" OFF)

# ============================================================
# 依赖
# ============================================================
find_package(OpenCV 4.5 REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# ============================================================
# 源文件
# ============================================================
set(SOURCES
    src/main.cpp
    src/detector.cpp
    src/target_selector.cpp
    src/visualizer.cpp
)

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
# 构建
# ============================================================
# 强制 MSVC 使用 UTF-8 编译
if(MSVC)
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
endif()
add_executable(${PROJECT_NAME} ${SOURCES})

# 在 target_link_libraries 之前添加 ONNX Runtime 路径
set(ONNXRUNTIME_ROOT "C:\\onnxruntime-win-x64-gpu-1.19.0")
include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    onnxruntime
)

# 根据编译器选择合适选项
# 仅 MSVC 编译时启用 /utf-8 避免中文编码警告
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /utf-8)
endif()


message(STATUS "OpenCV version: ${OpenCV_VERSION}")
