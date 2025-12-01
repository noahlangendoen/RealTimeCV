cmake_minimum_required(VERSION 3.15)
project(RealTimeCVPipeline VERSION 1.0.0 LANGUAGES CXX)

if(MSVC)
    set(CMAKE_GENERATOR_PLATFORM x64)
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler warnings
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Set OpenCV path directly
set(OpenCV_DIR "D:/opencv/build/x64/vc16/lib")

# Find OpenCV
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found at ${OpenCV_DIR}")
endif()

# ONNX Runtime configuration
set(ONNXRUNTIME_ROOT_DIR "D:/onnxruntime" CACHE PATH "ONNX Runtime root directory")

find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS "${ONNXRUNTIME_ROOT_DIR}/include"
    NO_DEFAULT_PATH
)

find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    PATHS "${ONNXRUNTIME_ROOT_DIR}/lib"
    NO_DEFAULT_PATH
)

if(ONNXRUNTIME_LIB AND ONNXRUNTIME_INCLUDE_DIR)
    message(STATUS "ONNX Runtime found!")
    message(STATUS "  Include: ${ONNXRUNTIME_INCLUDE_DIR}")
    message(STATUS "  Library: ${ONNXRUNTIME_LIB}")
    set(HAS_ONNXRUNTIME TRUE)
else()
    message(WARNING "ONNX Runtime not found at ${ONNXRUNTIME_ROOT_DIR}")
    message(WARNING "Will build without ONNX support")
    set(HAS_ONNXRUNTIME FALSE)
endif()

# Include project directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})

# Source files
set(SOURCES
    src/main.cpp
    src/core/frame.cpp
)

# Create executable
add_executable(cv_pipeline ${SOURCES})

# Link libraries
target_link_libraries(cv_pipeline ${OpenCV_LIBS})

if(HAS_ONNXRUNTIME)
    # Add ONNX Runtime include directory to the target
    target_include_directories(cv_pipeline PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(cv_pipeline ${ONNXRUNTIME_LIB})
    target_compile_definitions(cv_pipeline PRIVATE HAS_ONNXRUNTIME)
    
    # Copy DLL for runtime
    set(ONNXRUNTIME_DLL "${ONNXRUNTIME_ROOT_DIR}/lib/onnxruntime.dll")
    if(EXISTS "${ONNXRUNTIME_DLL}")
        add_custom_command(TARGET cv_pipeline POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_DLL}"
            $<TARGET_FILE_DIR:cv_pipeline>
            COMMENT "Copying onnxruntime.dll"
        )
    endif()
endif()