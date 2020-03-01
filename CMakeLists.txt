cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(SimpleConcurrentGPUHashTable LANGUAGES CXX CUDA)

# put predefined cmake projects in their own solution folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_35,code=sm_35")

set(SOURCE_FILES src/test.cpp src/linearprobing.h src/linearprobing.cu)

include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(test ${SOURCE_FILES})

# visual studio project should mimic directory structure
# this isn't working for me; I think because
# https://developercommunity.visualstudio.com/content/problem/777578/source-grouptree-no-longer-works.html
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_FILES})
