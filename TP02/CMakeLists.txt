cmake_minimum_required (VERSION 3.18)
project(ISIMA_TP_2 LANGUAGES CXX CUDA)

set(CUDA_ARCHITECTURES 60)

include("FetchContent")
FetchContent_Declare(
	stb
	GIT_REPOSITORY https://github.com/nothings/stb.git
)
FetchContent_GetProperties(stb)
if(NOT stb)
	FetchContent_Populate(stb)
	#include the headerfiles and libs
endif()

add_executable(ISIMA_TP_2 src/main.cpp)
add_library(gpgpu include/gpgpu.h src/gpgpu/gpgpu.cu)

target_include_directories(ISIMA_TP_2 PRIVATE ${stb_SOURCE_DIR} include)
target_include_directories(gpgpu PRIVATE include)

target_link_libraries(ISIMA_TP_2 PUBLIC gpgpu)
