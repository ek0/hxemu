cmake_minimum_required(VERSION 3.24)
include(FetchContent)

# Required for static windows runtime
cmake_policy(SET CMP0091 NEW)

project(hxsym CXX)

# Enforce C++ standard
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD 20)

if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
	# Enforce static windows runtime
	set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
	message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo")
	#
	# Fetch triton lib
	#
	FetchContent_Declare(
	  triton
	  URL https://github.com/ek0/triton-build/releases/download/1.0/triton-win-x64-clang-static-relwithdebinfo.zip
	)
	FetchContent_MakeAvailable(triton)
else()
	message(FATAL_ERROR "Unsupported configuration: ${CMAKE_BUILD_TYPE}")
endif()

#
# IDA SDK location.
#
message(STATUS "IDA SDK location: ${IDA_SDK}")
if(NOT DEFINED IDA_SDK)
	message(FATAL_ERROR "Please specify the path to the IDA SDK")
endif()

add_library(hxemu
	"${CMAKE_CURRENT_LIST_DIR}/src/hxemu.h"
	"${CMAKE_CURRENT_LIST_DIR}/src/hxemu.cc")

target_include_directories(hxemu PRIVATE "${IDA_SDK}\\include")
target_link_libraries(hxemu "${IDA_SDK}\\lib\\x64_win_vc_64\\ida.lib")
set_target_properties(hxemu PROPERTIES PUBLIC_HEADER "src/hxemu.h")
target_link_libraries(hxemu "${triton_SOURCE_DIR}/lib/triton_all.lib")
target_include_directories(hxemu PRIVATE "${triton_SOURCE_DIR}/include")