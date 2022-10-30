cmake_minimum_required(VERSION 3.24)
include(FetchContent)

# Required for static windows runtime
cmake_policy(SET CMP0091 NEW)

project(hxemu CXX)

# Enforce C++ standard
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD 20)

if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
	# Enforce static windows runtime
	set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
	message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo" OR ${CMAKE_BUILD_TYPE} STREQUAL "Release")
	#
	# Fetch triton lib
	#
	FetchContent_Declare(
	  triton
	  URL https://github.com/ek0/triton-build/releases/download/1.0/triton-win64-release-static_runtime.zip
	  URL_HASH SHA1=05CB3B4A8C1744EC845A523803D6318B49D68E86
	)
	FetchContent_MakeAvailable(triton)
elseif(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
	#
	# Fetch triton lib
	#
	FetchContent_Declare(
	  triton
	  URL https://github.com/ek0/triton-build/releases/download/1.0/triton-win64-debug-static_runtime.zip
	)
	FetchContent_MakeAvailable(triton)
else()
	message(FATAL_ERROR "Unsupported configuration: ${CMAKE_BUILD_TYPE}")
endif()

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

find_package(triton PATHS "${triton_SOURCE_DIR}" CONFIG REQUIRED)

add_library(hxemu STATIC
	"${CMAKE_CURRENT_LIST_DIR}/src/hxemu.cc")

target_include_directories(hxemu PUBLIC "${CMAKE_CURRENT_LIST_DIR}/include")

target_link_libraries(hxemu PUBLIC triton::triton)

enable_testing()
add_executable(hxemu_test
	"${CMAKE_CURRENT_LIST_DIR}/tests/main.cc"
)
target_link_libraries(
  hxemu_test
  hxemu
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(hxemu_test)
