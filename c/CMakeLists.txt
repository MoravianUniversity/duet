cmake_minimum_required(VERSION 3.16.0)
project(DUET)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add ESP-IDF dummy includes directory
include_directories("esp-idf-dummy")

# Dummy idf_component_register function
function(idf_component_register)
    # Do nothing
endfunction()

# Add ESP-DSP
# Note: need to make some changes to esp-dsp:
# 1. dsps_cplx_gen_* file renaming to not conflict with other files (in CMakeLists.txt and the file itself)
# 2. Adding the following near the end of modules/common/include/dsp_common.h:
#     #elif defined(__clang__)
#     #define dsp_get_cpu_cycle_count  __builtin_readcyclecounter
include(esp-dsp/CMakeLists.txt)
foreach(pth IN LISTS include_dirs)
    cmake_path(ABSOLUTE_PATH pth BASE_DIRECTORY "esp-dsp")
    include_directories(${pth})
endforeach()
foreach(pth IN LISTS srcs)
    cmake_path(ABSOLUTE_PATH pth BASE_DIRECTORY "esp-dsp")
    list(APPEND dsp_srcs ${pth})
endforeach()

set(CMAKE_OSX_DEPLOYMENT_TARGET "10.9" CACHE STRING "Minimum OS X deployment version" FORCE)
set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "Build architectures for Mac OS X" FORCE)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -ffast-math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math")
add_compile_definitions(DUET_OFFLINE_PREPROCESS)
add_library(DUET SHARED duet.cpp fast_math.cpp largest_k.cpp ${dsp_srcs})
set_target_properties(DUET PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(DUET PROPERTIES PUBLIC_HEADER duet.h)
# set_target_properties(DUET PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH};${CMAKE_INSTALL_PREFIX}/lib;${CMAKE_SOURCE_DIR}")
# set_target_properties(DUET PROPERTIES BUILD_RPATH "${CMAKE_SOURCE_DIR}")
