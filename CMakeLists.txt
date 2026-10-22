cmake_minimum_required(VERSION 3.20)

# Host compiler: when the default g++ is newer than nvcc supports, prefer
# g++-15; otherwise fall back to the override flag with -std=c++20 (newer
# libstdc++ headers do not parse under nvcc's c++17 mode). Runs before
# project(), where CMAKE_CXX_COMPILER* are not yet known.
find_program(_cxx_compiler NAMES $ENV{CXX} c++ g++)
if(_cxx_compiler)
    execute_process(COMMAND ${_cxx_compiler} -dumpversion
                    OUTPUT_VARIABLE _gcc_version OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
endif()
if(_gcc_version VERSION_GREATER_EQUAL 16)
    find_program(_gcc15 NAMES g++-15)
    if(_gcc15)
        set(CMAKE_CUDA_HOST_COMPILER "${_gcc15}")
    else()
        set(CMAKE_CUDA_FLAGS_INIT "-allow-unsupported-compiler -std=c++20")
    endif()
endif()

project(gpu-bundle-adjustment LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
if(CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_STANDARD 17)
else()
    set(CMAKE_CUDA_STANDARD 20)
endif()
set(CMAKE_CUDA_ARCHITECTURES native)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED)

add_library(gba INTERFACE)
target_include_directories(gba INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(gba INTERFACE
    CUDA::cudart CUDA::cublas Eigen3::Eigen
)

add_library(gba_bal STATIC src/bal_reader.cpp)
target_link_libraries(gba_bal PUBLIC gba)

add_executable(bench_ceres benchmarks/bench_ceres.cpp)
target_link_libraries(bench_ceres PRIVATE gba_bal Ceres::ceres)

add_executable(quantize_bal tools/quantize_bal.cu)
target_link_libraries(quantize_bal PRIVATE gba_bal)

add_executable(fp8_residual_probe tools/fp8_residual_probe.cu)
target_link_libraries(fp8_residual_probe PRIVATE gba_bal)

add_library(gba_lm STATIC src/lm.cu)
target_link_libraries(gba_lm PUBLIC gba_bal)

add_executable(bench_bal benchmarks/bench_bal.cpp)
target_link_libraries(bench_bal PRIVATE gba_lm)

add_executable(test_schur tests/test_schur.cu)
target_link_libraries(test_schur PRIVATE gba)

add_executable(test_vs_ceres tests/test_vs_ceres.cpp)
target_link_libraries(test_vs_ceres PRIVATE gba_lm)

enable_testing()
add_test(NAME schur COMMAND test_schur)
add_test(NAME vs_ceres COMMAND test_vs_ceres
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
