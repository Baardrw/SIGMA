cmake_minimum_required(VERSION 3.18)

# Set project name and version
project(SIGMA VERSION 1.0.0 LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable CUDA separable compilation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Find required packages
find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

set(CMAKE_CUDA_ARCHITECTURES 90)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Source files
set(SIGMA_SOURCES
    src/main.cpp
    src/muon_segment.cu
)

# Header files
set(SIGMA_HEADERS
    include/muon_segment.cuh
)

# Create main executable
add_executable(sigma ${SIGMA_SOURCES} ${SIGMA_HEADERS})

# Set properties for CUDA compilation
set_target_properties(sigma PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_RUNTIME_LIBRARY SHARED
)

# Link libraries
target_link_libraries(sigma
    ${CUDA_LIBRARIES}
    Threads::Threads
    # ${ROOT_LIBRARIES}
    # ${Boost_LIBRARIES}
)

# Optional: Create a library for reusable components
add_library(sigma_lib STATIC
    src/muon_segment.cpp
    src/gpu_kernels.cu
    src/data_manager.cpp
)

set_target_properties(sigma_lib PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_RUNTIME_LIBRARY SHARED
)

target_link_libraries(sigma_lib
    ${CUDA_LIBRARIES}
)

# Optional: Build tests
option(BUILD_TESTS "Build test programs" ON)
if(BUILD_TESTS)
    enable_testing()
    
    # Find Google Test (optional)
    # find_package(GTest REQUIRED)
    
    add_executable(sigma_tests
        tests/test_main.cpp
        tests/test_muon_segment.cpp
        tests/test_gpu_kernels.cu
    )
    
    target_link_libraries(sigma_tests
        sigma_lib
        # ${GTEST_LIBRARIES}
        Threads::Threads
    )
    
    # Add test
    add_test(NAME SigmaTests COMMAND sigma_tests)
endif()

# Optional: Build benchmarks
option(BUILD_BENCHMARKS "Build benchmark programs" OFF)
if(BUILD_BENCHMARKS)
    add_executable(sigma_benchmark
        benchmarks/benchmark_main.cpp
        benchmarks/segment_formation_benchmark.cu
    )
    
    target_link_libraries(sigma_benchmark
        sigma_lib
    )
endif()

# Install targets
install(TARGETS sigma sigma_lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES ${SIGMA_HEADERS}
    DESTINATION include/sigma
)

# Print configuration summary
message(STATUS "SIGMA Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA Standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build Tests: ${BUILD_TESTS}")
message(STATUS "  Build Benchmarks: ${BUILD_BENCHMARKS}")