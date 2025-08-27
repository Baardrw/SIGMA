cmake_minimum_required(VERSION 3.18)

# Set project name and version
project(SIGMA VERSION 1.0.0 LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Export compile commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable CUDA separable compilation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Find required packages
find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Set compiler flags
# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -fmad=false --expt-relaxed-constexpr -diag-suppress=20015 -diag-suppress=20012")

set(CMAKE_CUDA_ARCHITECTURES 90)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# ===== SIGMA LIBRARY =====
# Library source files (everything except main)
set(SIGMA_LIB_SOURCES
    src/lib/muon_segment.cu
    src/lib/residual_math.cu
    src/lib/line_math.cu
    src/lib/matrix_math.cu
)

# Library header files
set(SIGMA_LIB_HEADERS
    include/config.h
    include/muon_segment.h
    include/residual_math.h
    include/data_structures.h
    include/line_math.h
    include/matrix_math.h
)

# Create the reusable library
add_library(sigma_lib STATIC ${SIGMA_LIB_SOURCES} ${SIGMA_LIB_HEADERS})

# Set CUDA properties for library
set_target_properties(sigma_lib PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_RUNTIME_LIBRARY SHARED
)

# Library dependencies
target_link_libraries(sigma_lib
    ${CUDA_LIBRARIES}
    Eigen3::Eigen
)

# Make headers available to library consumers
target_include_directories(sigma_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# ===== MAIN EXECUTABLE =====
# Main executable source (only main.cpp)
set(SIGMA_MAIN_SOURCES
    src/main.cpp
)

# Create main executable
add_executable(sigma ${SIGMA_MAIN_SOURCES})

# Link the main executable against the library
target_link_libraries(sigma
    sigma_lib                 
    Threads::Threads
    Eigen3::Eigen
    ${CUDA_LIBRARIES}         # CUDA libraries
)


# ===== Common testing lib for testing utils =====

set(SIGMA_TESTING_LIB_SOURCES
    tests/common/test_data_utils.cu
    tests/common/test_math_utils.cu
)

# Library header files
set(SIGMA_TESING_LIB_HEADERS
    tests/common/test_data_utils.h
    tests/common/test_math_utils.h
)

add_library(sigma_testing_lib STATIC ${SIGMA_TESTING_LIB_SOURCES} ${SIGMA_LIB_HEADERS})

# Set CUDA properties for library
set_target_properties(sigma_testing_lib PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_RUNTIME_LIBRARY SHARED
)

# Library dependencies
target_link_libraries(sigma_testing_lib
    sigma_lib
    ${CUDA_LIBRARIES}
    Eigen3::Eigen
)

# Make headers available to library consumers
target_include_directories(sigma_testing_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/tests/common
)


# ===== TESTS =====

option(BUILD_TESTS "Build test programs" ON)
if(BUILD_TESTS)

    enable_testing()
    
    # GPU Kernels Test
    add_executable(sigma_gpu_tests
        tests/test_gpu_kernels.cu
    )
    
    target_include_directories(sigma_gpu_tests PRIVATE
        tests/                          
    )
    
    target_link_libraries(sigma_gpu_tests
        sigma_lib             
        sigma_testing_lib
        Threads::Threads
        Eigen3::Eigen
    )
    
    set_target_properties(sigma_gpu_tests PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    
    # Matrix Inversion Test
    add_executable(sigma_matrix_tests
        tests/test_matrix_inversion.cu
    )
    
    target_include_directories(sigma_matrix_tests PRIVATE
        tests/                          
    )
    
    target_link_libraries(sigma_matrix_tests
        sigma_lib             
        sigma_testing_lib
        Threads::Threads
        Eigen3::Eigen
    )
    
    set_target_properties(sigma_matrix_tests PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    
    # Add both tests to CTest
    add_test(NAME SigmaGPUTests COMMAND sigma_gpu_tests)
    add_test(NAME SigmaMatrixTests COMMAND sigma_matrix_tests)
    
endif()

# # ===== OPTIONAL BENCHMARKS =====
option(BUILD_BENCHMARKS "Build benchmark programs" ON)
if(BUILD_BENCHMARKS)
    add_executable(sigma_benchmark
        benchmarks/benchmark_main.cu
    )
    
    # Benchmarks also link against the library
    target_link_libraries(sigma_benchmark
        sigma_lib
        sigma_testing_lib
        Threads::Threads
        Eigen3::Eigen
    )
endif()

# ===== INSTALLATION =====
install(TARGETS sigma sigma_lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES ${SIGMA_LIB_HEADERS}
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