#pragma once
#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <vector_types.h>


#define EPSILON 1e-6
#define FULL_MASK 0xffffffff
#define TILE_SIZE 16
#define OVERFLOW_TILE_SIZE                                                     \
  32 // Amount of threads needed to handle overflowing buckets

typedef double real_t;
typedef double3 real_t3;

using Matrix2 = Eigen::Matrix2d;
using Matrix3 = Eigen::Matrix3d;
using Matrix4 = Eigen::Matrix4d;
