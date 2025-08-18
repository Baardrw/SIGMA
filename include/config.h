#pragma once
#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <vector_types.h>


#define EPSILON 1e-6
#define FULL_MASK 0xffffffff
#define MAX_MPB 16 // Max measurements per bucket
#define OVERFLOW_TILE_SIZE                                                     \
  32 // Amount of threads needed to handle overflowing buckets

typedef double real_t;

using Vector3 = Eigen::Vector3d;
using Vector4 = Eigen::Vector4d;
using Matrix3 = Eigen::Matrix3d;
using Matrix4 = Eigen::Matrix4d;
