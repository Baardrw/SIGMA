#pragma once
#include <cuda_runtime.h>
#include <vector_types.h>
#include <Eigen/Dense>

// #define DEBUG 1

#define EPSILON 1e-6
#define FULL_MASK 0xffffffff
#define MAX_MPB 16 // Max measurements per bucket
#define OVERFLOW_TILE_SIZE 32 // Amount of threads needed to handle overflowing buckets

typedef double real_t;
typedef double3 real3_t;

using Vector3 = Eigen::Vector3d;
using Vector4 = Eigen::Vector4d;
using Matrix3 = Eigen::Matrix3d;
using Matrix4 = Eigen::Matrix4d;

// // if float3 is used:
// #if (real3_t == float3)
// inline __device__ real_t norm3(real3_t v) {
//   return static_cast<real_t>(norm3df(v.x, v.y, v.z));
// }

// using Vector3 = Eigen::Vector3f; 
// using Vector4 = Eigen::Vector4f;
// using Matrix3 = Eigen::Matrix3f;
// using Matrix4 = Eigen::Matrix4f;

// # else
// inline __device__ real_t norm3(real3_t v) {
//   return static_cast<real_t>(norm3d(v.x, v.y, v.z));
// }
// using Vector3 = Eigen::Vector3d;
// using Vector4 = Eigen::Vector4d;
// using Matrix3 = Eigen::Matrix3d;
// using Matrix4 = Eigen::Matrix4d;
// #endif