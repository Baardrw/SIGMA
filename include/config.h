#pragma once
#include <cuda_runtime.h>
#include <vector_types.h>
#include <Eigen/Dense>

#define EPSILON 1e-6
#define FULL_MASK 0xffffffff
#define MAX_MPB 16 // Max measurements per bucket

typedef float real_t;
typedef float3 real3_t;

// if float3 is used:
#if (real3_t == float3)
inline __device__ real_t norm3(real3_t v) {
  return static_cast<real_t>(norm3df(v.x, v.y, v.z));
}

using Vector3 = Eigen::Vector3f; 
using Vector4 = Eigen::Vector4f;
using Matrix3 = Eigen::Matrix3f;
using Matrix4 = Eigen::Matrix4f;

# else
inline __device__ real_t norm3(real3_t v) {
  return static_cast<real_t>(norm3d(v.x, v.y, v.z));
}
using Vector3 = Eigen::Vector3d;
using Vector4 = Eigen::Vector4d;
using Matrix3 = Eigen::Matrix3d;
using Matrix4 = Eigen::Matrix4d;
#endif