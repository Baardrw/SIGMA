#pragma once
#include <__clang_cuda_math.h>
#include <cuda_runtime.h>
#include <vector_types.h>

#define EPSILON 1e-6

typedef float real_t;
typedef float3 real3_t;

// if float3 is used:
#if (real3_t == float3)
inline __device__ real_t norm3(real3_t v) {
  return static_cast<real_t>(norm3df(v.x, v.y, v.z));
}
# else
inline __device__ real_t norm3(real3_t v) {
  return static_cast<real_t>(norm3d(v.x, v.y, v.z));
}
#endif