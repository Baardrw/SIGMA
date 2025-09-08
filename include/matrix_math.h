#pragma once
#include "config.h"
#include <cooperative_groups.h>

#include "data_structures.h"

#define INVERSE_VECTOR3(v) Vector3(1.0 / v.x, 1.0 / v.y, 1.0 / v.z)

using namespace cooperative_groups;
namespace cg = cooperative_groups;

namespace matrixMath {
__device__ inline Vector3 elementwise_mult(const Vector3 &v1,
                                           const Vector3 &v2) {
  return Vector3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}

__device__ Vector4 matrix_vector_mult(Matrix4 mat, Vector4 vec);

__device__ bool invert_2x2(const Matrix2 &input, Matrix2 &output);

__device__ bool invert_4x4(cg::thread_block_tile<TILE_SIZE> &bucket_tile,
                           Matrix4 &input, Matrix4 &output);

__global__ void test_invert_4x4(Matrix4 *input_matrices,
                                Matrix4 *output_matrices, int *results,
                                int num_matrices);
} // namespace matrixMath