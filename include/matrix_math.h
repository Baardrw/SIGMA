#pragma once
#include "config.h"
#define EIGEN_DONT_VECTORIZE
#define EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT
#include <Eigen/Dense>
#include <cooperative_groups.h>

#define INVERSE_VECTOR3(v) Vector3(1.0 / (v)(0), 1.0 / (v)(1), 1.0 / (v)(2))

using namespace cooperative_groups;
namespace cg = cooperative_groups;

namespace matrixMath {
__device__ inline Vector3 elementwise_mult(const Vector3 &v1,
                                           const Vector3 &v2) {
  return Vector3(v1(0) * v2(0), v1(1) * v2(1), v1(2) * v2(2));
}

__device__ bool invert_2x2(const Matrix2 &input, Matrix2 &output);

// Does an in place inversion of a 4x4 matrix using all threads in the CG
__device__ bool invert_4x4(cg::thread_block_tile<TILE_SIZE> &bucket_tile,
                           Matrix4 &input_ouput);

__global__ void test_invert_4x4(Matrix4 *input_matrices,
                                Matrix4 *output_matrices, int *results,
                                int num_matrices);
} // namespace matrixMath