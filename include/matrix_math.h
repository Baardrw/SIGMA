#pragma once
#include "config.h"
#include <Eigen/Dense>




#define INVERSE_VECTOR3(v) Vector3(1.0 / (v)(0), 1.0 / (v)(1), 1.0 / (v)(2))

// Overload for matrix multiplication
__device__ inline Vector3 elementwise_mult(const Vector3 &v1, const Vector3 &v2) {
  return Vector3(v1(0) * v2(0), v1(1) * v2(1), v1(2) * v2(2));
}

namespace matrixMath {
__device__ bool invert_2x2(const Matrix2 &input, Matrix2 &output);

} // namespace matrixMath