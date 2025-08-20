#pragma once
#include <Eigen/Dense>
#include "config.h"

namespace matrixMath {
__device__ bool invert_2x2(const Matrix2 &input, Matrix2 &output);
} // namespace matrixMath