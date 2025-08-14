#pragma once
#include <Eigen/Dense>

namespace matrixMath {
__device__ bool invert_2x2(const Eigen::Matrix2f &input, Eigen::Matrix2f &output);
} // namespace matrixMath