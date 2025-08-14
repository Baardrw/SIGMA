#include "matrix_math.h"


namespace matrixMath {

__device__ bool invert_2x2(const Eigen::Matrix2f &input, Eigen::Matrix2f &output) {
    float det = input.determinant();
    if (fabsf(det) < 1e-7) {
        return false; // Matrix is singular, cannot invert
    }
    
    output(0, 0) = input(1, 1) / det;
    output(0, 1) = -input(0, 1) / det;
    output(1, 0) = -input(1, 0) / det;
    output(1, 1) = input(0, 0) / det;
    
    return true; 
}

}