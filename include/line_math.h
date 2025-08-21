#pragma once
#include <Eigen/Dense>
#include <cuda_runtime.h>

#include "config.h"
#include "data_structures.h"



namespace lineMath {

// Helper to compute D_ortho when only D_ortho is needed and not the other
// derived quantities
__host__ __device__ void compute_D_ortho(line_t &line);

// Creates a line from the given parameters
__host__ __device__ void create_line(real_t x0, real_t y0, real_t phi,
                                            real_t theta, line_t &line);

__host__ __device__ void
update_derivatives(line_t &line);
} // namespace lineMath