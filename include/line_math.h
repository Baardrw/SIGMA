#pragma once
#include <Eigen/Dense>
#include <cuda_runtime.h>

#include "config.h"
#include "data_structures.h"

namespace lineMath {

// Helper to compute D_ortho when only Dortho is needed and not the other
// derived quantities
__host__ __device__ void compute_Dortho(line_t &line, const Vector3 &W);

// Creates a line from the given parameters
__host__ __device__ void create_line(real_t x0, real_t y0, real_t phi,
                                            real_t theta, line_t &line);

inline __host__ __device__ void
update_derived_line_quantities(line_t &line, Vector3 &sensor_direction);
} // namespace lineMath