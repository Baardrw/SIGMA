#pragma once
#include <Eigen/Dense>
#include <cuda_runtime.h>

#include "config.h"
#include "data_structures.h"

namespace residualMath {
__device__ real_t compute_residual(struct Data *data, line_t &line,
                                   int bucket_start, int rpc_start,
                                   int bucket_end);

__device__ real_t compute_chi2(struct Data *data, line_t &line,
                               int bucket_start, int rpc_start, int bucket_end);
} // namespace residualMath