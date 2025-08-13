#pragma once
#include <Eigen/Dense>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include "config.h"
#include "data_structures.h"

using namespace cooperative_groups;
namespace cg = cooperative_groups;

namespace residualMath {
__device__ real_t compute_residual(struct Data *data, line_t &line,
                                   int bucket_start, int rpc_start,
                                   int bucket_end);

__device__ real_t compute_chi2(cg::thread_block_tile<16> block_tile,
                               struct Data *data, line_t &line,
                               int bucket_start, int rpc_start, int bucket_end);
} // namespace residualMath