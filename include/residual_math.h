#pragma once
#include <Eigen/Dense>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include "config.h"
#include "data_structures.h"

using namespace cooperative_groups;
namespace cg = cooperative_groups;

typedef struct {
  real_t residual;
  real_t yz_residual_sign;
  real_t delta_residual[4];
} residual_cache_t;

namespace residualMath {

/**
 * Computes the residuals for all measurements in the bucket.
 * Stores the residual in the residual_cache.
 *
 * WARNING: lineMath::update_derivatives Must be called before this function
 * is called to ensure that line.D_ortho is up to date.
 */
__device__ void compute_residual(Data *data, line_t &line, int tid,
                                 int bucket_start, int rpc_start,
                                 int bucket_end,
                                 residual_cache_t &residual_cache);

/**
 * Computes the delta residuals for all measurements in the bucket.
 * Stores the delta residuals in the residual_cache.
 *
 * WARNING: residualMath::compute_residual Must be called before this function
 * is called to ensure that the yz_residual_sign is up to date.
 */
__device__ void compute_delta_residuals(Data *data, line_t &line, int tid,
                                        int bucket_start, int rpc_start,
                                        int bucket_end,
                                        residual_cache_t &residual_cache);

__device__ void compute_dd_residuals(Data *data, line_t &line, int tid,
                                     int bucket_start, int rpc_start,
                                     int bucket_end,
                                     residual_cache_t &residual_cache);

__device__ Vector4 get_gradient(cg::thread_block_tile<16> &bucket_tile,
                                Data *data, line_t &line,
                                residual_cache_t &residual_cache,
                                int bucket_start, int rpc_start,
                                int bucket_end);

__device__ Matrix4 get_hessian(cg::thread_block_tile<16> &bucket_tile,
                               Data *data, line_t &line,
                               residual_cache_t &residual_cache,
                               int bucket_start, int rpc_start, int bucket_end);

__device__ real_t get_chi2(cg::thread_block_tile<16> &bucket_tile, Data *data,
                           line_t &line, residual_cache_t &residual_cache,
                           int bucket_start, int rpc_start, int bucket_end);
} // namespace residualMath