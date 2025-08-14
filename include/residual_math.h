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
  real_t dd_residual[3]; // THETA_THETA, PHI_PHI, THETA_PHI
} residual_cache_t;

namespace residualMath {

/**
 * Computes the residuals for all measurements in the bucket.
 * Stores the residual in the residual_cache.
 *
 * WARNING: lineMath::update_derivatives Must be called before this function
 * is called to ensure that line.D_ortho is up to date.
 */
__device__ void compute_residual(line_t &line, const Vector3 &K,
                                 const Vector3 &W, const real_t drift_radius,
                                 const int tid, const int num_mdt_measurements,
                                 const int num_rpc_measurements,
                                 residual_cache_t &residual_cache);

/**
 * Computes the delta residuals for all measurements in the bucket.
 * Stores the delta residuals in the residual_cache.
 *
 * WARNING: residualMath::compute_residual Must be called before this function
 * is called to ensure that the yz_residual_sign is up to date.
 */
__device__ void compute_delta_residuals(line_t &line, const int tid,
                                        const int num_mdt_measurements,
                                        const int num_rpc_measurements,
                                        const Vector3 &K, const Vector3 &W,
                                        residual_cache_t &residual_cache);

__device__ void compute_dd_residuals(line_t &line, const int tid,
                                     const int num_mdt_measurements,
                                     const int num_rpc_measurements,
                                     const Vector3 &K, const Vector3 &W,
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

/**
 * Computes the chi2 for the line given the measurements in the bucket
 *
 * @returns Chi2 value for the line, ONLY ON THREAD 0
 */
__device__ real_t get_chi2(cg::thread_block_tile<MAX_MPB> &bucket_tile,
                           real_t inverse_sigma_squared,
                           residual_cache_t &residual_cache);
} // namespace residualMath