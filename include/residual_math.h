#pragma once
#include <Eigen/Dense>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include "config.h"
#include "data_structures.h"

using namespace cooperative_groups;
namespace cg = cooperative_groups;

namespace residualMath {

__host__ std::vector<real_t>
compute_residuals(line_t &line, const std::vector<Vector3> &K,
                  const std::vector<real_t> &drift_radius,
                  const int num_mdt_measurements,
                  const int num_rpc_measurements);

/**
 * Computes the residuals for all measurements in the bucket.
 * Stores the residual in the residual_cache.
 *
 * WARNING: lineMath::update_derivatives Must be called before this function
 * is called to ensure that line.D_ortho is up to date.
 */
__device__ void compute_residual(line_t &line, const int tid,
                                 const int num_mdt_measurements,
                                 const int num_rpc_measurements,
                                 const measurement_cache_t &measurement_cache,
                                 residual_cache_t &residual_cache);


/**
 * More optimized version of computing all the residuals in different functions,
 * avoids function calling overheads,
 * and allows to easier reuse computation results.

 * This function fills the residual_cache with all the residuals, delta
 residuals, and delta delta residuals
 */
__device__ void
update_residual_cache(line_t &line, const int tid,
                          const int num_mdt_measurements,
                          const int num_rpc_measurements,
                          const measurement_cache_t &measurement_cache,
                          residual_cache_t &residual_cache);


/**
 * Computes the gradient vector for the line via a shfl_down reduction
 * across the threads in the bucket_tile.
 *
 * WARNING: the gradient is only valid in thread 0 of the bucket_tile.
 */
template <unsigned int TILE_SIZE>
__device__ Vector4 get_gradient(cg::thread_block_tile<TILE_SIZE> &bucket_tile,
                                const Vector3 &inverse_sigma_squared,
                                residual_cache_t &residual_cache);

template <unsigned int TILE_SIZE>
__device__ Matrix4 get_hessian(cg::thread_block_tile<TILE_SIZE> &bucket_tile,
                               const Vector3 &inverse_sigma_squared,
                               residual_cache_t &residual_cache);

__host__ real_t get_chi2(const std::vector<real_t> &residuals,
                         const std::vector<real_t> &inverse_sigma_squared);
/**
 * Computes the chi2 for the line given the measurements in the bucket
 *
 * @returns Chi2 value for the line, ONLY ON THREAD 0
 */
template <unsigned int TILE_SIZE>
__device__ real_t get_chi2(cg::thread_block_tile<TILE_SIZE> &bucket_tile,
                           const Vector3 &inverse_sigma_squared,
                           residual_cache_t &residual_cache);
} // namespace residualMath