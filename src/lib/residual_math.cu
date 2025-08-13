#include "config.h"
#include "data_structures.h"
#include "line_math.h"
#include "residual_math.h"

using namespace cooperative_groups;
namespace cg = cooperative_groups;

extern __constant__ real_t W_components[3];

namespace residualMath {

// TODO: Refactor the same way as delta residuals is structured
__device__ void compute_residual(struct Data *data, line_t &line, int tid,
                                 int bucket_start, int rpc_start,
                                 int bucket_end,
                                 residual_cache_t &residual_cache) {
  unsigned int mdt_measurments = rpc_start - bucket_start;

  real_t yz_res = 0.0f;
  if (tid < mdt_measurments) {
    const Vector3 T = SENSOR_POS(data, bucket_start + tid);
    const Vector3 K = T - line.S0;
    const Vector3 W = {W_components[0], W_components[1], W_components[2]};
    const real_t drift_radius = data->drift_radius[bucket_start + tid];

    yz_res = abs(K.cross(line.D_ortho).dot(W)) - drift_radius;
  }

  residual_cache.yz_residual_sign = (yz_res < 0.0f) ? -1.0f : 1.0f;
  residual_cache.residual = yz_res;
}

__device__ inline void compute_mdt_measurement_delta_residuals(
    Data *data, line_t &line, const int tid, const int data_index,
    const int num_mdt_measurements, const Vector3 &W,
    residual_cache_t &residual_cache) {

  // Zero out all delta residuals
  for (int i = 0; i < 4; i++) {
    residual_cache.delta_residual[i] = 0.0f;
  }

  if (tid < num_mdt_measurements) {
    // Shared data
    real_t sign = residual_cache.yz_residual_sign;

    // ====== Intersection Residuals =====
#pragma unroll
    for (int i = X0; i <= Y0; i++) {
      Vector3 delta_K = -line.dS0[i];
      residual_cache.delta_residual[i] +=
          sign * delta_K.cross(line.D_ortho).dot(W);
    }

    //===== Direction Residuals =====
#pragma unroll
    for (int i = THETA; i <= PHI; i++) {
      Vector3 T = SENSOR_POS(data, data_index);
      Vector3 K = T - line.S0;
      residual_cache.delta_residual[i] +=
          sign * K.cross(line.dD_ortho[i]).dot(W);
    }
  }
}

__device__ void compute_delta_residuals(Data *data, line_t &line, int tid,
                                        int bucket_start, int rpc_start,
                                        int bucket_end,
                                        residual_cache_t &residual_cache) {
  const Vector3 W = {W_components[0], W_components[1], W_components[2]};
  const int num_mdt_measurements = rpc_start - bucket_start;
  const int data_index = bucket_start + tid;

  compute_mdt_measurement_delta_residuals(
      data, line, tid, data_index, num_mdt_measurements, W, residual_cache);
}

/**
 * Computes the chi2 for the line given the measurements in the bucket
 *
 * @returns Chi2 value for the line, ONLY ON THREAD 0
 */
__device__ real_t get_chi2(cg::thread_block_tile<MAX_MPB> &bucket_tile,
                           struct Data *data, line_t &line,
                           residual_cache_t &residual_cache, int bucket_start,
                           int rpc_start, int bucket_end) {
  real_t chi2 = 0.0f;

  real_t inverse_sigma_squared = 0.0f;
  if (bucket_tile.thread_rank() < bucket_end - bucket_start) {
    inverse_sigma_squared +=
        1.0f / (data->sigma[bucket_start + bucket_tile.thread_rank()] *
                data->sigma[bucket_start + bucket_tile.thread_rank()]);
  }

  real_t residual = residual_cache.residual;
  real_t chi_val = residual * residual * inverse_sigma_squared;

  for (int i = warpSize / 2; i >= 1; i /= 2) {
    bucket_tile.sync();
    chi_val += bucket_tile.shfl_down(chi_val, i);
  }

  chi2 = chi_val;
  return chi2;
}

__device__ Vector4 get_gradient(cg::thread_block_tile<16> &bucket_tile,
                                Data *data, line_t &line,
                                residual_cache_t &residual_cache,
                                int bucket_start, int rpc_start,
                                int bucket_end) {
  Vector4 gradient = {0.0f, 0.0f, 0.0f, 0.0f};
}
} // namespace residualMath