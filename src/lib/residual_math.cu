#include "config.h"
#include "data_structures.h"
#include "line_math.h"
#include "residual_math.h"

using namespace cooperative_groups;
namespace cg = cooperative_groups;

extern __constant__ real_t W_components[3];

namespace residualMath {

// ================ Residuals ================
// TODO: Refactor the same way as delta residuals is structured
__device__ void compute_residual(line_t &line, const Vector3 &K,
                                 const Vector3 &W, const real_t drift_radius,
                                 const int tid,
                                 const int num_mdt_measurements,
                                 const int num_rpc_measurements,
                                 residual_cache_t &residual_cache) {

  real_t yz_res = 0.0f;
  if (tid < num_mdt_measurements) {
    yz_res = abs(K.cross(line.D_ortho).dot(W)) - drift_radius;
  }

  residual_cache.yz_residual_sign = (yz_res < 0.0f) ? -1.0f : 1.0f;
  residual_cache.residual = yz_res;
}

// ================ Delta Residuals ================

__device__ inline void
compute_mdt_measurement_delta_residuals(line_t &line, const Vector3 K,
                                        const Vector3 &W,
                                        residual_cache_t &residual_cache) {

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
    residual_cache.delta_residual[i] += sign * K.cross(line.dD_ortho[i]).dot(W);
  }
}

// TODO:
__device__ inline void
compute_rpc_delta_residuals(line_t &line, const int tid, const Vector3 &W,
                            residual_cache_t &residual_cache) {}

__device__ void compute_delta_residuals(line_t &line, const int tid,
                                        const int num_mdt_measurements,
                                        const int num_rpc_measurements,
                                        const Vector3 &K, const Vector3 &W,
                                        residual_cache_t &residual_cache) {

  // Zero out all delta residuals
  for (int i = 0; i < 4; i++) {
    residual_cache.delta_residual[i] = 0.0f;
  }

  if (tid < num_mdt_measurements) {
    compute_mdt_measurement_delta_residuals(line, K, W, residual_cache);
  }

  if (tid < num_rpc_measurements) {
    compute_rpc_delta_residuals(line, tid, W, residual_cache);
  }
}

// ================ Delta Delta Residuals ================
__device__ void compute_dd_residuals(line_t &line, const int tid,
                                     const int num_mdt_measurements,
                                     const int num_rpc_measurements,
                                     const Vector3 &K, const Vector3 &W,
                                     residual_cache_t &residual_cache) {
  for (int i = 0; i < 4; i++) {
    residual_cache.dd_residual[i] = 0.0f;
  }
  const real_t sign = residual_cache.yz_residual_sign;

  if (tid < num_mdt_measurements) {
    for (int i = DD_THETA_THETA; i <= DD_THETA_PHI; i++) {
      // Compute the delta delta residuals
      residual_cache.dd_residual[i] += sign * K.cross(line.ddD_ortho[i]).dot(W);
    }
  }
}

// ================ Chi2 ================
__device__ real_t get_chi2(cg::thread_block_tile<MAX_MPB> &bucket_tile,
                           real_t inverse_sigma_squared,
                           residual_cache_t &residual_cache) {
  real_t chi2 = 0.0f;

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