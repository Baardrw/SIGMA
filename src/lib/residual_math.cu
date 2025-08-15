#include "config.h"
#include "data_structures.h"
#include "line_math.h"
#include "residual_math.h"

#include <vector>

using namespace cooperative_groups;
namespace cg = cooperative_groups;

extern __constant__ real_t W_components[3];

namespace residualMath {

// ================ Residuals ================
// TODO: Refactor the same way as delta residuals is structured

__host__ std::vector<real_t> compute_residuals(line_t &line,
                                               const std::vector<Vector3> &K,
                                               const std::vector<real_t> &drift_radius,
                                               const int num_mdt_measurements,
                                               const int num_rpc_measurements) {

  std::vector<real_t> residuals(num_mdt_measurements + num_rpc_measurements,
                                0.0f);
  Vector3 W(W_components[0], W_components[1], W_components[2]);
  for (int i = 0; i < num_mdt_measurements; i++) {
    real_t cross_product = K[i].cross(line.D_ortho).dot(W);
    residuals[i] = abs(cross_product) - drift_radius[i];
  }

  return residuals;
}
__device__ void compute_residual(line_t &line, const Vector3 &K,
                                 const Vector3 &W, const real_t drift_radius,
                                 const int tid, const int num_mdt_measurements,
                                 const int num_rpc_measurements,
                                 residual_cache_t &residual_cache) {

  real_t yz_res = 0.0f;
  real_t cross_product = 0.0f;
  if (tid < num_mdt_measurements) {
    cross_product = K.cross(line.D_ortho).dot(W);
    yz_res = abs(cross_product) - drift_radius;
  }

  residual_cache.yz_residual_sign = (cross_product < 0.0f) ? -1.0f : 1.0f;
  residual_cache.residual = yz_res;
}

// ================ Delta Residuals ================

__device__ inline void
compute_mdt_delta_residuals(line_t &line, const Vector3 K, const Vector3 &W,
                            residual_cache_t &residual_cache) {

  // Shared data
  real_t sign = residual_cache.yz_residual_sign;

  // ====== Intersection Residuals =====
#pragma unroll
  for (int i = X0; i <= Y0; i++) {
    Vector3 delta_K = -line.dS0[i - X0];
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
    compute_mdt_delta_residuals(line, K, W, residual_cache);
  }

  if (num_mdt_measurements <= tid &&
      tid < num_mdt_measurements + num_rpc_measurements) {
    compute_rpc_delta_residuals(line, tid, W, residual_cache);
  }
}

// ================ Delta Delta Residuals ================
__device__ void compute_dd_residuals(line_t &line, const int tid,
                                     const int num_mdt_measurements,
                                     const int num_rpc_measurements,
                                     const Vector3 &K, const Vector3 &W,
                                     residual_cache_t &residual_cache) {
  // Zero out all delta delta residuals
  for (int i = 0; i < 3; i++) {
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
__host__ real_t get_chi2(const std::vector<real_t> &residuals,
                         const std::vector<real_t> &inverse_sigma_squared) {
  real_t chi2 = 0.0f;
  for (size_t i = 0; i < residuals.size(); i++) {
    chi2 += residuals[i] * residuals[i] * inverse_sigma_squared[i];
  }
  return chi2;
}

template <unsigned int TILE_SIZE>
__device__ real_t get_chi2(cg::thread_block_tile<TILE_SIZE> &bucket_tile,
                           real_t inverse_sigma_squared,
                           residual_cache_t &residual_cache) {
  real_t chi2 = 0.0f;

  real_t residual = residual_cache.residual;
  real_t chi_val = residual * residual * inverse_sigma_squared;

  for (int i = bucket_tile.num_threads() / 2; i >= 1; i /= 2) {
    bucket_tile.sync();
    chi_val += bucket_tile.shfl_down(chi_val, i);
  }

  chi2 = chi_val;
  return chi2;
}

template <unsigned int TILE_SIZE>
__device__ Vector4 get_gradient(cg::thread_block_tile<TILE_SIZE> &bucket_tile,
                                real_t inverse_sigma_squared,
                                residual_cache_t &residual_cache) {
  Vector4 gradient = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int i = 0; i < 4; i++) {
    gradient[i] = 2 * inverse_sigma_squared * residual_cache.residual *
                  residual_cache.delta_residual[i];
  }

  for (int i = bucket_tile.num_threads() / 2; i >= 1; i /= 2) {
    bucket_tile.sync();
    for (int j = 0; j < 4; j++) {
      gradient[j] += bucket_tile.shfl_down(gradient[j], i);
    }
  }

  return gradient;
}

__device__ real_t get_delta_delta_residual(int param1_idx, int param2_idx,
                                           residual_cache_t &residual_cache) {
  if (param1_idx == Y0 || param2_idx == Y0 || param1_idx == X0 ||
      param2_idx == X0) {
    return 0;
  } else if (param1_idx == THETA && param2_idx == THETA) {
    return residual_cache.dd_residual[DD_THETA_THETA];
  } else if (param1_idx == PHI && param2_idx == PHI) {
    return residual_cache.dd_residual[DD_PHI_PHI];
  } else {
    return residual_cache.dd_residual[DD_THETA_PHI]; // Mixed derivative
  }
}

template <unsigned int TILE_SIZE>
__device__ Matrix4 get_hessian(cg::thread_block_tile<TILE_SIZE> &bucket_tile,
                               real_t inverse_sigma_squared,
                               residual_cache_t &residual_cache) {
  Matrix4 hessian = Matrix4::Zero();

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      if (j >= i) {
        hessian(i, j) = 2 * inverse_sigma_squared *
                        (residual_cache.delta_residual[i] *
                             residual_cache.delta_residual[j] +
                         get_delta_delta_residual(i, j, residual_cache));
        hessian(j, i) = hessian(i, j); // Symmetric matrix
      }
    }
  }

  for (int i = bucket_tile.num_threads() / 2; i >= 1; i /= 2) {
    bucket_tile.sync();
    for (int j = 0; j < 4; j++) {
      for (int k = j; k < 4; k++) {
        hessian(j, k) += bucket_tile.shfl_down(hessian(j, k), i);
        hessian(k, j) = hessian(j, k); // Ensure symmetry
      }
    }
  }

  return hessian;
}

// Explicit template instantiations for linking
template __device__ real_t get_chi2<16>(cg::thread_block_tile<16> &bucket_tile,
                                        real_t inverse_sigma_squared,
                                        residual_cache_t &residual_cache);

template __device__ real_t get_chi2<32>(cg::thread_block_tile<32> &bucket_tile,
                                        real_t inverse_sigma_squared,
                                        residual_cache_t &residual_cache);

template __device__ Vector4 get_gradient<16>(
    cg::thread_block_tile<16> &bucket_tile, real_t inverse_sigma_squared,
    residual_cache_t &residual_cache);

template __device__ Vector4 get_gradient<32>(
    cg::thread_block_tile<32> &bucket_tile, real_t inverse_sigma_squared,
    residual_cache_t &residual_cache);

template __device__ Matrix4
get_hessian<16>(cg::thread_block_tile<16> &bucket_tile,
                real_t inverse_sigma_squared, residual_cache_t &residual_cache);

template __device__ Matrix4
get_hessian<32>(cg::thread_block_tile<32> &bucket_tile,
                real_t inverse_sigma_squared, residual_cache_t &residual_cache);
} // namespace residualMath