#include "config.h"
#include "data_structures.h"
#include "residual_math.h"

#include <vector>

using namespace cooperative_groups;
namespace cg = cooperative_groups;

extern __constant__ real_t W_components[3];

namespace residualMath {

// ================ Residuals ================
// TODO: Refactor the same way as delta residuals is structured

__host__ std::vector<real_t>
compute_residuals(line_t &line, const std::vector<Vector3> &K,
                  const std::vector<real_t> &drift_radius,
                  const int num_mdt_measurements,
                  const int num_rpc_measurements) {

  std::vector<real_t> residuals;
  Vector3 W(1, 0, 0);
  for (int i = 0; i < num_mdt_measurements; i++) {
    real_t cross_product = K[i].cross(line.D_ortho).dot(W);
    residuals.push_back(abs(cross_product) - drift_radius[i]);
  }

  return residuals;
}

__device__ __forceinline__ void
compute_mdt_residuals(line_t &line, const Vector3 &K, real_t drift_radius,
                      residual_cache_t &residual_cache) {
  real_t cross_product = K.cross(line.D_ortho)[0];
  residual_cache.residual[0] = abs(cross_product) - drift_radius;
}

__device__ __forceinline__ void
compute_rpc_residuals(line_t &line, const Vector3 &plane_normal,
                      const Vector3 &sensor_dir, const Vector3 &measurement_pos,
                      residual_cache_t &residual_cache) {
  // Compute the residual for the RPC measurement
  // real_t distance =
}

// Used by seed line to avoid computing derivatives unnecessarily
__device__ void compute_residual(line_t &line, const int tid,
                                 const int num_mdt_measurements,
                                 const int num_rpc_measurements,
                                 const measurement_cache_t &measurement_cache,
                                 residual_cache_t &residual_cache) {

// Zero out the residual cache
#pragma unroll
  for (int i = 0; i < 3; i++) {
    residual_cache.residual[i] = 0.0f;
  }

  if (tid < num_mdt_measurements) {
    compute_mdt_residuals(line, measurement_cache.connection_vector,
                          measurement_cache.drift_radius, residual_cache);
  } else {
    compute_rpc_residuals(line, measurement_cache.plane_normal,
                          measurement_cache.sensor_direction,
                          measurement_cache.sensor_pos, residual_cache);
  }
}

// TODO: Remove unneccessary calculations from the cross product computation
__device__ __forceinline__ void compute_straw_residuals_and_derivatives(
    line_t &line, const measurement_cache_t &measurement_cache,
    residual_cache_t &residual_cache) {
  // Local vars
  const Vector3 &K = measurement_cache.connection_vector;
  const real_t drift_radius = measurement_cache.drift_radius;
  const Vector3 &W = MDT_DIR;

  // ================ Residual ================
  real_t cross_product = K.cross(line.D_ortho)[0];
  residual_cache.residual[BENDING] = abs(cross_product) - drift_radius;
  real_t yz_residual_sign = (cross_product < 0.0f) ? -1.0f : 1.0f;

// ================ Delta Residual ===============
#pragma unroll
  for (int i = X0; i <= Y0; i++) {
    Vector3 delta_K = -line.dS0[i - X0];
    residual_cache.delta_residual[i][BENDING] +=
        yz_residual_sign * delta_K.cross(line.D_ortho)[0];
  }

#pragma unroll
  for (int i = THETA; i <= PHI; i++) {
    residual_cache.delta_residual[i][BENDING] +=
        yz_residual_sign * K.cross(line.dD_ortho[i])[0];
  }

  // ================ Delta Delta Residual ===============
#pragma unroll
  for (int i = DD_THETA_THETA; i <= DD_THETA_PHI; i++) {
    // Compute the delta delta residuals
    residual_cache.dd_residual[i][BENDING] +=
        yz_residual_sign * K.cross(line.ddD_ortho[i])[0];
  }
}

__device__ __forceinline__ void compute_strip_residuals_and_derivatives(
    line_t &line, const measurement_cache_t &measurement_cache,
    residual_cache_t &residual_cache) {
  // Local vars
  const Vector3 &P = measurement_cache.sensor_pos;
  const Vector3 &N = measurement_cache.plane_normal;
  const Vector3 &S0 = line.S0;
  const Vector3 &D = line.D;
  const real_t inverse_N_dot_D = 1.0f / N.dot(D);

  // TODO:
  const Vector3 v1 = measurement_cache.sensor_direction;
  const Vector3 v2 = N.cross(v1); // IDK i guess this is bending?

  // ================ Residual ================
  real_t traveled_distance = (P.dot(N) - S0.dot(N)) * inverse_N_dot_D;
  Vector3 distance_vector = S0 + traveled_distance * D - P;

  residual_cache.residual[BENDING] = (distance_vector).dot(v2);
  residual_cache.residual[NON_BENDING] = (distance_vector).dot(v1);

  // ================ Delta Residual ===============
#pragma unroll
  for (int i = X0; i <= Y0; i++) {
    Vector3 delta_r_vector =
        line.dS0[i - X0] + line.dS0[i - X0].dot(N) * inverse_N_dot_D * D;

    residual_cache.delta_residual[i][BENDING] += (delta_r_vector).dot(v2);
    residual_cache.delta_residual[i][NON_BENDING] += (delta_r_vector).dot(v1);
  }

#pragma unroll
  for (int i = THETA; i <= PHI; i++) {
    Vector3 delta_r_vector =
        traveled_distance *
        (line.dD[i] + line.dD[i].dot(N) * inverse_N_dot_D * D);

    residual_cache.delta_residual[i][BENDING] += (delta_r_vector).dot(v2);
    residual_cache.delta_residual[i][NON_BENDING] += (delta_r_vector).dot(v1);
  }
}

__device__ void
update_residual_cache(line_t &line, const int tid,
                      const int num_mdt_measurements,
                      const int num_rpc_measurements,
                      const measurement_cache_t &measurement_cache,
                      residual_cache_t &residual_cache) {
// Zero out the residual cache
#pragma unroll
  for (int i = 0; i < 3; i++) {
    residual_cache.residual[i] = 0.0f;
    residual_cache.dd_residual[i] = Vector3(0.0f, 0.0f, 0.0f);
  }

#pragma unroll
  for (int i = 0; i < 4; i++) {
    residual_cache.delta_residual[i] = Vector3(0.0f, 0.0f, 0.0f);
  }

  if (tid < num_mdt_measurements) {
    compute_straw_residuals_and_derivatives(line, measurement_cache,
                                            residual_cache);
  } else if (tid >= num_mdt_measurements &&
             tid < num_mdt_measurements + num_rpc_measurements) {
    compute_strip_residuals_and_derivatives(line, measurement_cache,
                                            residual_cache);
  }
}

__device__ __forceinline__ real_t
accumulate(const Vector3 &v1, const Vector3 &v2,
           const Vector3 &inverse_sigma_squared) {
  real_t result = 0.0f;

#pragma unroll
  for (int i = 0; i < 3; i++) {
    result += v1[i] * v2[i] * inverse_sigma_squared[i];
  }

  return result;
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
                           const Vector3 &inverse_sigma_squared,
                           residual_cache_t &residual_cache) {
  real_t chi2 = 0.0f;
  real_t chi_val = accumulate(residual_cache.residual, residual_cache.residual,
                              inverse_sigma_squared);

  for (int i = bucket_tile.num_threads() / 2; i >= 1; i /= 2) {
    bucket_tile.sync();
    chi_val += bucket_tile.shfl_down(chi_val, i);
  }

  chi2 = chi_val;
  return chi2;
}

template <unsigned int TILE_SIZE>
__device__ Vector4 get_gradient(cg::thread_block_tile<TILE_SIZE> &bucket_tile,
                                const Vector3 &inverse_sigma_squared,
                                residual_cache_t &residual_cache) {
  Vector4 gradient = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int i = 0; i < 4; i++) {
    gradient[i] =
        2 * accumulate(residual_cache.residual,
                       residual_cache.delta_residual[i], inverse_sigma_squared);
  }

  for (int i = bucket_tile.num_threads() / 2; i >= 1; i /= 2) {
    for (int j = 0; j < 4; j++) {
      bucket_tile.sync();
      gradient[j] += bucket_tile.shfl_down(gradient[j], i);
    }
  }

  return gradient;
}

__device__ Vector3 get_delta_delta_residual(int param1_idx, int param2_idx,
                                            residual_cache_t &residual_cache) {
  if (param1_idx == Y0 || param2_idx == Y0 || param1_idx == X0 ||
      param2_idx == X0) {
    return Vector3(0.0f, 0.0f, 0.0f); // No delta delta residual for Y0 or X0
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
                               const Vector3 &inverse_sigma_squared,
                               residual_cache_t &residual_cache) {
  Matrix4 hessian = Matrix4::Zero();

#pragma unroll
  for (int i = 0; i < 4; i++) {
#pragma unroll
    for (int j = 0; j < 4; j++) {
      if (j >= i) {
        // hessian(i, j) = 2 * inverse_sigma_squared *
        //                 (residual_cache.delta_residual[i] *
        //                      residual_cache.delta_residual[j] +
        //                  get_delta_delta_residual(i, j, residual_cache));
        hessian(i, j) =
            2 * (accumulate(residual_cache.delta_residual[i],
                            residual_cache.delta_residual[j],
                            inverse_sigma_squared) +
                 accumulate(get_delta_delta_residual(i, j, residual_cache),
                            Vector3(1.0f, 1.0f, 1.0f), inverse_sigma_squared));
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
                                        const Vector3 &inverse_sigma_squared,
                                        residual_cache_t &residual_cache);

template __device__ real_t get_chi2<32>(cg::thread_block_tile<32> &bucket_tile,
                                        const Vector3 &inverse_sigma_squared,
                                        residual_cache_t &residual_cache);

template __device__ Vector4 get_gradient<16>(
    cg::thread_block_tile<16> &bucket_tile,
    const Vector3 &inverse_sigma_squared, residual_cache_t &residual_cache);

template __device__ Vector4 get_gradient<32>(
    cg::thread_block_tile<32> &bucket_tile,
    const Vector3 &inverse_sigma_squared, residual_cache_t &residual_cache);

template __device__ Matrix4 get_hessian<16>(
    cg::thread_block_tile<16> &bucket_tile,
    const Vector3 &inverse_sigma_squared, residual_cache_t &residual_cache);

template __device__ Matrix4 get_hessian<32>(
    cg::thread_block_tile<32> &bucket_tile,
    const Vector3 &inverse_sigma_squared, residual_cache_t &residual_cache);
} // namespace residualMath