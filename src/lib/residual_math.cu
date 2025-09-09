#include "config.h"
#include "data_structures.h"
#include "line_math.h"
#include "residual_math.h"

#include <vector>

using namespace cooperative_groups;
namespace cg = cooperative_groups;

extern __constant__ real_t dS0_const[2][3];
#define GET_DS0(idx) Vector3(dS0_const[idx])

namespace residualMath {

// ================ Residuals ================

__host__ std::vector<Vector3>
compute_residuals(line_t &line, measurement_cache_t<false> *measurement_cache,
                  const int num_mdt_measurements,
                  const int num_rpc_measurements) {

  std::vector<Vector3> residuals;
  Vector3 W(1, 0, 0);

  for (int i = 0; i < num_mdt_measurements; i++) {
    Vector3 K = measurement_cache[i].connection_vector;
    real_t drift_radius = measurement_cache[i].drift_radius;

    real_t cross_product = K.cross(line.D_ortho).dot(W);
    residuals.push_back(Vector3(abs(cross_product) - drift_radius, 0.0f, 0.0f));
  }

  for (int i = num_mdt_measurements;
       i < num_mdt_measurements + num_rpc_measurements; i++) {
    const Vector3 &P = measurement_cache[i].sensor_pos;
    const Vector3 &N = measurement_cache[i].plane_normal;
    const Vector3 &S0 = line.S0;
    const Vector3 &D = line.D;
    const real_t inverse_N_dot_D = 1.0f / N.dot(D);
    const Vector3 v1 = measurement_cache[i].sensor_direction;
    const Vector3 v2 = N.cross(v1);
    real_t traveled_distance = (P.dot(N) - S0.dot(N)) * inverse_N_dot_D;
    Vector3 distance_vector = S0 + traveled_distance * D - P;

    real_t res_v1 = distance_vector.dot(v1);
    real_t res_v2 = distance_vector.dot(v2);
    residuals.push_back(
        Vector3(res_v2, res_v1, 0.0f)); // Bending, Non-bending);
  }

  return residuals;
}

template <bool Overflow>
__device__ void
compute_mdt_residuals(line_t &line, const Vector3 &K, real_t drift_radius,
                      residual_cache_t<Overflow> &residual_cache) {
  real_t cross_product = K.cross(line.D_ortho)[0];
  residual_cache.residual[0] += abs(cross_product) - drift_radius;
}

template <bool Overflow>
__device__ void
compute_rpc_residuals(line_t &line,
                      const measurement_cache_t<Overflow> &measurement_cache,
                      residual_cache_t<Overflow> &residual_cache) {

  // Reference to store computed residuals in the cache
  // Without overflow we can store the rpc residual in the same cache as mdt
  // residuals because no thread computes both
  Vector3 &rpc_residual = residual_cache.residual;
  if constexpr (Overflow) {
    // If there is an overflow one thread calculates both rpc and mdt, so it has
    // to use another seperate cache
    rpc_residual = residual_cache.rpc_residual;
  }

  // Local vars
  Vector3 P = measurement_cache.sensor_pos;
  const Vector3 &N = measurement_cache.plane_normal;
  const Vector3 &S0 = line.S0;
  const Vector3 &D = line.D;
  Vector3 v1 = measurement_cache.sensor_direction;
  Vector3 v2 = N.cross(v1); // IDK i guess this is bending?

  const real_t inverse_N_dot_D = 1.0f / N.dot(D);

  if constexpr (Overflow) {
    // If there is an overflow there is a contention for the shared variables
    // so we need handle that with different variables for rpc than mdt
    P = measurement_cache.strip_pos;
    v1 = measurement_cache.strip_direction;
    v2 = N.cross(v1);
  }

  // ================ Residual ================
  real_t traveled_distance = (P.dot(N) - S0.dot(N)) * inverse_N_dot_D;
  Vector3 distance_vector = S0 + traveled_distance * D - P;

  rpc_residual[BENDING] += (distance_vector).dot(v2);
  rpc_residual[NON_BENDING] += (distance_vector).dot(v1);
}

template <bool Overflow>
__device__ bool should_compute_rpc(const int tid,
                                   const int num_mdt_measurements,
                                   const int num_rpc_measurements) {
  if constexpr (Overflow) {
    // In overflow mode: first num_rpc_measurements threads compute RPC as well
    // as MDT
    return (tid < num_rpc_measurements);
  } else {
    // In normal mode: threads after MDT range compute RPC
    return (tid >= num_mdt_measurements &&
            tid < num_mdt_measurements + num_rpc_measurements);
  }
}

// Used by seed line to avoid computing derivatives unnecessarily
template <bool Overflow>
__device__ void
compute_residual(line_t &line, const int tid, const int num_mdt_measurements,
                 const int num_rpc_measurements,
                 const measurement_cache_t<Overflow> &measurement_cache,
                 residual_cache_t<Overflow> &residual_cache) {

  // Zero out residual cache
#pragma unroll
  for (int i = 0; i < 3; i++) {
    residual_cache.residual[i] = 0.0f;

    if constexpr (Overflow) {
      residual_cache.rpc_residual[i] = 0.0f;
    }
  }

  // Compute MDT residuals for threads handling MDT measurements
  if (tid < num_mdt_measurements) {
    compute_mdt_residuals<Overflow>(line, measurement_cache.connection_vector,
                                    measurement_cache.drift_radius,
                                    residual_cache);
  }

  if (should_compute_rpc<Overflow>(tid, num_mdt_measurements,
                                   num_rpc_measurements)) {
    compute_rpc_residuals<Overflow>(line, measurement_cache, residual_cache);
  }
}

// TODO: Remove unneccessary calculations from the cross product computation
template <bool Overflow>
__device__ void compute_straw_residuals_and_derivatives(
    line_t &line, const measurement_cache_t<Overflow> &measurement_cache,
    residual_cache_t<Overflow> &residual_cache) {
  // Local vars
  const Vector3 &K = measurement_cache.connection_vector;
  const real_t drift_radius = measurement_cache.drift_radius;

  // ================ Residual ================
  real_t cross_product = K.cross(line.D_ortho)[0];
  residual_cache.residual[BENDING] += abs(cross_product) - drift_radius;
  real_t yz_residual_sign = (cross_product < 0.0f) ? -1.0f : 1.0f;

// ================ Delta Residual ===============
#pragma unroll
  for (int i = X0; i <= Y0; i++) {
    Vector3 delta_K = - GET_DS0(i - X0);
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

template <bool Overflow>
__device__ void compute_strip_residuals_and_derivatives(
    line_t &line, const measurement_cache_t<Overflow> &measurement_cache,
    residual_cache_t<Overflow> &residual_cache) {
  // References to store computed residuals in the correct cache
  // to account for the data overflow. If there is an overflow one thread
  // calculates both rpc and mdt, so it has to use another seperate cache
  Vector3 *residual = &residual_cache.residual;
  Vector3 *delta_residual = residual_cache.delta_residual;
  Vector3 *dd_residual = residual_cache.dd_residual;
  if constexpr (Overflow) {
    residual = &residual_cache.rpc_residual;
    delta_residual = residual_cache.rpc_delta_residual;
    dd_residual = residual_cache.rpc_dd_residual;
  }

  // Local vars
  Vector3 P = measurement_cache.sensor_pos;
  const Vector3 &N = measurement_cache.plane_normal;
  const Vector3 &S0 = line.S0;
  const Vector3 &D = line.D;
  Vector3 v1 = measurement_cache.sensor_direction;
  Vector3 v2 = N.cross(v1); // IDK i guess this is bending?

  const real_t inverse_N_dot_D = 1.0f / N.dot(D);

  if constexpr (Overflow) {
    // If there is an overflow there is a contention for the shared variables
    // so we need handle that with different variables for rpc than mdt
    P = measurement_cache.strip_pos;
    v1 = measurement_cache.strip_direction;
    v2 = N.cross(v1);
  }

  // ================ Residual ================

  real_t traveled_distance = (P.dot(N) - S0.dot(N)) * inverse_N_dot_D;
  Vector3 Sm = S0 + traveled_distance * D;

  residual[0][BENDING] += (Sm - P).dot(v2);
  residual[0][NON_BENDING] += (Sm - P).dot(v1);

  // ================ Delta Residual ===============
#pragma unroll
  for (int i = X0; i <= Y0; i++) {
    Vector3 delta_Sm =
        GET_DS0(i - X0) -N[i-X0] * inverse_N_dot_D * D;

    delta_residual[i][BENDING] += (delta_Sm).dot(v2);
    delta_residual[i][NON_BENDING] += (delta_Sm).dot(v1);
  }

  Vector3 delta_SM[2];
#pragma unroll
  for (int i = THETA; i <= PHI; i++) {
    real_t partial_dist =
        -traveled_distance * line.dD[i].dot(N) * inverse_N_dot_D;

    Vector3 delta_Sm = traveled_distance * line.dD[i] + partial_dist * D;

    delta_SM[i] = delta_Sm;

    delta_residual[i][BENDING] += (delta_Sm).dot(v2);
    delta_residual[i][NON_BENDING] += (delta_Sm).dot(v1);
  }
  if constexpr (Overflow) {
  }

  // ================ Delta Delta Residual ===============
#pragma unroll
  for (int i = DD_THETA_THETA; i <= DD_THETA_PHI; i++) {
    int delta_1, delta_2;
    GET_DELTA_VECTORS(i, line, delta_1, delta_2);
    Vector3 delta_D1 = line.dD[delta_1];
    Vector3 delta_D2 = line.dD[delta_2];

    Vector3 delta_Sm =
        traveled_distance *
            (line.ddD[i] - (N.dot(line.ddD[i]) * inverse_N_dot_D) * D) -
        N.dot(delta_D2) * inverse_N_dot_D * delta_SM[delta_1] -
        N.dot(delta_D1) * inverse_N_dot_D * delta_SM[delta_2];

    dd_residual[i][BENDING] += (delta_Sm).dot(v2);
    dd_residual[i][NON_BENDING] += (delta_Sm).dot(v1);
  }
}

template <bool Overflow>
__device__ void
update_residual_cache(line_t &line, const int tid,
                      const int num_mdt_measurements,
                      const int num_rpc_measurements,
                      const measurement_cache_t<Overflow> &measurement_cache,
                      residual_cache_t<Overflow> &residual_cache) {

// Zero out the residual cache
#pragma unroll
  for (int i = 0; i < 3; i++) {
    residual_cache.residual[i] = 0.0f;
    residual_cache.dd_residual[i] = Vector3(0.0f, 0.0f, 0.0f);

    if constexpr (Overflow) {
      residual_cache.rpc_residual[i] = 0.0f;
      residual_cache.rpc_dd_residual[i] = Vector3(0.0f, 0.0f, 0.0f);
    }
  }

#pragma unroll
  for (int i = 0; i < 4; i++) {
    residual_cache.delta_residual[i] = Vector3(0.0f, 0.0f, 0.0f);

    if constexpr (Overflow) {
      residual_cache.rpc_delta_residual[i] = Vector3(0.0f, 0.0f, 0.0f);
    }
  }

  if (tid < num_mdt_measurements) {
    compute_straw_residuals_and_derivatives(line, measurement_cache,
                                            residual_cache);
  }

  if (should_compute_rpc<Overflow>(tid, num_mdt_measurements,
                                   num_rpc_measurements)) {
    compute_strip_residuals_and_derivatives(line, measurement_cache,
                                            residual_cache);
  }
}

__device__ real_t contract(const Vector3 &v1, const Vector3 &v2,
                           const Vector3 &inverse_sigma_squared) {
  real_t result = 0.0f;

#pragma unroll
  for (int i = 0; i < 3; i++) {
    result += v1[i] * v2[i] * inverse_sigma_squared[i];
  }

  return result;
}

// ================ Chi2 ================

__host__ real_t host_contract(const Vector3 &v1, const Vector3 &v2,
                              const Vector3 &inverse_sigma_squared) {
  real_t result = 0.0f;
  for (size_t i = 0; i < 3; i++) {
    result += v1[i] * v2[i] * inverse_sigma_squared[i];
  }
  return result;
}

__host__ real_t get_chi2(const std::vector<Vector3> &residuals,
                         const std::vector<Vector3> &inverse_sigma_squared) {
  real_t chi2 = 0.0f;
  for (size_t i = 0; i < residuals.size(); i++) {
    chi2 += host_contract(residuals[i], residuals[i], inverse_sigma_squared[i]);
  }

  return chi2;
}

// TODO: Refactor if this works
template <bool Overflow>
__device__ real_t get_chi2(cg::thread_block_tile<TILE_SIZE> &bucket_tile,
                           int num_measurements,
                           residual_cache_t<Overflow> &residual_cache) {

  real_t chi2 = 0.0f;

  real_t chi_val = contract(residual_cache.residual, residual_cache.residual,
                            residual_cache.inverse_sigma_squared);
  if constexpr (Overflow) {
    chi_val +=
        contract(residual_cache.rpc_residual, residual_cache.rpc_residual,
                 residual_cache.rpc_inverse_sigma_squared);
  }

  for (int i = bucket_tile.num_threads() / 2; i >= 1; i /= 2) {
    bucket_tile.sync();
    chi_val += bucket_tile.shfl_down(chi_val, i);
  }

  chi2 = chi_val;
  return chi2;
}

template <bool Overflow>
__device__ Vector4 get_gradient(cg::thread_block_tile<TILE_SIZE> &bucket_tile,
                                int num_measurements,
                                residual_cache_t<Overflow> &residual_cache) {
  Vector4 gradient = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll
  for (int i = 0; i < 4; i++) {
    gradient[i] =
        2 * contract(residual_cache.residual, residual_cache.delta_residual[i],
                     residual_cache.inverse_sigma_squared);

    if constexpr (Overflow) {
      gradient[i] += 2 * contract(residual_cache.rpc_residual,
                                  residual_cache.rpc_delta_residual[i],
                                  residual_cache.rpc_inverse_sigma_squared);
    }
  }

#pragma unroll
  for (int j = 0; j < 4; j++) {
    for (int i = bucket_tile.num_threads() / 2; i >= 1; i /= 2) {
      bucket_tile.sync();
      gradient[j] += bucket_tile.shfl_down(gradient[j], i);
    }
  }

  return gradient;
}

__device__ Vector3 get_delta_delta_residual(int param1_idx, int param2_idx,
                                            Vector3 *dd_residual) {
  if (param1_idx == Y0 || param2_idx == Y0 || param1_idx == X0 ||
      param2_idx == X0) {
    return Vector3(0.0f, 0.0f, 0.0f); // No delta delta residual for Y0 or X0
  } else if (param1_idx == THETA && param2_idx == THETA) {
    return dd_residual[DD_THETA_THETA];
  } else if (param1_idx == PHI && param2_idx == PHI) {
    return dd_residual[DD_PHI_PHI];
  } else {
    return dd_residual[DD_THETA_PHI]; // Mixed derivative
  }
}

template <bool Overflow>
__device__ Matrix4 get_hessian(cg::thread_block_tile<TILE_SIZE> &bucket_tile,
                               int num_measurements,
                               residual_cache_t<Overflow> &residual_cache) {
  Matrix4 hessian = Matrix4::Zero();

#pragma unroll
  for (int col = 0; col < 4; col++) {
#pragma unroll
    for (int row = 0; row < 4; row++) {
      if (row >= col) { // Lower triangular
        hessian(row, col) +=
            2 * (contract(residual_cache.delta_residual[row],
                          residual_cache.delta_residual[col],
                          residual_cache.inverse_sigma_squared) +
                 contract(get_delta_delta_residual(row, col,
                                                   residual_cache.dd_residual),
                          residual_cache.residual,
                          residual_cache.inverse_sigma_squared));

        if constexpr (Overflow) {
          hessian(row, col) +=
              2 * (contract(residual_cache.rpc_delta_residual[row],
                            residual_cache.rpc_delta_residual[col],
                            residual_cache.rpc_inverse_sigma_squared) +
                   contract(get_delta_delta_residual(
                                row, col, residual_cache.rpc_dd_residual),
                            residual_cache.rpc_residual,
                            residual_cache.rpc_inverse_sigma_squared));
        }
      }
    }
  }

  // Reduce hessian across the threads in the bucket_tile
  for (int i = bucket_tile.num_threads() / 2; i >= 1; i /= 2) {
    bucket_tile.sync();
#pragma unroll
    for (int col = 0; col < 4; col++) {
#pragma unroll
      for (int row = 0; row < 4; row++) {
        if (row >= col) { // Lower triangular
          hessian(row, col) += bucket_tile.shfl_down(hessian(row, col), i);
        }
      }
    }
  }

  // Broadcast the hessian to all threads:
#pragma unroll
  for (int col = 0; col < 4; col++) {
#pragma unroll
    for (int row = 0; row < 4; row++) {
      bucket_tile.sync();
      hessian(row, col) = bucket_tile.shfl(hessian(row, col), 0);
      hessian(col, row) = hessian(row, col); // Ensure symmetry
    }
  }

  return hessian;
}

// update_residual_cache instantiations
template __device__ void
update_residual_cache<true>(line_t &line, const int tid,
                            const int num_mdt_measurements,
                            const int num_rpc_measurements,
                            const measurement_cache_t<true> &measurement_cache,
                            residual_cache_t<true> &residual_cache);

template __device__ void update_residual_cache<false>(
    line_t &line, const int tid, const int num_mdt_measurements,
    const int num_rpc_measurements,
    const measurement_cache_t<false> &measurement_cache,
    residual_cache_t<false> &residual_cache);

// compute_residual instantiations
template __device__ void
compute_residual<true>(line_t &line, const int tid,
                       const int num_mdt_measurements,
                       const int num_rpc_measurements,
                       const measurement_cache_t<true> &measurement_cache,
                       residual_cache_t<true> &residual_cache);

template __device__ void
compute_residual<false>(line_t &line, const int tid,
                        const int num_mdt_measurements,
                        const int num_rpc_measurements,
                        const measurement_cache_t<false> &measurement_cache,
                        residual_cache_t<false> &residual_cache);

// get_chi2 instantiations
template __device__ real_t
get_chi2<true>(cg::thread_block_tile<TILE_SIZE> &bucket_tile,
               int num_measurements, residual_cache_t<true> &residual_cache);

template __device__ real_t
get_chi2<false>(cg::thread_block_tile<TILE_SIZE> &bucket_tile,
                int num_measurements, residual_cache_t<false> &residual_cache);

// get_gradient instantiations
template __device__ Vector4 get_gradient<true>(
    cg::thread_block_tile<TILE_SIZE> &bucket_tile, int num_measurements,
    residual_cache_t<true> &residual_cache);

template __device__ Vector4 get_gradient<false>(
    cg::thread_block_tile<TILE_SIZE> &bucket_tile, int num_measurements,
    residual_cache_t<false> &residual_cache);

// get_hessian instantiations
template __device__ Matrix4
get_hessian<true>(cg::thread_block_tile<TILE_SIZE> &bucket_tile,
                  int num_measurements, residual_cache_t<true> &residual_cache);

template __device__ Matrix4 get_hessian<false>(
    cg::thread_block_tile<TILE_SIZE> &bucket_tile, int num_measurements,
    residual_cache_t<false> &residual_cache);

} // namespace residualMath