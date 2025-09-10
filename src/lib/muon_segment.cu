#include <cmath>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include "config.h"
#include "data_structures.h"
#include "line_math.h"
#include "matrix_math.h"
#include "muon_segment.h"
#include "residual_math.h"
// #define VERBOSE 1
// #define DEBUG 1

#define NEWTON_RAPHSON_MAX_ITER 20

using namespace cooperative_groups;
namespace cg = cooperative_groups;

enum class WorkType { SEED_LINE, FIT_LINE };

// ============================================================================
// Constants
// ============================================================================

__constant__ real_t rho_0_comb[4] = {1, -1, 1, -1};
__constant__ real_t rho_1_comb[4] = {1, 1, -1, -1};

__constant__ float dS0_const[2][3] = {
    {1.0f, 0.0f, 0.0f}, // dS0/dx0
    {0.0f, 1.0f, 0.0f}  // dS0/dy0
};

// ============================================================================
// Utility Functions
// ============================================================================

__host__ __device__ void estimate_phi(struct Data *data, int rpc_start,
                                      int bucket_end, real_t &phi) {
  // Get RPC positions
  real_t rpc_0_x = data->sensor_pos_x[rpc_start];
  real_t rpc_0_y = data->sensor_pos_y[rpc_start];
  real_t rpc_1_x = data->sensor_pos_x[bucket_end - 1];
  real_t rpc_1_y = data->sensor_pos_y[bucket_end - 1];

  // Calculate phi from RPC positions
  real_t a = (rpc_1_x - rpc_0_x) / (rpc_1_y - rpc_0_y + EPSILON);
  phi = atan(1.0f / (a + EPSILON));
}

__device__ inline void estimate_theta(real_t T12_y, real_t T12_z, real_t rho_0,
                                      real_t rho_1, real_t phi, real_t &theta) {
  // Initial theta estimate
  theta = atan((T12_y - (rho_0 - rho_1)) / (T12_z * sin(phi)));

// Iterative refinement
#pragma unroll
  for (int j = 0; j < 10; j++) {
    theta = atan((T12_y - (rho_0 - rho_1) * sqrt(1 + tan(theta) * tan(theta) *
                                                         sin(phi) * sin(phi))) /
                 (T12_z * sin(phi)));
  }
}

__device__ void initialize_bucket_indexing(struct Data *data, int bucket_index,
                                           int &bucket_start, int &rpc_start,
                                           int &bucket_end,
                                           int &thread_data_index) {
  bucket_start = data->buckets[bucket_index * 3];
  rpc_start = data->buckets[bucket_index * 3 + 1];
  bucket_end = data->buckets[bucket_index * 3 + 2];

  thread_data_index = bucket_start + threadIdx.x % TILE_SIZE;
  if (thread_data_index >= bucket_end) {
    thread_data_index = bucket_end;
  }
}

// ============================================================================
// Work Implementation Functions
// ============================================================================

template <bool Overflow>
__device__ void
load_measurement_cache(struct Data *data, const int thread_data_index,
                       real_t x0, real_t y0, const int num_mdt_measurements,
                       measurement_cache_t<Overflow> &measurement_cache) {
  // clang-format off
    // Get sensor position
    // Initialize measurement cache
    measurement_cache.sensor_pos        = SENSOR_POS(data, thread_data_index);
    measurement_cache.drift_radius      = DRIFT_RADIUS(data, thread_data_index);
    measurement_cache.sensor_direction  = SENSOR_DIRECTION(data, thread_data_index);
    measurement_cache.plane_normal      = PLANE_NORMAL(data, thread_data_index);
  // clang-format on

  if constexpr (Overflow) {
    // clang-format off
    measurement_cache.strip_direction = SENSOR_DIRECTION(data, thread_data_index + num_mdt_measurements);
    measurement_cache.strip_pos       = SENSOR_POS(data, thread_data_index + num_mdt_measurements);
    measurement_cache.plane_normal      = PLANE_NORMAL(data, thread_data_index + num_mdt_measurements);
    // clang-format on
  }
}

__device__ Vector3 get_inverse_sigma_squared(const int index,
                                             struct Data *data) {
  Vector3 sigma = SIGMA(data, index);
  Vector3 sigma_squared = matrixMath::elementwise_mult(sigma, sigma);
#pragma unroll
  for (int i = 0; i < 3; i++) {
    if (sigma_squared[i] < EPSILON) {
      sigma_squared[i] = EPSILON; // Avoid division by zero
    }
  }
  return INVERSE_VECTOR3(sigma_squared);
}

template <bool Overflow>
__device__ void seed_line_impl(struct Data *data, const int thread_data_index,
                               int bucket_index, int bucket_start,
                               int rpc_start, int bucket_end,
                               cg::thread_block_tile<TILE_SIZE> &bucket_tile) {
  real_t phi = 0.0f;
  real_t theta[4], x0[4], y0[4];

  // ================ Estimate Line Parameters ================

  // Estimate phi (only thread 0)
  if (bucket_tile.thread_rank() == 0) {
    estimate_phi(data, rpc_start, bucket_end, phi);
  }
  bucket_tile.sync();
  phi = bucket_tile.shfl(phi, 0);

  // Estimate theta and line positions (first 4 threads)
  if (bucket_tile.thread_rank() < 4) {
    const int tid = bucket_tile.thread_rank();

    // Get sensor positions
    real_t T0_y = data->sensor_pos_y[bucket_start];
    real_t T0_z = data->sensor_pos_z[bucket_start];
    real_t T1_y = data->sensor_pos_y[rpc_start - 1];
    real_t T1_z = data->sensor_pos_z[rpc_start - 1];

    real_t T12_y = T0_y - T1_y;
    real_t T12_z = T0_z - T1_z;

    // Compute drift radii with combinatorics
    real_t rho_0 = data->drift_radius[bucket_start] * rho_0_comb[tid];
    real_t rho_1 = data->drift_radius[rpc_start - 1] * rho_1_comb[tid];

    // Get RPC reference point
    real_t rpc_0_x = data->sensor_pos_x[rpc_start];
    real_t rpc_0_y = data->sensor_pos_y[rpc_start];

    // Estimate theta for this combination
    estimate_theta(T12_y, T12_z, rho_0, rho_1, phi, theta[tid]);

    // Calculate line parameters
    real_t tan_theta = tan(theta[tid]);
    real_t sin_phi = sin(phi);
    real_t cos_phi = cos(phi);

    y0[tid] = T0_y - T0_z * tan_theta * sin_phi -
              rho_0 * sqrt(1 + tan_theta * tan_theta * sin_phi * sin_phi);

    x0[tid] = rpc_0_x - cos_phi / sin_phi * (rpc_0_y - y0[tid]);
  }

  // Broadcast all parameters to all threads
  for (int j = 0; j < 4; j++) {
    bucket_tile.sync();
    theta[j] = bucket_tile.shfl(theta[j], j);
    x0[j] = bucket_tile.shfl(x0[j], j);
    y0[j] = bucket_tile.shfl(y0[j], j);
  }

  // ================ Evaluate Line Quality ================
  const int num_mdt_measurements = rpc_start - bucket_start;
  const int num_rpc_measurements = bucket_end - rpc_start;

  // Evaluate each line candidate
  real_t chi2_arr[4];
  for (int j = 0; j < 4; j++) {
    measurement_cache_t<Overflow> measurement_cache;
    load_measurement_cache<Overflow>(data, thread_data_index, x0[j], y0[j],
                                     num_mdt_measurements, measurement_cache);

    residual_cache_t<Overflow> residual_cache;

    residual_cache.inverse_sigma_squared =
        get_inverse_sigma_squared(thread_data_index, data);

    // If there is an overflow we need to load the RPC data into one of the mdt
    // threads and do computations for both on that thread
    if constexpr (Overflow) {
      if (bucket_tile.thread_rank() < num_rpc_measurements) {
        residual_cache.rpc_inverse_sigma_squared = get_inverse_sigma_squared(
            rpc_start + bucket_tile.thread_rank(), data);
      }
    }

    line_t line;
    line.update_line(x0[j], y0[j], phi, theta[j]);

    residualMath::compute_residual<Overflow>(
        line, bucket_tile.thread_rank(), num_mdt_measurements,
        num_rpc_measurements, measurement_cache, residual_cache);

    chi2_arr[j] = residualMath::get_chi2<Overflow>(
        bucket_tile, num_mdt_measurements + num_rpc_measurements,
        residual_cache);
  }

  // Store best line parameters (thread 0 only)
  if (bucket_tile.thread_rank() == 0) {
    int best_index = 0;
    real_t min_residual = chi2_arr[0];

    for (int j = 1; j < 4; j++) {
      if (chi2_arr[j] < min_residual) {
        min_residual = chi2_arr[j];
        best_index = j;
      }
    }

    data->theta[bucket_index] = theta[best_index];
    data->phi[bucket_index] = phi;
    data->x0[bucket_index] = x0[best_index];
    data->y0[bucket_index] = y0[best_index];
  }
}

template <bool Overflow>
__device__ void fit_line_impl(struct Data *data, const int thread_data_index,
                              int bucket_index, int bucket_start, int rpc_start,
                              int bucket_end,
                              cg::thread_block_tile<TILE_SIZE> &bucket_tile) {

  // Initialize from seed parameters
  Vector4 params = {data->theta[bucket_index], data->phi[bucket_index],
                    data->x0[bucket_index], data->y0[bucket_index]};

  const int num_mdt_measurements = rpc_start - bucket_start;
  const int num_rpc_measurements = bucket_end - rpc_start;

  // Construct measurement cache
  measurement_cache_t<Overflow> measurement_cache;
  load_measurement_cache<Overflow>(data, thread_data_index, params[X0],
                                   params[Y0], num_mdt_measurements,
                                   measurement_cache);

  // Newton-Raphson iteration
  int iteration = 0;

  residual_cache_t<Overflow> residual_cache;
  residual_cache.inverse_sigma_squared =
      get_inverse_sigma_squared(thread_data_index, data);

  if constexpr (Overflow) {
    if (bucket_tile.thread_rank() < num_rpc_measurements) {
      residual_cache.rpc_inverse_sigma_squared = get_inverse_sigma_squared(
          rpc_start + bucket_tile.thread_rank(), data);
    }
  }

  line_t line;
  for (iteration = 0; iteration < NEWTON_RAPHSON_MAX_ITER; iteration++) {

    // Update line and compute derivatives
    line.update_line(params[X0], params[Y0], params[PHI], params[THETA]);
    line.update_derivatives(params[THETA], params[PHI]);

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

    if (residualMath::should_compute_rpc<Overflow>(bucket_tile.thread_rank(),
                                                   num_mdt_measurements,
                                                   num_rpc_measurements)) {
      residualMath::compute_strip_residuals_and_derivatives(
          line, measurement_cache, residual_cache);
    }

    line.swap_to_Dortho();

    if (bucket_tile.thread_rank() < num_mdt_measurements) {
      residualMath::compute_straw_residuals_and_derivatives<Overflow>(
          line, measurement_cache, residual_cache);
    }

    // Get gradient and hessian
    int num_measurements = num_mdt_measurements + num_rpc_measurements;
    Vector4 gradient = residualMath::get_gradient<Overflow>(
        bucket_tile, num_measurements, residual_cache);

    // if (threadIdx.x == 16) {
    //   // Debug print gradient
    // printf("Gradient: dtheta: %.6f, dphi: %.6f, dx0: %.6f, dy0: %.6f\n",
    //        gradient[THETA], gradient[PHI], gradient[X0], gradient[Y0]);
    // }

    Matrix4 hessian = residualMath::get_hessian<Overflow>(
        bucket_tile, num_measurements, residual_cache);

    // if (threadIdx.x == 16) {
    //   // Debug print hessian
    //   printf("Hessian:\n");
    //   for (int i = 0; i < 4; i++) {
    //     for (int j = 0; j < 4; j++) {
    //       printf("%.6f ", hessian(i, j));
    //     }
    //     printf("\n");
    //   }
    // } 


    // In place inversion of hessian
    if (!matrixMath::invert_4x4(bucket_tile, hessian)) {
      if (bucket_tile.thread_rank() == 16) {
        printf("Singular hessian at iteration %d\n", iteration);
        printf("Bucket index: %d\n", bucket_index);
      }
      iteration = NEWTON_RAPHSON_MAX_ITER; // Force exit
    }

    if (bucket_tile.thread_rank() == 0) {
      Vector4 delta_params = -hessian * gradient;
      float delta_params_norm = delta_params.norm();
      params += delta_params;

      // Early stopping criterion
      if (delta_params_norm < 1e-6) {
        iteration = NEWTON_RAPHSON_MAX_ITER; // Force exit
#ifdef VERBOSE
        printf("Converged in %d iterations. delta_param norm: %.15f\n",
               iteration, delta_params_norm);
#endif
      }
    }

    // Broadcast error flag
    bucket_tile.sync();
    iteration = bucket_tile.shfl(iteration, 0);

// Broadcast updated parameters
#pragma unroll
    for (int i = 0; i < 4; i++) {
      bucket_tile.sync();
      params[i] = bucket_tile.shfl(params[i], 0);
    }

  } // End of Newton-Raphson loop

  // Store final parameters (thread 0 only)
  if (bucket_tile.thread_rank() == 0) {
    data->theta[bucket_index] = params[THETA];
    data->phi[bucket_index] = params[PHI];
    data->x0[bucket_index] = params[X0];
    data->y0[bucket_index] = params[Y0];
  }

  bucket_tile.sync();
  return;
}

// ============================================================================
// Work Dispatcher
// ============================================================================

template <bool Overflow>
__device__ void execute_work(WorkType work_type, Data *data,
                             const int thread_data_index, int bucket_index,
                             int bucket_start, int rpc_start, int bucket_end,
                             cg::thread_block_tile<TILE_SIZE> &bucket_tile) {

  switch (work_type) {
  case WorkType::SEED_LINE:
    seed_line_impl<Overflow>(data, thread_data_index, bucket_index,
                             bucket_start, rpc_start, bucket_end, bucket_tile);
    break;
  case WorkType::FIT_LINE:
    fit_line_impl<Overflow>(data, thread_data_index, bucket_index, bucket_start,
                            rpc_start, bucket_end, bucket_tile);
    break;
  default:
    printf("Unknown work type\n");
    break;
  }
}

// ============================================================================
// Main Work Partitioning Function
// ============================================================================

__device__ void partition_and_execute_work(struct Data *data, int num_buckets,
                                           WorkType work_type) {
  const unsigned int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int primary_bucket = global_thread_id / TILE_SIZE;
  const int secondary_bucket =
      (threadIdx.x < TILE_SIZE) ? primary_bucket + 1 : primary_bucket - 1;

  // Early exit if primary bucket is out of bounds
  if (primary_bucket >= num_buckets) {
    return;
  }

  // Initialize bucket data for both buckets
  int bucket_starts[2], rpc_starts[2], bucket_ends[2], thread_data_indices[2];
  int bucket_indices[2] = {primary_bucket, secondary_bucket};

  for (int i = 0; i < 2; i++) {
    if (bucket_indices[i] < num_buckets) {
      initialize_bucket_indexing(data, bucket_indices[i], bucket_starts[i],
                                 rpc_starts[i], bucket_ends[i],
                                 thread_data_indices[i]);
    } else {
      bucket_starts[i] = bucket_ends[i] = -1; // Mark as invalid
    }
  }

  const bool has_overflow = (bucket_starts[0] != -1 &&
                             (bucket_ends[0] - bucket_starts[0]) > TILE_SIZE) ||
                            (bucket_starts[1] != -1 &&
                             (bucket_ends[1] - bucket_starts[1]) > TILE_SIZE);

  cg::thread_block_tile<TILE_SIZE> bucket_tile =
      cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());

  if (bucket_indices[0] >= num_buckets) {
    return; // Invalid bucket, exit early
  }

  if (has_overflow) {
    execute_work<true>(work_type, data, thread_data_indices[0],
                       bucket_indices[0], bucket_starts[0], rpc_starts[0],
                       bucket_ends[0], bucket_tile);
  } else {
    execute_work<false>(work_type, data, thread_data_indices[0],
                        bucket_indices[0], bucket_starts[0], rpc_starts[0],
                        bucket_ends[0], bucket_tile);
  }
}

// ============================================================================
// Kernel Entry Points
// ============================================================================

__global__ void seed_lines(struct Data *data, int num_buckets) {
  partition_and_execute_work(data, num_buckets, WorkType::SEED_LINE);
}

__global__ void fit_lines(struct Data *data, int num_buckets) {
  partition_and_execute_work(data, num_buckets, WorkType::FIT_LINE);
}