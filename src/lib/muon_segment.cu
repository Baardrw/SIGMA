#include <cmath>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include "config.h"
#include "data_structures.h"
#include "line_math.h"
#include "muon_segment.h"
#include "residual_math.h"

using namespace cooperative_groups;
namespace cg = cooperative_groups;

// Constant detirmining the combinatorics for estimating theta
__constant__ real_t rho_0_comb[4] = {1, -1, 1, -1};
__constant__ real_t rho_1_comb[4] = {1, 1, -1, -1};
__constant__ real_t W_components[3] = {1.0f, 0.0f, 0.0f};

__host__ __device__ inline void estimate_phi(struct Data *data, int rpc_start,
                                             int bucket_end, real_t &phi) {
  // Seed phi
  real_t rpc_0_x = data->sensor_pos_x[rpc_start];
  real_t rpc_0_y = data->sensor_pos_y[rpc_start];
  real_t rpc_1_x = data->sensor_pos_x[bucket_end - 1];
  real_t rpc_1_y = data->sensor_pos_y[bucket_end - 1];

  real_t gradient = (rpc_1_y - rpc_0_y) / (rpc_1_x - rpc_0_x + EPSILON);
  phi = atanf(gradient);
}

__device__ inline void estimate_theta(real_t T12_y, real_t T12_z, real_t rho_0,
                                      real_t rho_1, real_t phi, real_t &theta) {
  // estimate theta
  theta = atan((T12_y - (rho_0 - rho_1)) / (T12_z * sin(phi) + EPSILON));

// Iterativley refine theta
#pragma unroll
  for (int j = 0; j < 4; j++) {
    theta = atan((T12_y - (rho_0 - rho_1) * sqrt(1 + tan(theta) * tan(theta) *
                                                         sin(phi) * sin(phi))) /
                 (T12_z * sin(phi) + EPSILON));
  }
}

__device__ inline void
initialize_bucket_indexing(struct Data *data, int bucket_index,
                           int &bucket_start, int &rpc_start, int &bucket_end) {

  // Initialize bucket size
  bucket_start = *(data->buckets + bucket_index * 3);
  rpc_start = *(data->buckets + bucket_index * 3 + 1);
  bucket_end = *(data->buckets + bucket_index * 3 + 2);
  assert(bucket_end - bucket_start < MAX_MPB);
}

__global__ void seed_lines(struct Data *data, int num_buckets) {
  unsigned int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int bucket_index = global_thread_id / MAX_MPB;

  if (bucket_index >= num_buckets) {
    return;
  }

  cg::thread_block_tile<MAX_MPB> bucket_tile =
      cg::tiled_partition<MAX_MPB>(cg::this_thread_block());

  int bucket_start, rpc_start, bucket_end;
  initialize_bucket_indexing(data, bucket_index, bucket_start, rpc_start,
                             bucket_end);

  int thread_data_index = bucket_start + bucket_tile.thread_rank();
  if (thread_data_index >= bucket_end) {
    thread_data_index = bucket_end - 1; // Ensure valid index
  }
  
  // Line params
  real_t phi = 0.0f;
  real_t theta[4];
  real_t x0[4];
  real_t y0[4];

  // ================ Line params ================

  if (bucket_tile.thread_rank() == 0) {
    estimate_phi(data, rpc_start, bucket_end, phi);
  }

  bucket_tile.sync();
  phi = bucket_tile.shfl(phi, 0);

  if (bucket_tile.thread_rank() < 4) {
    real_t T0_y = data->sensor_pos_y[bucket_start];
    real_t T0_z = data->sensor_pos_z[bucket_start];
    real_t T1_y = data->sensor_pos_y[rpc_start - 1];
    real_t T1_z = data->sensor_pos_z[rpc_start - 1];

    real_t T12_y = T0_y - T1_y;
    real_t T12_z = T0_z - T1_z;

    real_t rho_0 = data->drift_radius[bucket_start] *
                   rho_0_comb[bucket_tile.thread_rank()];
    real_t rho_1 = data->drift_radius[rpc_start - 1] *
                   rho_1_comb[bucket_tile.thread_rank()];

    real_t rpc_0_x = data->sensor_pos_x[rpc_start];
    real_t rpc_0_y = data->sensor_pos_y[rpc_start];

    estimate_theta(T12_y, T12_z, rho_0, rho_1, phi,
                   theta[bucket_tile.thread_rank()]);

    y0[bucket_tile.thread_rank()] =
        (T0_y - T0_z * tan(theta[bucket_tile.thread_rank()]) * sin(phi) -
         rho_0 * sqrt(1 + tan(theta[bucket_tile.thread_rank()]) *
                              tan(theta[bucket_tile.thread_rank()]) * sin(phi) *
                              sin(phi)));

    x0[bucket_tile.thread_rank()] =
        rpc_0_x -
        cos(phi) / sin(phi) * (rpc_0_y - y0[bucket_tile.thread_rank()]);
  }

  for (int j = 0; j < 4; j++) {
    bucket_tile.sync();
    theta[j] = bucket_tile.shfl(theta[j], j);
    x0[j] = bucket_tile.shfl(x0[j], j);
    y0[j] = bucket_tile.shfl(y0[j], j);
  }

  // ================ Calculate Residuals ================

  // Constants used by residual calculations
  const Vector3 sensor_pos = SENSOR_POS(data, thread_data_index);
  const real_t drift_radius = DRIFT_RADIUS(data, thread_data_index);
  const real_t sigma = SIGMA(data, thread_data_index);

  const Vector3 W = {W_components[0], W_components[1], W_components[2]};
  const int num_mdt_measurements = rpc_start - bucket_start;
  const int num_rpc_measurements = bucket_end - rpc_start;
  const real_t inverse_sigma_squared = 1.0f / (sigma * sigma);

  real_t chi2_arr[4];
  for (int j = 0; j < 4; j++) {
    residual_cache_t residual_cache;
    line_t line;

    lineMath::create_line(x0[j], y0[j], phi, theta[j], line);
    lineMath::compute_D_ortho(line, W);

    Vector3 K = sensor_pos - line.S0;

    residualMath::compute_residual(
        line, K, W, drift_radius, bucket_tile.thread_rank(),
        num_mdt_measurements, num_rpc_measurements, residual_cache);

    real_t chi2_val = residualMath::get_chi2(bucket_tile, inverse_sigma_squared,
                                             residual_cache);
    chi2_arr[j] = chi2_val;
  }

  // Residuals are only valid for thread id 0, due to shuffle down sum
  if (bucket_tile.thread_rank() == 0) {
    // Store the best line parameters
    real_t min_residual = chi2_arr[0];
    int best_index = 0;
    for (int j = 1; j < 4; j++) {
      if (chi2_arr[j] < min_residual) {
        min_residual = chi2_arr[j];
        best_index = j;
      }
    }

    data->seed_theta[bucket_index] = theta[best_index];
    data->seed_phi[bucket_index] = phi;
    data->seed_x0[bucket_index] = x0[best_index];
    data->seed_y0[bucket_index] = y0[best_index];
  }

  return;
}

__global__ void fit_line(struct Data *data, int num_buckets) {
  // Initialize the thread block tile
  unsigned int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int bucket_index = global_thread_id / MAX_MPB;

  if (bucket_index >= num_buckets) {
    return;
  }

  cg::thread_block_tile<MAX_MPB> bucket_tile =
      cg::tiled_partition<MAX_MPB>(cg::this_thread_block());

  int bucket_start, rpc_start, bucket_end;
  initialize_bucket_indexing(data, bucket_index, bucket_start, rpc_start,
                             bucket_end);

  const int thread_data_index = bucket_start + bucket_tile.thread_rank();
  if (thread_data_index >= bucket_end) {
    return; // No data for this thread
  }

  // Initialize line parameters from seed
  line_t line;
  lineMath::create_line(
      data->seed_x0[bucket_index], data->seed_y0[bucket_index],
      data->seed_phi[bucket_index], data->seed_theta[bucket_index], line);

  // Initialize constants
  const Vector3 sensor_pos = SENSOR_POS(data, thread_data_index);
  const Vector3 W = {W_components[0], W_components[1], W_components[2]};
  const real_t drift_radius = DRIFT_RADIUS(data, thread_data_index);
  const int num_mdt_measurements = rpc_start - bucket_start;
  const int num_rpc_measurements = bucket_end - rpc_start;
  const real_t sigma = SIGMA(data, thread_data_index);
  const real_t inverse_sigma_squared = 1.0f / (sigma * sigma);

  residual_cache_t residual_cache;
  for (int j = 0; j < 1000; j++) {
    // Update derivatives
    lineMath::update_derivatives(line, W);
    Vector3 K = sensor_pos - line.S0;

    // ================ Compute Residuals and derivatives ================
    residualMath::compute_residual(
        line, K, W, drift_radius, bucket_tile.thread_rank(),
        num_mdt_measurements, num_rpc_measurements, residual_cache);

    residualMath::compute_delta_residuals(
        line, bucket_tile.thread_rank(), num_mdt_measurements,
        num_rpc_measurements, K, W, residual_cache);

    residualMath::compute_dd_residuals(
        line, bucket_tile.thread_rank(), num_mdt_measurements,
        num_rpc_measurements, K, W, residual_cache);

    // All residuals data is now stored in the residual_cache

    Vector4 gradient =
        residualMath::get_gradient(bucket_tile, data, line, residual_cache,
                                   bucket_start, rpc_start, bucket_end);

    // Matrix4 hessian = residualMath::get_hessian(
    //     bucket_tile, data, line, bucket_start, rpc_start, bucket_end);
  }
}