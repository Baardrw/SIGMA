#include <cmath>
#include <cuda_runtime.h>

#include "config.h"
#include "data_structures.h"
#include "line_math.h"
#include "muon_segment.h"
#include "residual_math.h"

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

__global__ void seed_lines(struct Data *data, int num_buckets) {
  // Line params
  real_t phi = 0.0f;
  real_t theta[4];
  real_t x0[4];
  real_t y0[4];

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  int bucket_index = i / 32; // one warp per bucket
  int tid = i % 32;          // thread id in warp

  if (bucket_index >= num_buckets)
    return;

  // Initialize bucket size
  int bucket_start = *(data->buckets + bucket_index * 3);
  int rpc_start = *(data->buckets + bucket_index * 3 + 1);
  int bucket_end = *(data->buckets + bucket_index * 3 + 2);
  assert(bucket_end - bucket_start < warpSize);

  if (tid == 0) {
    estimate_phi(data, rpc_start, bucket_end, phi);
  }
  phi = __shfl_sync(FULL_MASK, phi, 0);

  if (tid < 4) {
    real_t T0_y = data->sensor_pos_y[bucket_start];
    real_t T0_z = data->sensor_pos_z[bucket_start];
    real_t T1_y = data->sensor_pos_y[rpc_start - 1];
    real_t T1_z = data->sensor_pos_z[rpc_start - 1];

    real_t T12_y = T0_y - T1_y;
    real_t T12_z = T0_z - T1_z;

    real_t rho_0 = data->drift_radius[bucket_start] * rho_0_comb[tid];
    real_t rho_1 = data->drift_radius[rpc_start - 1] * rho_1_comb[tid];

    real_t rpc_0_x = data->sensor_pos_x[rpc_start];
    real_t rpc_0_y = data->sensor_pos_y[rpc_start];

    estimate_theta(T12_y, T12_z, rho_0, rho_1, phi, theta[tid]);
    printf("Thread %d: theta = %f\n", tid, theta[tid]);

    y0[tid] = (T0_y - T0_z * tan(theta[tid]) * sin(phi) -
               rho_0 * sqrt(1 + tan(theta[tid]) * tan(theta[tid]) * sin(phi) *
                                    sin(phi)));

    x0[tid] = rpc_0_x - cos(phi) / sin(phi) * (rpc_0_y - y0[tid]);
  }

  for (int j = 0; j < 4; j++) {
    theta[j] = __shfl_sync(FULL_MASK, theta[j], j);
    x0[j] = __shfl_sync(FULL_MASK, x0[j], j);
    y0[j] = __shfl_sync(FULL_MASK, y0[j], j);
  }

  real_t chi2_arr[4];
  for (int j = 0; j < 4; j++) {
    line_t line;
    lineMath::create_line(x0[j], y0[j], phi, theta[j], line);
    lineMath::compute_Dortho(
        line, {W_components[0], W_components[1], W_components[2]});

    real_t chi2_val = residualMath::compute_chi2(data, line, bucket_start,
                                                 rpc_start, bucket_end);
    chi2_arr[j] = chi2_val;
  }

  // Residuals are only valid for thread id 0

  if (tid == 0) {
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