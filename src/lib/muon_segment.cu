#include "muon_segment.h"
#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>

// Constant detirmining the combinatorics for estimating theta
__constant__ real_t rho_0_comb[4] = {1, -1, 1, -1};
__constant__ real_t rho_1_comb[4] = {1, 1, -1, -1};

__device__ inline void estimate_phi(struct Data *data, int rpc_start,
                                    int bucket_end, real_t &phi) {
  // Seed phi
  real_t rpc_0_x = data->sensor_pos_x[rpc_start];
  real_t rpc_0_y = data->sensor_pos_y[rpc_start];
  real_t rpc_1_x = data->sensor_pos_x[bucket_end - 1];
  real_t rpc_1_y = data->sensor_pos_y[bucket_end - 1];

  real_t gradient = (rpc_1_y - rpc_0_y) / (rpc_1_x - rpc_0_x + 1e-6);
  phi = atan(gradient);
}

__device__ inline void estimate_theta(real_t T12_y, real_t T12_z, real_t rho_0,
                                      real_t rho_1, real_t phi, real_t &theta) {
  // Compute theta
  theta = atan(T12_y - (rho_0 - rho_1) / (T12_z * sin(phi) + 1e-6));

// Iterativley refine theta
#pragma unroll
  for (int j = 0; j < 4; j++) {
    theta = atan(T12_y -
                 (rho_0 - rho_1) *
                     sqrt(1 + tan(theta) * tan(theta) * sin(phi) * sin(phi)) /
                     (T12_z * sin(phi) + 1e-6));
  }
}

__global__ void seed_lines(struct Data *data, int num_buckets) {
  // Line params
  real_t phi;
  real_t theta;
  real_t x0;
  real_t y0;

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  int bucket_index = i / 32; // one warp per bucket
  int tid = i % 32;          // thread id in warp

  if (bucket_index >= num_buckets)
    return;

  // Initialize bucket size
  int bucket_start = *(data->buckets + bucket_index * 2);
  int rpc_start = *(data->buckets + bucket_index * 2 + 1);
  int bucket_end = *(data->buckets + bucket_index * 2 + 2);

  if (tid == 0) {
    estimate_phi(data, rpc_start, bucket_end, phi);
  }
  phi = __shfl_sync(0xFFFFFFFF, phi, 0);

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

    estimate_theta(T12_y, T12_z, rho_0, rho_1, phi, theta);

    y0 = (T0_y - T0_z * tan(theta) * sin(phi) -
          rho_0 * sqrt(1 + tan(theta) * tan(theta) * sin(phi) * sin(phi)));

    x0 = rpc_0_x - cos(phi) / sin(phi) * (rpc_0_y - y0);
  }

  ///////////////////////////////////////////////////////
  // At this point we split the warp into 4 groups of 8 threads to work out the combinatorics
  // of the 4 different rho combinations
  ///////////////////////////////////////////////////////

  for (int j = 0; j < 4; j++) {
    unsigned int mask = __ballot_sync(0xFFFFFFFF, (tid >= j * 8 && tid < (j+1) * 8));
    unsigned int source_thread = j;

    x0 = __shfl_sync(mask, x0, source_thread);
    y0 = __shfl_sync(mask, y0, source_thread);
    theta = __shfl_sync(mask, theta, source_thread);
  }

  
}