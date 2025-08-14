#include <cmath>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include "config.h"
#include "data_structures.h"
#include "line_math.h"
#include "matrix_math.h"
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
  // Get RPC positions
  real_t rpc_0_x = data->sensor_pos_x[rpc_start];
  real_t rpc_0_y = data->sensor_pos_y[rpc_start];
  real_t rpc_1_x = data->sensor_pos_x[bucket_end - 1];
  real_t rpc_1_y = data->sensor_pos_y[bucket_end - 1];

  // Calculate 'a' as in Python: a = (x1-x0)/(y1-y0)
  real_t a = (rpc_1_x - rpc_0_x) / (rpc_1_y - rpc_0_y + EPSILON);

  // Calculate phi as in Python: phi = atan(1/a)
  phi = atan(1.0f / (a + EPSILON));
}

__device__ inline void estimate_theta(real_t T12_y, real_t T12_z, real_t rho_0,
                                      real_t rho_1, real_t phi, real_t &theta) {
  // estimate theta
  theta = atan((T12_y - (rho_0 - rho_1)) / (T12_z * sin(phi)));

// Iterativley refine theta
#pragma unroll
  for (int j = 0; j < 10; j++) {
    theta = atan((T12_y - (rho_0 - rho_1) * sqrt(1 + tan(theta) * tan(theta) *
                                                         sin(phi) * sin(phi))) /
                 (T12_z * sin(phi)));
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

  if (bucket_start != 0) {
    return;
  }

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
    printf("Initial chi2 values: | %f |\n"
           "                      | %f |\n"
           "                      | %f |\n"
           "                      | %f |\n",
           chi2_arr[0], chi2_arr[1], chi2_arr[2], chi2_arr[3]);
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

  return;
}

__global__ void fit_lines(struct Data *data, int num_buckets) {
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

  if (bucket_start != 0) {
    return;
  }

  const int thread_data_index = bucket_start + bucket_tile.thread_rank();
  if (thread_data_index >= bucket_end) {
    return; // No data for this thread
  }

  // Initialize line parameters from seed
  line_t line;

  // Initialize constants
  const Vector3 sensor_pos = SENSOR_POS(data, thread_data_index);
  const Vector3 W = {W_components[0], W_components[1], W_components[2]};
  const real_t drift_radius = DRIFT_RADIUS(data, thread_data_index);
  const int num_mdt_measurements = rpc_start - bucket_start;
  const int num_rpc_measurements = bucket_end - rpc_start;
  const real_t sigma = SIGMA(data, thread_data_index);
  const real_t inverse_sigma_squared = 1.0f / (sigma * sigma);

  residual_cache_t residual_cache;
  Vector4 params = {data->theta[bucket_index], data->phi[bucket_index],
                    data->x0[bucket_index], data->y0[bucket_index]};

  if (bucket_tile.thread_rank() == 0) {
    printf("Initial parameters: | %f |\n"
           "                      | %f |\n"
           "                      | %f |\n"
           "                      | %f |\n",
           params[THETA], params[PHI], params[X0], params[Y0]);
  }
  for (int j = 0; j < 1; j++) {
    // Update line
    lineMath::create_line(params[X0], params[Y0], params[PHI], params[THETA],
                          line);
    lineMath::update_derivatives(line, W);
    Vector3 K = sensor_pos - line.S0;

    // ================ Compute Residuals and derivatives ================
    residualMath::compute_residual(
        line, K, W, drift_radius, bucket_tile.thread_rank(),
        num_mdt_measurements, num_rpc_measurements, residual_cache);

    printf("Residual %d: %lf\n", bucket_tile.thread_rank(),
           residual_cache.residual);

    residualMath::compute_delta_residuals(
        line, bucket_tile.thread_rank(), num_mdt_measurements,
        num_rpc_measurements, K, W, residual_cache);

    residualMath::compute_dd_residuals(
        line, bucket_tile.thread_rank(), num_mdt_measurements,
        num_rpc_measurements, K, W, residual_cache);

    // All residuals data is now stored in the residual_cache

    Vector4 gradient = residualMath::get_gradient(
        bucket_tile, inverse_sigma_squared, residual_cache);
    Matrix4 hessian = residualMath::get_hessian(
        bucket_tile, inverse_sigma_squared, residual_cache);

    // For testing turn system into a 2x2 system for theta and y0 to avoid using
    if (bucket_tile.thread_rank() == 0) {

      printf("Sensor position: | %f %f %f |\n", sensor_pos.x(), sensor_pos.y(),
             sensor_pos.z());
      printf("Line starting point: | %f %f %f |\n", line.S0.x(), line.S0.y(),
             line.S0.z());

      Eigen::Vector2f gradient_2x2;
      gradient_2x2[0] = gradient[THETA];
      gradient_2x2[1] = gradient[Y0];

      Eigen::Matrix2f hessian_2x2;
      hessian_2x2(0, 0) = hessian(THETA, THETA);
      hessian_2x2(0, 1) = hessian(THETA, Y0);
      hessian_2x2(1, 0) = hessian(Y0, THETA);
      hessian_2x2(1, 1) = hessian(Y0, Y0);

      Eigen::Matrix2f inverse_hessian;
      if (!matrixMath::invert_2x2(hessian_2x2, inverse_hessian)) {
        // If the matrix is singular, skip the update
        printf("Hessian: | %f %f |\n"
               "         | %f %f |\n",
               hessian_2x2(0, 0), hessian_2x2(0, 1), hessian_2x2(1, 0),
               hessian_2x2(1, 1));
        return;
      }

      Eigen::Vector2f delta_params = -inverse_hessian * gradient_2x2;
      // Update parameters
      params[THETA] += delta_params[0];
      params[Y0] += delta_params[1];

      if (delta_params.norm() < 1e-8) {
        // If the update is small enough, break the loop
        printf("Converged after %d iterations\n", j);
        break;
      }
      printf("delta_params: | %f |\n"
             "              | %f |\n",
             delta_params[0], delta_params[1]);
      printf("params: | %f |\n"
             "        | %f |\n",
             params[THETA], params[Y0]);
    }

    // Share params with all threads in the tile
    for (int i = 0; i < 4; i++) {
      bucket_tile.sync();
      params[i] = bucket_tile.shfl(params[i], 0);
    }
  }

  if (bucket_tile.thread_rank() == 0) {
    // Store the best line parameters
    data->theta[bucket_index] = params[THETA];
    data->phi[bucket_index] = params[PHI];
    data->x0[bucket_index] = params[X0];
    data->y0[bucket_index] = params[Y0];
  }
}