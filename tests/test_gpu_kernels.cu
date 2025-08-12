#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>

#include "config.h"
#include "data_structures.h"
#include "muon_segment.h"

// Forward declaration of the kernel we want to test
__global__ void seed_lines(struct Data *data, int num_buckets);

// Helper function to check CUDA errors
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "    \
                << cudaGetErrorString(error) << std::endl;                     \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Test data generator
void generate_test_data(Data &h_data, int mdt_measurements_per_bucket,
                        int rpc_measurements_per_bucket, int num_buckets,
                        real_t true_phi, real_t true_theta, real_t true_x0,
                        real_t true_y0) {
  // Allocate host memory
  int num_measurements = mdt_measurements_per_bucket * num_buckets +
                         rpc_measurements_per_bucket * num_buckets;
  h_data.sensor_pos_x = new real_t[num_measurements];
  h_data.sensor_pos_y = new real_t[num_measurements];
  h_data.sensor_pos_z = new real_t[num_measurements];
  h_data.drift_radius = new real_t[num_measurements];
  h_data.sigma = new real_t[num_measurements];

  // Line direction
  real_t dx = sin(true_theta) * cos(true_phi);
  real_t dy = sin(true_theta) * sin(true_phi);
  real_t dz = cos(true_theta);

  // Generate points along the line with some noise
  for (int bucket = 0; bucket < num_buckets; bucket++) {
    for (int i = 0; i < mdt_measurements_per_bucket; i++) {
      int idx =
          bucket * (mdt_measurements_per_bucket + rpc_measurements_per_bucket) +
          i;
      real_t t = idx * 2.0f; // Parameter along line
      real_t sensor_to_hit_angle =
          idx % 2 == 0 ? 0.78539816f
                       : 0.78539816f + M_PI; // 45 degrees or opposite side
      Eigen::Vector3f hit(true_x0 + t * dx, true_y0 + t * dy, t * dz);

      real_t drift_radius =
          rand() % 10 + 1; // Random drift radius between 1 and 10
      real_t noise =
          (rand() % 100 - 50) / 1000.0f; // Random noise in [-0.05, 0.05]
      drift_radius += noise;
      h_data.sensor_pos_x[idx] = 10.0f; // Keep x constant for the sensors
      h_data.sensor_pos_y[idx] =
          drift_radius * sin(sensor_to_hit_angle) + hit.y();
      h_data.sensor_pos_z[idx] =
          drift_radius * cos(sensor_to_hit_angle) + hit.z();
      h_data.drift_radius[idx] = drift_radius;
      h_data.sigma[idx] = 0.1f; // Small sigma for all
    }

    for (int i = 0; i < rpc_measurements_per_bucket; i++) {
      int idx =
          bucket * (mdt_measurements_per_bucket + rpc_measurements_per_bucket) +
          mdt_measurements_per_bucket + i;

      real_t t = idx * 2.0f; // Parameter along line
      Eigen::Vector3f hit(true_x0 + t * dx, true_y0 + t * dy, t * dz);

      h_data.sensor_pos_x[idx] = hit.x();
      h_data.sensor_pos_y[idx] = hit.y();
      h_data.sensor_pos_z[idx] = hit.z();
    }
  }
}

void setup_buckets(Data &h_data, int num_buckets,
                   int mdt_measurements_per_bucket,
                   int rpc_measurements_per_bucket) {
  h_data.buckets = new int[num_buckets * 3];
  h_data.seed_theta = new real_t[num_buckets];
  h_data.seed_phi = new real_t[num_buckets];
  h_data.seed_x0 = new real_t[num_buckets];
  h_data.seed_y0 = new real_t[num_buckets];

  int measurements_per_bucket =
      mdt_measurements_per_bucket + rpc_measurements_per_bucket;

  // Set up bucket boundaries
  for (int i = 0; i < num_buckets; i++) {
    int bucket_start = i * measurements_per_bucket;
    int rpc_start = bucket_start + mdt_measurements_per_bucket;
    int bucket_end = (i + 1) * measurements_per_bucket;

    h_data.buckets[i * 3 + 0] = bucket_start; // bucket_start
    h_data.buckets[i * 3 + 1] = rpc_start;    // rpc_start
    h_data.buckets[i * 3 + 2] = bucket_end;   // bucket_end

    // Initialize output arrays
    h_data.seed_theta[i] = 0.0f;
    h_data.seed_phi[i] = 0.0f;
    h_data.seed_x0[i] = 0.0f;
    h_data.seed_y0[i] = 0.0f;
  }
}

Data copy_to_device(const Data &h_data, int num_measurements, int num_buckets) {
  Data d_data;

  // Allocate device memory
  CUDA_CHECK(
      cudaMalloc(&d_data.sensor_pos_x, num_measurements * sizeof(real_t)));
  CUDA_CHECK(
      cudaMalloc(&d_data.sensor_pos_y, num_measurements * sizeof(real_t)));
  CUDA_CHECK(
      cudaMalloc(&d_data.sensor_pos_z, num_measurements * sizeof(real_t)));
  CUDA_CHECK(
      cudaMalloc(&d_data.drift_radius, num_measurements * sizeof(real_t)));
  CUDA_CHECK(cudaMalloc(&d_data.sigma, num_measurements * sizeof(real_t)));
  CUDA_CHECK(cudaMalloc(&d_data.buckets, num_buckets * 3 * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_data.seed_theta, num_buckets * sizeof(real_t)));
  CUDA_CHECK(cudaMalloc(&d_data.seed_phi, num_buckets * sizeof(real_t)));
  CUDA_CHECK(cudaMalloc(&d_data.seed_x0, num_buckets * sizeof(real_t)));
  CUDA_CHECK(cudaMalloc(&d_data.seed_y0, num_buckets * sizeof(real_t)));

  // Copy data to device
  CUDA_CHECK(cudaMemcpy(d_data.sensor_pos_x, h_data.sensor_pos_x,
                        num_measurements * sizeof(real_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_data.sensor_pos_y, h_data.sensor_pos_y,
                        num_measurements * sizeof(real_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_data.sensor_pos_z, h_data.sensor_pos_z,
                        num_measurements * sizeof(real_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_data.drift_radius, h_data.drift_radius,
                        num_measurements * sizeof(real_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_data.sigma, h_data.sigma,
                        num_measurements * sizeof(real_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_data.buckets, h_data.buckets,
                        num_buckets * 3 * sizeof(int), cudaMemcpyHostToDevice));

  return d_data;
}

void copy_results_to_host(Data &h_data, const Data &d_data, int num_buckets) {
  CUDA_CHECK(cudaMemcpy(h_data.seed_theta, d_data.seed_theta,
                        num_buckets * sizeof(real_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_data.seed_phi, d_data.seed_phi,
                        num_buckets * sizeof(real_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_data.seed_x0, d_data.seed_x0,
                        num_buckets * sizeof(real_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_data.seed_y0, d_data.seed_y0,
                        num_buckets * sizeof(real_t), cudaMemcpyDeviceToHost));
}

void cleanup_host(Data &h_data) {
  delete[] h_data.sensor_pos_x;
  delete[] h_data.sensor_pos_y;
  delete[] h_data.sensor_pos_z;
  delete[] h_data.drift_radius;
  delete[] h_data.sigma;
  delete[] h_data.buckets;
  delete[] h_data.seed_theta;
  delete[] h_data.seed_phi;
  delete[] h_data.seed_x0;
  delete[] h_data.seed_y0;
}

void cleanup_device(Data &d_data) {
  cudaFree(d_data.sensor_pos_x);
  cudaFree(d_data.sensor_pos_y);
  cudaFree(d_data.sensor_pos_z);
  cudaFree(d_data.drift_radius);
  cudaFree(d_data.sigma);
  cudaFree(d_data.buckets);
  cudaFree(d_data.seed_theta);
  cudaFree(d_data.seed_phi);
  cudaFree(d_data.seed_x0);
  cudaFree(d_data.seed_y0);
}

bool test_seed_lines_basic() {
  std::cout << "Running test_seed_lines_basic..." << std::endl;

  const int num_buckets = 4;
  const int mdt_measurements_per_bucket = 16; // Must be < 32 (warpSize)
  const int rpc_measurements_per_bucket = 4;  // 4 RPC measurements per bucket
  const int total_measurements =
      num_buckets * (mdt_measurements_per_bucket + rpc_measurements_per_bucket);

  // Generate test data
  Data h_data;
  // Generate synthetic line data: a straight line in 3D space
  real_t true_phi = -1.29891977f;   // 30 degrees in radians
  real_t true_theta = -0.442532422f; // 60 degrees in radians
  real_t true_x0 = 786.488959f;
  real_t true_y0 = -10.3632995f;
  generate_test_data(h_data, mdt_measurements_per_bucket,
                     rpc_measurements_per_bucket, num_buckets, true_phi,
                     true_theta, true_x0, true_y0);
  setup_buckets(h_data, num_buckets, mdt_measurements_per_bucket,
                rpc_measurements_per_bucket);

  // Calculate expected phi:
  // real_t expected_phi =
  //     estimate_phi(h_data, total_measurements, measurements_per_bucket);

  // Copy to device
  Data d_data = copy_to_device(h_data, total_measurements, num_buckets);

  // Launch kernel
  const int block_size = 32;          // One warp
  const int num_blocks = num_buckets; // One block per bucket

  seed_lines<<<num_blocks, block_size>>>(&d_data, num_buckets);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());

  // Copy results back
  copy_results_to_host(h_data, d_data, num_buckets);

  // Validate results
  bool passed = true;
  for (int i = 0; i < num_buckets; i++) {
    std::cout << "Bucket " << i << ": "
              << "phi=" << h_data.seed_phi[i]
              << ", theta=" << h_data.seed_theta[i]
              << ", x0=" << h_data.seed_x0[i] << ", y0=" << h_data.seed_y0[i]
              << std::endl;

    // Check that values are within reasonable bounds of true values
    if (std::abs(h_data.seed_phi[i] - true_phi) > 0.2f) {
      std::cerr << "Error: phi mismatch in bucket " << i << std::endl;
      passed = false;
    }

    if (std::abs(h_data.seed_theta[i] - true_theta) > 0.2f) {
      std::cerr << "Error: theta mismatch in bucket " << i << std::endl;
      passed = false;
    }

    if (std::abs(h_data.seed_x0[i] - true_x0) > 0.5f) {
      std::cerr << "Error: x0 mismatch in bucket " << i << std::endl;
      passed = false;
    }

    if (std::abs(h_data.seed_y0[i] - true_y0) > 0.5f) {
      std::cerr << "Error: y0 mismatch in bucket " << i << std::endl;
      passed = false;
    }
  }

  // Cleanup
  cleanup_host(h_data);
  cleanup_device(d_data);

  return passed;
}


int main() {
  std::cout << "Starting GPU kernel tests..." << std::endl;

  // Initialize CUDA
  CUDA_CHECK(cudaSetDevice(0));

  bool all_passed = true;

  // Run tests
  all_passed &= test_seed_lines_basic();
  // all_passed &= test_seed_lines_single_bucket();

  if (all_passed) {
    std::cout << "All tests PASSED!" << std::endl;
    return 0;
  } else {
    std::cout << "Some tests FAILED!" << std::endl;
    return 1;
  }
}