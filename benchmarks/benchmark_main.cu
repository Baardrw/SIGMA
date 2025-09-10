#define EIGEN_DONT_VECTORIZE
#define EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT
#include <Eigen/Core>
#include <cassert>
#include <chrono>
#include <clocale>
#include <cmath>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <math.h>
#include <numeric>
#include <string>
#include <vector>

#include "config.h"
#include "data_structures.h"
#include "muon_segment.h"

// Test utils - include the CSV loading functions from the test file
#include "../tests/common/test_common_includes.h"
#include "../tests/common/test_data_utils.h"

#define NUM_BUCKETS 1000 // Number of desired bins will be determined by this

// Forward declaration of the kernel we want to test
__global__ void seed_lines(struct Data *data, int num_buckets);
__global__ void fit_lines(struct Data *data, int num_buckets);

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

// Function to create round-robin duplications of loaded data
void create_round_robin_data(
    Data &h_data, const std::vector<long> &original_bucket_ids,
    const std::map<long, std::vector<MDTHit>> &mdt_bucket,
    const std::map<long, std::vector<RPCHit>> &rpc_bucket,
    const std::vector<BucketGroundTruth> &bucket_ground_truths,
    int desired_buckets) {

  int original_buckets = original_bucket_ids.size();
  if (original_buckets == 0) {
    std::cerr << "Error: No original buckets to duplicate" << std::endl;
    return;
  }

  std::cout << "Creating " << desired_buckets << " buckets from "
            << original_buckets
            << " original buckets using round-robin duplication" << std::endl;

  // Calculate total measurements needed for round-robin duplication
  int total_measurements = 0;
  for (int i = 0; i < desired_buckets; i++) {
    uint64_t original_bucket_id = original_bucket_ids[i % original_buckets];

    int mdt_count = 0;
    int rpc_count = 0;

    auto mdt_it = mdt_bucket.find(original_bucket_id);
    if (mdt_it != mdt_bucket.end()) {
      mdt_count = mdt_it->second.size();
    }

    auto rpc_it = rpc_bucket.find(original_bucket_id);
    if (rpc_it != rpc_bucket.end()) {
      rpc_count = rpc_it->second.size();
    }

    total_measurements += mdt_count + rpc_count;
  }

  std::cout << "Total measurements needed: " << total_measurements << std::endl;

  // Allocate host memory
  h_data.sensor_pos_x = new real_t[total_measurements];
  h_data.sensor_pos_y = new real_t[total_measurements];
  h_data.sensor_pos_z = new real_t[total_measurements];
  h_data.drift_radius = new real_t[total_measurements];
  h_data.sigma_x = new real_t[total_measurements];
  h_data.sigma_y = new real_t[total_measurements];
  h_data.sigma_z = new real_t[total_measurements];
  h_data.buckets = new int[desired_buckets * 3];
  h_data.theta = new real_t[desired_buckets];
  h_data.phi = new real_t[desired_buckets];
  h_data.x0 = new real_t[desired_buckets];
  h_data.y0 = new real_t[desired_buckets];
  h_data.plane_normal_x = new real_t[total_measurements];
  h_data.plane_normal_y = new real_t[total_measurements];
  h_data.plane_normal_z = new real_t[total_measurements];
  h_data.sensor_dir_x = new real_t[total_measurements];
  h_data.sensor_dir_y = new real_t[total_measurements];
  h_data.sensor_dir_z = new real_t[total_measurements];

  int measurement_idx = 0;
  int running_total = 0;

  // Fill data for each bucket using round-robin pattern
  for (int bucket_idx = 0; bucket_idx < desired_buckets; bucket_idx++) {
    uint64_t original_bucket_id =
        original_bucket_ids[bucket_idx % original_buckets];

    // Get the original bucket data
    std::vector<MDTHit> mdt_hits;
    std::vector<RPCHit> rpc_hits;

    auto mdt_it = mdt_bucket.find(original_bucket_id);
    if (mdt_it != mdt_bucket.end()) {
      mdt_hits = mdt_it->second;
    }

    auto rpc_it = rpc_bucket.find(original_bucket_id);
    if (rpc_it != rpc_bucket.end()) {
      rpc_hits = rpc_it->second;
    }

    // Set bucket boundaries
    h_data.buckets[bucket_idx * 3 + 0] = running_total; // bucket_start
    h_data.buckets[bucket_idx * 3 + 1] =
        running_total + mdt_hits.size(); // rpc_start
    h_data.buckets[bucket_idx * 3 + 2] =
        running_total + mdt_hits.size() + rpc_hits.size(); // bucket_end

    // Initialize output arrays
    h_data.theta[bucket_idx] = 0.0f;
    h_data.phi[bucket_idx] = 0.0f;
    h_data.x0[bucket_idx] = 0.0f;
    h_data.y0[bucket_idx] = 0.0f;

    // Fill MDT data
    for (const auto &hit : mdt_hits) {
      h_data.sensor_pos_x[measurement_idx] = hit.surface_pos_x;
      h_data.sensor_pos_y[measurement_idx] = hit.surface_pos_y;
      h_data.sensor_pos_z[measurement_idx] = hit.surface_pos_z;

      // Calculate drift radius as distance between tube position and hit
      // position
      real_t dy = hit.surface_pos_y - hit.poca_y;
      real_t dz = hit.surface_pos_z - hit.poca_z;
      h_data.drift_radius[measurement_idx] = std::sqrt(dy * dy + dz * dz);

      h_data.sigma_x[measurement_idx] =
          1 / (EPSILON + h_data.drift_radius[measurement_idx]);
      h_data.sigma_y[measurement_idx] = 1.0f;
      h_data.sigma_z[measurement_idx] = 1.0f;
      measurement_idx++;
    }

    // Fill RPC data
    for (const auto &hit : rpc_hits) {
      h_data.sensor_pos_x[measurement_idx] = hit.poca_x;
      h_data.sensor_pos_y[measurement_idx] = hit.poca_y;
      h_data.sensor_pos_z[measurement_idx] = hit.poca_z;

      h_data.drift_radius[measurement_idx] =
          0.0f; // RPC doesn't have drift radius
      h_data.sigma_x[measurement_idx] = 1 / sqrt(12);
      h_data.sigma_y[measurement_idx] = 1 / sqrt(12);
      h_data.sigma_z[measurement_idx] = 1.0f;

      // Generate plane normal and strip direction
      Vector3 plane_normal(0, 0, 1);
      Vector3 sensor_dir(0, 0, 0);
      h_data.sensor_dir_x[measurement_idx] = sensor_dir.x();
      h_data.sensor_dir_y[measurement_idx] = sensor_dir.y();
      h_data.sensor_dir_z[measurement_idx] = sensor_dir.z();
      h_data.plane_normal_x[measurement_idx] = plane_normal.x();
      h_data.plane_normal_y[measurement_idx] = plane_normal.y();
      h_data.plane_normal_z[measurement_idx] = plane_normal.z();

      measurement_idx++;
    }

    running_total += mdt_hits.size() + rpc_hits.size();

    if (bucket_idx < 10 || bucket_idx % (desired_buckets / 10) == 0) {
      std::cout << "Bucket " << bucket_idx << " (from original bucket "
                << (original_bucket_id & 0xFFFFFFFF) << ", volume "
                << (original_bucket_id >> 32) << "): " << mdt_hits.size()
                << " MDT + " << rpc_hits.size() << " RPC hits" << std::endl;
    }
  }

  std::cout << "Loaded " << measurement_idx << " total measurements into "
            << desired_buckets << " buckets" << std::endl;
}

bool benchmark_seed_line_with_round_robin_data() {
  std::cout << "Running benchmark_seed_line_with_round_robin_data..."
            << std::endl;

  // File paths - update these to your actual file paths
  std::string mdt_filename = "mdt_hits.csv";
  std::string rpc_filename = "rpc_hits.csv";

  // Load ALL available data first (events 0-100)
  int start_event = 0;
  int end_event = 100;
  int desired_buckets = NUM_BUCKETS; // Total number of buckets we want

  Data h_data;

  // Timing variables
  auto start_total = std::chrono::high_resolution_clock::now();
  auto start_load = std::chrono::high_resolution_clock::now();

  // Use the comprehensive data loading from the test file
  auto [bucket_ids, mdt_bucket, rpc_bucket, bucket_ground_truths] =
      load_and_organize_data(mdt_filename, rpc_filename, start_event,
                             end_event);

  auto end_load = std::chrono::high_resolution_clock::now();

  int original_buckets = bucket_ids.size();
  if (original_buckets == 0) {
    std::cerr << "Error: No buckets found in CSV data" << std::endl;
    return false;
  }

  std::cout << "Loaded " << original_buckets << " original buckets from events "
            << start_event << " to " << end_event << std::endl;

  // Create round-robin duplicated data
  auto start_setup = std::chrono::high_resolution_clock::now();
  // Create duplicated data structures
  std::vector<long> duplicated_bucket_ids;
  std::map<long, std::vector<MDTHit>> duplicated_mdt_bucket;
  std::map<long, std::vector<RPCHit>> duplicated_rpc_bucket;
  std::vector<BucketGroundTruth> duplicated_ground_truths;

  // Round-robin duplicate the loaded data
  for (int i = 0; i < desired_buckets; i++) {
    int original_idx = i % original_buckets;
    long original_bucket_id = bucket_ids[original_idx];

    // Create unique bucket ID
    long new_bucket_id =
        original_bucket_id + ((long)(i / original_buckets) << 40);
    duplicated_bucket_ids.push_back(new_bucket_id);

    // Copy original data with new bucket ID
    duplicated_mdt_bucket[new_bucket_id] = mdt_bucket[original_bucket_id];
    duplicated_rpc_bucket[new_bucket_id] = rpc_bucket[original_bucket_id];
  }

  // Now use the WORKING functions from your test code
  setup_h_data_buckets(h_data, duplicated_bucket_ids, duplicated_mdt_bucket,
                       duplicated_rpc_bucket);
  populate_h_data(h_data, duplicated_mdt_bucket, duplicated_rpc_bucket,
                  duplicated_bucket_ids, duplicated_ground_truths);

  auto end_setup = std::chrono::high_resolution_clock::now();

  // Calculate total measurements
  int total_measurements =
      h_data.buckets[(desired_buckets - 1) * 3 + 2]; // End of last bucket
  std::cout << "Total measurements to process: " << total_measurements
            << std::endl;
  std::cout << "Number of buckets: " << desired_buckets << std::endl;
  std::cout << "Average measurements per bucket: "
            << (float)total_measurements / desired_buckets << std::endl;
  std::cout << "Duplication factor: "
            << (float)desired_buckets / original_buckets << std::endl;

  // Copy to device with timing
  auto start_h2d = std::chrono::high_resolution_clock::now();
  Data d_data;
  Data *device_data_ptr =
      copy_to_device(h_data, d_data, total_measurements, desired_buckets);
  CUDA_CHECK(cudaDeviceSynchronize()); // Ensure copy is complete
  auto end_h2d = std::chrono::high_resolution_clock::now();

  // Create CUDA events for precise kernel timing
  cudaEvent_t kernel_start, kernel_end;
  CUDA_CHECK(cudaEventCreate(&kernel_start));
  CUDA_CHECK(cudaEventCreate(&kernel_end));

  // Launch kernel with timing
  const int warpSize = 32;
  const int cg_group_size =
      16; // Use 16 threads per warp for cooperative groups
  const int warps_per_block = 6;
  const int block_x = warpSize * warps_per_block;
  const int num_blocks =
      desired_buckets / (warps_per_block * (warpSize / cg_group_size)) +
      (desired_buckets % (warps_per_block * (warpSize / cg_group_size)) == 0
           ? 0
           : 1);

  std::cout << "Launching kernel with " << num_blocks << " blocks and "
            << block_x << " threads per block." << std::endl;
  const dim3 block_size(block_x, 1, 1);
  const dim3 grid_size(num_blocks, 1, 1);

  // Warm up run (optional - helps with consistent timing)
  seed_lines<<<grid_size, block_size>>>(device_data_ptr, desired_buckets);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());

  // Set up shared memory 
  // size_t shared_mem_size = block_x * sizeof(real_t
  fit_lines<<<num_blocks, block_size>>>(device_data_ptr, desired_buckets);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());
  // return true;
  // Timed runs
  const int num_runs = 10; // Run multiple times for better statistics
  std::vector<float> kernel_times;

  auto start_kernel_total = std::chrono::high_resolution_clock::now();

  for (int run = 0; run < num_runs; run++) {
    CUDA_CHECK(cudaEventRecord(kernel_start, 0));
    seed_lines<<<num_blocks, block_size>>>(device_data_ptr, desired_buckets);
    fit_lines<<<num_blocks, block_size>>>(device_data_ptr, desired_buckets);
    CUDA_CHECK(cudaEventRecord(kernel_end, 0));
    CUDA_CHECK(cudaEventSynchronize(kernel_end));

    float kernel_time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_time_ms, kernel_start, kernel_end));
    kernel_times.push_back(kernel_time_ms);
  }

  auto end_kernel_total = std::chrono::high_resolution_clock::now();

  CUDA_CHECK(cudaGetLastError());

  // Copy results back with timing
  auto start_d2h = std::chrono::high_resolution_clock::now();
  copy_results_to_host(h_data, d_data, desired_buckets);
  CUDA_CHECK(cudaDeviceSynchronize()); // Ensure copy is complete
  auto end_d2h = std::chrono::high_resolution_clock::now();

  auto end_total = std::chrono::high_resolution_clock::now();

  // Calculate timing statistics
  auto load_time =
      std::chrono::duration<double, std::milli>(end_load - start_load).count();
  auto setup_time =
      std::chrono::duration<double, std::milli>(end_setup - start_setup)
          .count();
  auto h2d_time =
      std::chrono::duration<double, std::milli>(end_h2d - start_h2d).count();
  auto d2h_time =
      std::chrono::duration<double, std::milli>(end_d2h - start_d2h).count();
  auto total_time =
      std::chrono::duration<double, std::milli>(end_total - start_total)
          .count();
  auto kernel_total_time = std::chrono::duration<double, std::milli>(
                               end_kernel_total - start_kernel_total)
                               .count();

  // Kernel timing statistics
  float min_kernel_time =
      *std::min_element(kernel_times.begin(), kernel_times.end());
  float max_kernel_time =
      *std::max_element(kernel_times.begin(), kernel_times.end());
  float avg_kernel_time =
      std::accumulate(kernel_times.begin(), kernel_times.end(), 0.0f) /
      kernel_times.size();

  // Calculate standard deviation
  float variance = 0.0f;
  for (float time : kernel_times) {
    variance += (time - avg_kernel_time) * (time - avg_kernel_time);
  }
  float std_dev = std::sqrt(variance / kernel_times.size());

  // Performance metrics
  double measurements_per_second =
      (total_measurements * num_runs) / (kernel_total_time / 1000.0);
  double buckets_per_second =
      (desired_buckets * num_runs) / (kernel_total_time / 1000.0);

  // Memory bandwidth estimation (rough)
  size_t estimated_bytes_transferred =
      total_measurements *
          (sizeof(float) * 6 + sizeof(int)) + // Input data per measurement
      desired_buckets * sizeof(float) * 4;    // Output data per bucket
  double memory_bandwidth_gb_s = (estimated_bytes_transferred * num_runs) /
                                 (kernel_total_time / 1000.0) /
                                 (1024 * 1024 * 1024);

  // Print detailed results
  std::cout << "\n=== BENCHMARK RESULTS (Round-Robin Duplication) ==="
            << std::endl;
  std::cout << std::fixed << std::setprecision(3);

  std::cout << "\n--- Data Summary ---" << std::endl;
  std::cout << "Original buckets loaded:     " << original_buckets << std::endl;
  std::cout << "Total buckets created:       " << desired_buckets << std::endl;
  std::cout << "Duplication factor:          "
            << (float)desired_buckets / original_buckets << "x" << std::endl;
  std::cout << "Total measurements:          " << total_measurements
            << std::endl;

  std::cout << "\n--- Timing Breakdown ---" << std::endl;
  std::cout << "Data loading time:           " << load_time << " ms"
            << std::endl;
  std::cout << "Round-robin setup time:      " << setup_time << " ms"
            << std::endl;
  std::cout << "Host to Device copy:         " << h2d_time << " ms"
            << std::endl;
  std::cout << "Device to Host copy:         " << d2h_time << " ms"
            << std::endl;
  std::cout << "Total time:                  " << total_time << " ms"
            << std::endl;

  std::cout << "\n--- Kernel Performance (average of " << num_runs
            << " runs) ---" << std::endl;
  std::cout << "Average kernel time:         " << avg_kernel_time << " ms"
            << std::endl;
  std::cout << "Min kernel time:             " << min_kernel_time << " ms"
            << std::endl;
  std::cout << "Max kernel time:             " << max_kernel_time << " ms"
            << std::endl;
  std::cout << "Standard deviation:          " << std_dev << " ms" << std::endl;
  std::cout << "Coefficient of variation:    "
            << (std_dev / avg_kernel_time * 100) << "%" << std::endl;

  std::cout << "\n--- Performance Metrics ---" << std::endl;
  std::cout << "Measurements per second:     " << std::scientific
            << measurements_per_second << std::endl;
  std::cout << "Buckets per second:          " << buckets_per_second
            << std::endl;
  std::cout << "Est. memory bandwidth:       " << std::fixed
            << memory_bandwidth_gb_s << " GB/s" << std::endl;

  // Cleanup
  CUDA_CHECK(cudaEventDestroy(kernel_start));
  CUDA_CHECK(cudaEventDestroy(kernel_end));
  cleanup_host(h_data);
  cleanup_device(d_data);

  return true; // Return true if no errors occurred
}

int main() {
  std::cout << "Starting GPU kernel benchmark with round-robin CSV data..."
            << std::endl;

  // Initialize CUDA
  CUDA_CHECK(cudaSetDevice(0));

  bool all_passed = true;

  // Run round-robin benchmark
  all_passed &= benchmark_seed_line_with_round_robin_data();

  if (all_passed) {
    std::cout << "Benchmark completed successfully!" << std::endl;
    return 0;
  } else {
    std::cout << "Benchmark failed!" << std::endl;
    return 1;
  }
}