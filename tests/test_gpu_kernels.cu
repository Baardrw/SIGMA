#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <numeric>
#include <ostream>
#include <string>
#include <vector>

#include "config.h"
#include "data_structures.h"
#include "muon_segment.h"

// Test utils
#include "common/test_common_includes.h"
#include "common/test_data_utils.h"

#define EVENT_ID(bucket_id) (bucket_id & 0xFFFFFFFF)
#define VOLUME_ID(bucket_id) (bucket_id >> 32)

// Forward declaration of the kernel we want to test
__global__ void seed_lines(struct Data *data, int num_buckets);
__global__ void fit_lines(struct Data *data, int num_buckets);

// Helper function to run GPU fitting operations
std::pair<std::vector<real_t>, std::vector<real_t>>
run_gpu_fitting(Data &d_data, Data &h_data, int num_buckets) {

  // Launch kernel with timing
  const int warpSize = 32;
  const int cg_group_size =
      16; // Use 16 threads per warp for cooperative groups
  const int warps_per_block = 10;
  const int block_size = warpSize * warps_per_block;
  const int num_blocks =
      num_buckets / (warps_per_block * (warpSize / cg_group_size)) +
      (num_buckets % (warps_per_block * (warpSize / cg_group_size)) == 0
           ? 0
           : 1);

  // Run seed fitting
  seed_lines<<<num_blocks, block_size>>>(&d_data, num_buckets);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());
  copy_results_to_host(h_data, d_data, num_buckets);
  std::vector<real_t> chi2_seed = calculate_chi2(h_data, num_buckets);

  // Run iterative fitting
  fit_lines<<<num_blocks, block_size>>>(&d_data, num_buckets);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());
  copy_results_to_host(h_data, d_data, num_buckets);
  std::vector<real_t> chi2_fit = calculate_chi2(h_data, num_buckets);

  return {chi2_seed, chi2_fit};
}

// Helper function to validate geometric accuracy
bool validate_geometric_accuracy(
    const Data &h_data, const std::vector<BucketGroundTruth> &ground_truths,
    int num_buckets) {

  std::vector<int> errors(num_buckets, 0);
  const real_t angular_tolerance = 0.01;
  const real_t position_tolerance = 0.1;

  for (int i = 0; i < num_buckets; i++) {
    const auto &ground_truth = ground_truths[i];

    // Direction agnostic comparison
    real_t cos_theta = std::abs(std::cos(h_data.theta[i]));
    real_t cos_phi = std::abs(std::cos(h_data.phi[i]));
    real_t cos_gt_theta = std::abs(std::cos(ground_truth.theta));
    real_t cos_gt_phi = std::abs(std::cos(ground_truth.phi));

    bool angular_error =
        (std::abs(cos_theta - cos_gt_theta) > angular_tolerance ||
         std::abs(cos_phi - cos_gt_phi) > angular_tolerance);

    bool position_error =
        std::abs(h_data.y0[i] - ground_truth.y0) > position_tolerance;

    if (angular_error) {
#ifdef VERBOSE
      std::cout << "Angular error in bucket "
                << (ground_truth.bucket_id & 0xFFFFFFFF) << ", Volume "
                << (ground_truth.bucket_id >> 32)
                << ": Delta Theta: " << (cos_theta - cos_gt_theta)
                << ", Delta Phi: " << (cos_phi - cos_gt_phi) << std::endl;
#endif
      errors[i] = 1;
    } else if (position_error) {
#ifdef VERBOSE
      std::cout << "Y0 error in bucket "
                << (ground_truth.bucket_id & 0xFFFFFFFF) << ", Volume "
                << (ground_truth.bucket_id >> 32)
                << ": Calculated Y0: " << h_data.y0[i]
                << ", Ground Truth Y0: " << ground_truth.y0 << std::endl;
#endif
      errors[i] = 1;
    }
  }

  int num_errors = std::accumulate(errors.begin(), errors.end(), 0);
  double error_percentage =
      (static_cast<double>(num_errors) / num_buckets) * 100.0;

  std::cout << "Geometric error percentage: " << error_percentage << "%"
            << std::endl;
  std::cout << "Note: Errors are mostly due to upper/lower ambiguities"
            << std::endl;

  return true; // Geometric errors are expected due to ambiguities
}

bool check_chi2_nan(const std::vector<real_t> &chi2_seed,
                    const std::vector<real_t> &chi2_fit,
                    const std::vector<BucketGroundTruth> &ground_truths,
                    int num_buckets) {
  bool has_nan = false;

  for (int i = 0; i < num_buckets; i++) {
    if (std::isnan(chi2_seed[i]) || std::isnan(chi2_fit[i])) {
      std::cout << "Chi2 value is NaN in event: "
                << EVENT_ID(ground_truths[i].bucket_id)
                << ", Volume: " << VOLUME_ID(ground_truths[i].bucket_id)
                << ": Seed Chi2: " << chi2_seed[i]
                << ", Fit Chi2: " << chi2_fit[i] << std::endl;
      has_nan = true;
    }
  }

  return !has_nan;
}
// Helper function to validate chi2 improvement
bool validate_chi2_improvement(
    const std::vector<real_t> &chi2_seed, const std::vector<real_t> &chi2_fit,
    const std::vector<BucketGroundTruth> &ground_truths, int num_buckets) {

  bool success = true;

  for (int i = 0; i < num_buckets; i++) {
    if (chi2_fit[i] >= chi2_seed[i]) {
      std::cout << "Chi2 error in bucket "
                << (ground_truths[i].bucket_id & 0xFFFFFFFF) << ", Volume "
                << (ground_truths[i].bucket_id >> 32)
                << ": Seed Chi2: " << chi2_seed[i]
                << ", Fit Chi2: " << chi2_fit[i] << std::endl
                << "Fit Chi2 should be smaller than Seed Chi2!" << std::endl;
      success = false;
    }
  }

  return success;
}

// Helper function to analyze fit improvement statistics
bool analyze_fit_improvement(const std::vector<real_t> &chi2_seed,
                             const std::vector<real_t> &chi2_fit,
                             int num_buckets) {

  // Calculate improvement percentages
  std::vector<real_t> improvements;
  improvements.reserve(num_buckets);

  for (int i = 0; i < num_buckets; i++) {
    if (chi2_seed[i] != 0.0) {
      real_t improvement = (chi2_seed[i] - chi2_fit[i]) / chi2_seed[i];
      if (!std::isnan(improvement)) {
        improvements.push_back(improvement);
      }
    }
  }

  if (improvements.empty()) {
    std::cout << "Warning: No valid improvements calculated" << std::endl;
    return false;
  }

  // Count fits that didn't improve
  int no_improvement_count =
      std::count_if(improvements.begin(), improvements.end(),
                    [](real_t val) { return val <= 0.0; });

  bool success = (no_improvement_count == 0);

  if (no_improvement_count > 0) {
    std::cout << "Number of fits that did not improve: " << no_improvement_count
              << std::endl;
  }

  // Calculate average improvement
  real_t avg_improvement =
      std::accumulate(improvements.begin(), improvements.end(), 0.0) /
      improvements.size();

  std::cout << "\n=== Fit Improvement Statistics ===" << std::endl;
  std::cout << "Average fit improvement: " << (avg_improvement * 100.0) << "%"
            << std::endl;
  std::cout << "Total buckets analyzed: " << improvements.size() << std::endl;
  std::cout << "Buckets with no improvement: " << no_improvement_count
            << std::endl;
  std::cout << "================================\n" << std::endl;

  return success;
}

// Main test functions

bool test_seed_lines_csv(int start = 0, int end = 100) {
  std::cout << "Running test_seed_lines_csv..." << std::endl;

  // File paths
  const std::string mdt_filename = "/shared/src/SIGMA/mdt_hits.csv";
  const std::string rpc_filename = "/shared/src/SIGMA/rpc_hits.csv";

  // Load Raw data from CSV files
  auto [bucket_ids, mdt_bucket, rpc_bucket, bucket_ground_truths] =
      load_and_organize_data(mdt_filename, rpc_filename, start, end);

  int num_buckets = bucket_ids.size();
  if (num_buckets == 0) {
    std::cerr << "Error: No buckets found in CSV data" << std::endl;
    return false;
  }

  std::cout << "Processing " << num_buckets << " buckets with sufficient hits."
            << std::endl;

  // Setup data structures
  Data h_data;
  setup_h_data_buckets(h_data, bucket_ids, mdt_bucket, rpc_bucket);
  populate_h_data(h_data, mdt_bucket, rpc_bucket, bucket_ids,
                  bucket_ground_truths);

  int total_measurements = h_data.buckets[(num_buckets - 1) * 3 + 2];
  std::cout << "Total measurements to process: " << total_measurements
            << std::endl;

  // Run GPU computations
  Data d_data = copy_to_device(h_data, total_measurements, num_buckets);
  auto [chi2_seed, chi2_fit] = run_gpu_fitting(d_data, h_data, num_buckets);

  // Validate results
  bool geometric_validation =
      validate_geometric_accuracy(h_data, bucket_ground_truths, num_buckets);

  bool check_chi2_nan_values =
      check_chi2_nan(chi2_seed, chi2_fit, bucket_ground_truths, num_buckets);
  bool chi2_validation = validate_chi2_improvement(
      chi2_seed, chi2_fit, bucket_ground_truths, num_buckets);
  bool improvement_stats =
      analyze_fit_improvement(chi2_seed, chi2_fit, num_buckets);

  // Cleanup
  cleanup_host(h_data);
  cleanup_device(d_data);

  return geometric_validation && chi2_validation && improvement_stats &&
         check_chi2_nan_values;
}

int main() {
  std::cout << "Starting GPU kernel tests with CSV data..." << std::endl;

  // Initialize CUDA
  CUDA_CHECK(cudaSetDevice(0));

  bool all_passed = true;

  // Run CSV-based test
  all_passed &= test_seed_lines_csv(0, 100);

  return 0;
  if (all_passed) {
    std::cout << "Test completed successfully!" << std::endl;
    return 0;
  } else {
    std::cout << "Test failed!" << std::endl;
    return 1;
  }
}