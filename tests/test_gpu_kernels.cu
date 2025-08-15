#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "config.h"
#include "data_structures.h"
#include "line_math.h"
#include "muon_segment.h"
#include "residual_math.h"

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

// Structure to hold CSV row data
struct MDTHit {
  real_t surface_pos_x, surface_pos_y, surface_pos_z;
  real_t hit_pos_x, hit_pos_y, hit_pos_z;
  real_t poca_x, poca_y, poca_z;
  real_t hit_dir_x, hit_dir_y, hit_dir_z;
  int event_id, volume_id;
  long int bucket_id; // Unique ID for the bucket this hit belongs to
};

struct RPCHit {
  real_t strip_pos_x, strip_pos_y, strip_pos_z;
  real_t strip_dir_x, strip_dir_y, strip_dir_z;
  real_t strip_normal_x, strip_normal_y, strip_normal_z;
  real_t hit_pos_x, hit_pos_y, hit_pos_z;
  real_t poca_x, poca_y, poca_z;
  int event_id, volume_id;
  long int bucket_id; // Unique ID for the bucket this hit belongs to
};

struct BucketGroundTruth {
  long int bucket_id;
  int bucket_index;

  real_t theta;
  real_t phi;
  real_t x0;
  real_t y0;
};

// Function to split CSV line
std::vector<std::string> split_csv_line(const std::string &line) {
  std::vector<std::string> result;
  std::stringstream ss(line);
  std::string cell;

  while (std::getline(ss, cell, ',')) {
    result.push_back(cell);
  }
  return result;
}

// Function to read MDT data from CSV
void read_mdt_csv(const std::string &filename, int target_event_id,
                  std::vector<MDTHit> &hits) {
  std::ifstream file(filename);
  std::string line;

  if (!file.is_open()) {
    std::cerr << "Error: Could not open MDT file: " << filename << std::endl;
  }
  // Skip header line
  std::getline(file, line);

  while (std::getline(file, line)) {
    std::vector<std::string> columns = split_csv_line(line);
    if (columns.size() < 14)
      continue;

    MDTHit hit;
    hit.surface_pos_x = std::stod(columns[0]);
    hit.surface_pos_y = std::stod(columns[1]);
    hit.surface_pos_z = std::stod(columns[2]);
    hit.hit_pos_x = std::stod(columns[3]);
    hit.hit_pos_y = std::stod(columns[4]);
    hit.hit_pos_z = std::stod(columns[5]);
    hit.poca_x = std::stod(columns[6]);
    hit.poca_y = std::stod(columns[7]);
    hit.poca_z = std::stod(columns[8]);
    hit.hit_dir_x = std::stod(columns[9]);
    hit.hit_dir_y = std::stod(columns[10]);
    hit.hit_dir_z = std::stod(columns[11]);
    hit.event_id = std::stoi(columns[12]);
    hit.volume_id = std::stoi(columns[13]);

    // Filter by event ID
    if (hit.event_id == target_event_id) {
      hits.push_back(hit);
    }
  }

  file.close();
}

// Function to read RPC data from CSV
void read_rpc_csv(const std::string &filename, int target_event_id,
                  std::vector<RPCHit> &hits) {
  std::ifstream file(filename);
  std::string line;

  if (!file.is_open()) {
    std::cerr << "Error: Could not open RPC file: " << filename << std::endl;
  }

  // Skip header line
  std::getline(file, line);

  while (std::getline(file, line)) {
    std::vector<std::string> columns = split_csv_line(line);
    if (columns.size() < 16)
      continue;

    RPCHit hit;
    hit.strip_pos_x = std::stof(columns[0]);
    hit.strip_pos_y = std::stof(columns[1]);
    hit.strip_pos_z = std::stof(columns[2]);
    hit.strip_dir_x = std::stof(columns[3]);
    hit.strip_dir_y = std::stof(columns[4]);
    hit.strip_dir_z = std::stof(columns[5]);
    hit.strip_normal_x = std::stof(columns[6]);
    hit.strip_normal_y = std::stof(columns[7]);
    hit.strip_normal_z = std::stof(columns[8]);
    hit.hit_pos_x = std::stof(columns[9]);
    hit.hit_pos_y = std::stof(columns[10]);
    hit.hit_pos_z = std::stof(columns[11]);
    hit.poca_x = std::stof(columns[12]);
    hit.poca_y = std::stof(columns[13]);
    hit.poca_z = std::stof(columns[14]);
    hit.event_id = std::stoi(columns[15]);
    hit.volume_id = std::stoi(columns[16]);

    // Filter by event ID
    if (hit.event_id == target_event_id) {
      hits.push_back(hit);
    }
  }

  file.close();
}

// Function to get unique volume IDs from hits
std::vector<long int> get_bucket_ids(std::vector<MDTHit> &mdt_hits,
                                     std::vector<RPCHit> &rpc_hits) {
  std::set<long int> bucket_ids;

  for (auto &hit : mdt_hits) {
    long int bucket_id =
        static_cast<long int>(hit.volume_id) << 32 | hit.event_id;
    hit.bucket_id = bucket_id; // Store bucket ID in hit
    bucket_ids.insert(bucket_id);
  }
  for (auto &hit : rpc_hits) {
    long int bucket_id =
        static_cast<long int>(hit.volume_id) << 32 | hit.event_id;
    hit.bucket_id = bucket_id; // Store bucket ID in hit
    bucket_ids.insert(bucket_id);
  }
  std::vector<long int> result(bucket_ids.begin(), bucket_ids.end());
  return result;
}

void calculate_bucket_answer(BucketGroundTruth &bucket_ground_truth,
                             const std::vector<MDTHit> &mdt_hits,
                             const std::vector<RPCHit> &rpc_hits) {

  std::vector<Eigen::Vector3d> real_hits;
  for (const auto &hit : mdt_hits) {
    real_hits.emplace_back(hit.poca_x, hit.poca_y, hit.poca_z);
  }
  // There is some faulty RPC data

  // for (const auto &hit : rpc_hits) {
  //   real_hits.emplace_back(hit.poca_x, hit.poca_y, hit.poca_z);
  // }

  // Calculate best fit line parameters
  if (real_hits.size() < 2) {
    bucket_ground_truth.theta = 0.0;
    bucket_ground_truth.phi = 0.0;
    bucket_ground_truth.x0 = 0.0;
    bucket_ground_truth.y0 = 0.0;
    return;
  }

  // Step 1: Calculate centroid
  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  for (const auto &point : real_hits) {
    centroid += point;
  }
  centroid /= static_cast<double>(real_hits.size());

  // Step 2: Center the points
  Eigen::MatrixXd centered_points(3, real_hits.size());
  for (size_t i = 0; i < real_hits.size(); ++i) {
    centered_points.col(i) = real_hits[i] - centroid;
  }

  // Step 3: Perform SVD
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      centered_points, Eigen::ComputeThinU | Eigen::ComputeThinV);

  // The first column of U gives the direction of maximum variance (best fit
  // line)
  Eigen::Vector3d direction = svd.matrixU().col(0);

  // Ensure consistent direction
  if (direction.x() < 0) {
    direction = -direction;
  }

  // Convert to spherical coordinates
  double dx = direction.x();
  double dy = direction.y();
  double dz = direction.z();

  bucket_ground_truth.theta = std::acos(dz / direction.norm());
  bucket_ground_truth.phi = std::atan2(dy, dx);

  // Calculate (x0, y0) - point on line at z=0
  if (std::abs(dz) > 1e-10) {
    double t = -centroid.z() / dz;
    Eigen::Vector3d point_at_z0 = centroid + t * direction;
    std::cout << "Point at z=0: " << point_at_z0.transpose() << std::endl;
    bucket_ground_truth.x0 = point_at_z0.x();
    bucket_ground_truth.y0 = point_at_z0.y();
  } else {
    bucket_ground_truth.x0 = centroid.x();
    bucket_ground_truth.y0 = centroid.y();
  }
}

void populate_h_data(Data &h_data,
                     std::map<long int, std::vector<MDTHit>> &mdt_bucket,
                     std::map<long int, std::vector<RPCHit>> &rpc_bucket,
                     const std::vector<long int> &bucket_ids,
                     std::vector<BucketGroundTruth> &bucket_ground_truths) {

  // Calculate total measurements needed
  int total_measurements = 0;
  for (long int bucket_id : bucket_ids) {
    total_measurements +=
        mdt_bucket.at(bucket_id).size() + rpc_bucket.at(bucket_id).size();
  }

  std::cout << "Total measurements needed: " << total_measurements << std::endl;

  // Allocate host memory
  h_data.sensor_pos_x = new real_t[total_measurements];
  h_data.sensor_pos_y = new real_t[total_measurements];
  h_data.sensor_pos_z = new real_t[total_measurements];
  h_data.drift_radius = new real_t[total_measurements];
  h_data.sigma = new real_t[total_measurements];

  int measurement_idx = 0;

  // Fill data for each bucket (volume)
  for (size_t bucket = 0; bucket < bucket_ids.size(); bucket++) {
    long int bucket_id = bucket_ids[bucket];
    const std::vector<MDTHit> &mdt_volume_hits = mdt_bucket.at(bucket_id);
    const std::vector<RPCHit> &rpc_volume_hits = rpc_bucket.at(bucket_id);

    // Calculate correct answer for bucket
    BucketGroundTruth bucket_ground_truth;
    calculate_bucket_answer(bucket_ground_truth, mdt_volume_hits,
                            rpc_volume_hits);
    bucket_ground_truth.bucket_id = bucket_id;
    bucket_ground_truth.bucket_index = bucket;
    bucket_ground_truths.push_back(bucket_ground_truth);

    // Fill MDT data
    for (size_t i = 0; i < mdt_volume_hits.size(); i++) {
      const auto &hit = mdt_volume_hits[i];

      // Sensor position comes from surface position
      h_data.sensor_pos_x[measurement_idx] = hit.surface_pos_x;
      h_data.sensor_pos_y[measurement_idx] = hit.surface_pos_y;
      h_data.sensor_pos_z[measurement_idx] = hit.surface_pos_z;

      // Calculate drift radius as distance between tube position and hit
      // position Using only Y and Z coordinates as mentioned in Python comment
      real_t dy = hit.surface_pos_y - hit.poca_y;
      real_t dz = hit.surface_pos_z - hit.poca_z;
      h_data.drift_radius[measurement_idx] = std::sqrt(dy * dy + dz * dz);

      h_data.sigma[measurement_idx] =
          1 / (EPSILON + h_data.drift_radius[measurement_idx]);
      measurement_idx++;
    }

    // Fill RPC data
    for (size_t i = 0; i < rpc_volume_hits.size(); i++) {
      const auto &hit = rpc_volume_hits[i];

      // For RPC, sensor position comes from hit position (POCA)
      h_data.sensor_pos_x[measurement_idx] = hit.poca_x;
      h_data.sensor_pos_y[measurement_idx] = hit.poca_y;
      h_data.sensor_pos_z[measurement_idx] = hit.poca_z;

      // RPC doesn't have drift radius, set to 0
      h_data.drift_radius[measurement_idx] = 0.0f;
      h_data.sigma[measurement_idx] = 0.1f;
      measurement_idx++;
    }
  }

  std::cout << "Loaded " << measurement_idx << " total measurements"
            << std::endl;
}

void filter_buckets(std::vector<long int> &bucket_ids,
                    std::map<long int, std::vector<MDTHit>> &mdt_bucket,
                    std::map<long int, std::vector<RPCHit>> &rpc_bucket) {

  for (auto it = bucket_ids.begin(); it != bucket_ids.end();) {
    long int bucket_id = *it; // Store the bucket_id before potentially erasing

    if (mdt_bucket[bucket_id].size() < 3 || rpc_bucket[bucket_id].size() < 2) {
      // Remove bucket if it has less than 3 MDT or less than 2 RPC hits
      it = bucket_ids.erase(it); // erase returns the next valid iterator
    } else {
      ++it;
    }
  }
}

void setup_h_data_buckets(Data &h_data, std::vector<long int> &bucket_ids,
                          std::map<long int, std::vector<MDTHit>> &mdt_bucket,
                          std::map<long int, std::vector<RPCHit>> &rpc_bucket) {

  int num_buckets = bucket_ids.size();

  // Allocate bucket arrays
  h_data.buckets = new int[num_buckets * 3];
  h_data.theta = new real_t[num_buckets];
  h_data.phi = new real_t[num_buckets];
  h_data.x0 = new real_t[num_buckets];
  h_data.y0 = new real_t[num_buckets];

  int running_total = 0;

  // Set up bucket boundaries
  for (size_t i = 0; i < bucket_ids.size(); i++) {
    long int bucket_id = bucket_ids[i];
    int mdt_count = mdt_bucket[bucket_id].size();
    int rpc_count = rpc_bucket[bucket_id].size();

    h_data.buckets[i * 3 + 0] = running_total;             // bucket_start
    h_data.buckets[i * 3 + 1] = running_total + mdt_count; // rpc_start
    h_data.buckets[i * 3 + 2] =
        running_total + mdt_count + rpc_count; // bucket_end

    running_total += mdt_count + rpc_count;

    // Initialize output arrays
    h_data.theta[i] = 0.0f;
    h_data.phi[i] = 0.0f;
    h_data.x0[i] = 0.0f;
    h_data.y0[i] = 0.0f;
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
  CUDA_CHECK(cudaMalloc(&d_data.theta, num_buckets * sizeof(real_t)));
  CUDA_CHECK(cudaMalloc(&d_data.phi, num_buckets * sizeof(real_t)));
  CUDA_CHECK(cudaMalloc(&d_data.x0, num_buckets * sizeof(real_t)));
  CUDA_CHECK(cudaMalloc(&d_data.y0, num_buckets * sizeof(real_t)));

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
  CUDA_CHECK(cudaMemcpy(h_data.theta, d_data.theta,
                        num_buckets * sizeof(real_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_data.phi, d_data.phi, num_buckets * sizeof(real_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_data.x0, d_data.x0, num_buckets * sizeof(real_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_data.y0, d_data.y0, num_buckets * sizeof(real_t),
                        cudaMemcpyDeviceToHost));
}

void cleanup_host(Data &h_data) {
  delete[] h_data.sensor_pos_x;
  delete[] h_data.sensor_pos_y;
  delete[] h_data.sensor_pos_z;
  delete[] h_data.drift_radius;
  delete[] h_data.sigma;
  delete[] h_data.buckets;
  delete[] h_data.theta;
  delete[] h_data.phi;
  delete[] h_data.x0;
  delete[] h_data.y0;
}

void cleanup_device(Data &d_data) {
  cudaFree(d_data.sensor_pos_x);
  cudaFree(d_data.sensor_pos_y);
  cudaFree(d_data.sensor_pos_z);
  cudaFree(d_data.drift_radius);
  cudaFree(d_data.sigma);
  cudaFree(d_data.buckets);
  cudaFree(d_data.theta);
  cudaFree(d_data.phi);
  cudaFree(d_data.x0);
  cudaFree(d_data.y0);
}

std::vector<real_t> calculate_chi2(Data &h_data, int num_buckets) {

  std::vector<real_t> chi2_values(num_buckets, 0.0f);
  for (int i = 0; i < num_buckets; i++) {
    real_t x0 = h_data.x0[i];
    real_t y0 = h_data.y0[i];
    real_t phi = h_data.phi[i];
    real_t theta = h_data.theta[i];

    std::vector<real_t> residuals;
    std::vector<real_t> inverse_sigma_squared;

    // Get the start and end indices for this bucket
    int start_idx = h_data.buckets[i * 3];
    int rpc_idx = h_data.buckets[i * 3 + 1];
    int end_idx = h_data.buckets[i * 3 + 2];

    // Create line
    line_t line;
    lineMath::create_line(x0, y0, phi, theta, line);
    lineMath::compute_D_ortho(line, {1, 0, 0});

    // Compute K
    std::vector<Vector3> K;
    for (int j = start_idx; j < rpc_idx; j++) {
      K.emplace_back(h_data.sensor_pos_x[j], h_data.sensor_pos_y[j],
                     h_data.sensor_pos_z[j]);
    }

    // Get drift radius for this bucket
    std::vector<real_t> drift_radius;
    for (int j = start_idx; j < rpc_idx; j++) {
      drift_radius.emplace_back(h_data.drift_radius[j]);
    }

    // get inverse sigma squared
    for (int j = start_idx; j < end_idx; j++) {
      inverse_sigma_squared.emplace_back(1.0f /
                                         (h_data.sigma[j] * h_data.sigma[j]));
    }

    residuals = residualMath::compute_residuals(
        line, K, drift_radius, rpc_idx - start_idx, end_idx - rpc_idx);
    real_t chi2 = residualMath::get_chi2(residuals, inverse_sigma_squared);
    chi2_values[i] = chi2;
  }

  return chi2_values;
}

bool test_seed_lines_csv() {
  std::cout << "Running test_seed_lines_csv..." << std::endl;

  // File paths - update these to your actual file paths
  std::string mdt_filename = "/shared/src/SIGMA/mdt_hits.csv";
  std::string rpc_filename = "/shared/src/SIGMA/rpc_hits.csv";

  Data h_data;
  std::vector<long int> bucket_ids;

  // Load all data
  std::vector<MDTHit> mdt_hits;
  std::vector<RPCHit> rpc_hits;
  for (int i = 0; i < 100; i++) {
    read_mdt_csv(mdt_filename, i, mdt_hits);
    read_rpc_csv(rpc_filename, i, rpc_hits);
  }

  bucket_ids = get_bucket_ids(mdt_hits, rpc_hits);
  // Group hits by bucket_id
  std::map<long int, std::vector<MDTHit>> mdt_bucket;
  std::map<long int, std::vector<RPCHit>> rpc_bucket;
  std::vector<BucketGroundTruth> bucket_ground_truths;

  for (const auto &hit : mdt_hits) {
    mdt_bucket[hit.bucket_id].push_back(hit);
  }
  for (const auto &hit : rpc_hits) {
    rpc_bucket[hit.bucket_id].push_back(hit);
  }

  // for (const auto &bucket_id : bucket_ids) {
  //   std::cout << "Bucket ID: " << bucket_id << " has "
  //             << mdt_bucket[bucket_id].size() << " MDT hits and "
  //             << rpc_bucket[bucket_id].size() << " RPC hits." << std::endl;
  // }

  std::cout << "Found " << bucket_ids.size()
            << " unique buckets (volume IDs) in the data." << std::endl;

  // Remove bucekets with insufficient hits
  filter_buckets(bucket_ids, mdt_bucket, rpc_bucket);
  int num_buckets = bucket_ids.size();

  std::cout << "After filtering, " << num_buckets
            << " buckets remain with sufficient hits." << std::endl;

  if (num_buckets == 0) {
    std::cerr << "Error: No buckets found in CSV data" << std::endl;
    return false;
  }

  // Determine bucket structure
  setup_h_data_buckets(h_data, bucket_ids, mdt_bucket, rpc_bucket);
  populate_h_data(h_data, mdt_bucket, rpc_bucket, bucket_ids,
                  bucket_ground_truths);

  // Calculate total measurements
  int total_measurements =
      h_data.buckets[(num_buckets - 1) * 3 + 2]; // End of last bucket
  std::cout << "Total measurements to process: " << total_measurements
            << std::endl;

  // Copy to device
  Data d_data = copy_to_device(h_data, total_measurements, num_buckets);

  // Launch kernel
  const int block_size = 32;          // One warp
  const int num_blocks = num_buckets; // One block per bucket

  seed_lines<<<num_blocks, block_size>>>(&d_data, num_buckets);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());
  copy_results_to_host(h_data, d_data, num_buckets);
  std::vector<real_t> chi2_values_seed =
      calculate_chi2(h_data, num_buckets);

  fit_lines<<<num_blocks, block_size>>>(&d_data, num_buckets);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());
  copy_results_to_host(h_data, d_data, num_buckets);
  std::vector<real_t> chi2_values_fit =
      calculate_chi2(h_data, num_buckets);

  // Copy results back
  copy_results_to_host(h_data, d_data, num_buckets);

  std::vector<int> errors(num_buckets, 0);

  for (int i = 0; i < num_buckets; i++) {
    const auto &ground_truth = bucket_ground_truths[i];
    // Direction agnostic theta and phi
    real_t cos_theta = std::abs(std::cos(h_data.theta[i]));
    real_t cos_phi = std::abs(std::cos(h_data.phi[i]));
    real_t cos_ground_truth_theta = std::abs(std::cos(ground_truth.theta));
    real_t cos_ground_truth_phi = std::abs(std::cos(ground_truth.phi));

    if (std::abs(cos_theta - cos_ground_truth_theta) > 0.01 ||
        std::abs(cos_phi - cos_ground_truth_phi) > 0.01
        // std::abs(h_data.x0[i] - ground_truth.x0) > 0.1 ||
    ) {
      std::cout << "Angular error " << (ground_truth.bucket_id & 0xFFFFFFFF)
                << ", Volume " << (ground_truth.bucket_id >> 32) << ": "
                << "Delta Theta: " << cos_theta - cos_ground_truth_theta
                << ", Delta Phi: " << cos_phi - cos_ground_truth_phi
                << std::endl;
      errors[i] = 1; // Mark as error

    } else if (std::abs(h_data.y0[i] - ground_truth.y0) > 0.1) {
      std::cout << "Y0 error in bucket "
                << (ground_truth.bucket_id & 0xFFFFFFFF) << ", Volume "
                << (ground_truth.bucket_id >> 32) << ": "
                << "Calculated Y0: " << h_data.y0[i]
                << ", Ground Truth Y0: " << ground_truth.y0 << std::endl;
      errors[i] = 1; // Mark as error
    }

    else {
      errors[i] = 0; // No error
    }
  }

  int num_errors = std::accumulate(errors.begin(), errors.end(), 0);
  std::cout << "Error percentage: "
            << (static_cast<double>(num_errors) / num_buckets) * 100.0 << "%"
            << std::endl;
  std::cout << "Above errors are mostly due to upper lower ambiguities"
            << std::endl;


  bool fail = false;
  // Assert that the fit has smaller chi2 than the seed
  for (int i = 0; i < num_buckets; i++) {
    if (chi2_values_fit[i] > chi2_values_seed[i]) {
      std::cout << "Chi2 error in bucket "
                << (bucket_ground_truths[i].bucket_id & 0xFFFFFFFF)
                << ", Volume " << (bucket_ground_truths[i].bucket_id >> 32)
                << ": Seed Chi2: " << chi2_values_seed[i]
                << ", Fit Chi2: " << chi2_values_fit[i] << std::endl
                << "Fit Chi2 should be smaller than Seed Chi2!" << std::endl;
                fail = true;
    }
  }

  // Average fit improovement
  real_t average_fit_improvement = 0.0f;
  std::vector<real_t> fit_imp;
  for (int i = 0; i < num_buckets; i++) {
        (chi2_values_seed[i] / (chi2_values_fit[i])) ;
  }

  std::cout << "Average fit improvement: " << average_fit_improvement
            << std::endl;
  // Cleanup
  cleanup_host(h_data);
  cleanup_device(d_data);

  return fail;
}

int main() {
  std::cout << "Starting GPU kernel tests with CSV data..." << std::endl;

  // Initialize CUDA
  CUDA_CHECK(cudaSetDevice(0));

  bool all_passed = true;

  // Run CSV-based test
  all_passed &= test_seed_lines_csv();

  if (all_passed) {
    std::cout << "Test completed successfully!" << std::endl;
    return 0;
  } else {
    std::cout << "Test failed!" << std::endl;
    return 1;
  }
}