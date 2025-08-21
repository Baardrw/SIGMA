#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <map>
#include <set>

#include "data_structures.h"
#include "test_common_includes.h"
#include "test_math_utils.h"

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
  h_data.sigma_x = new real_t[total_measurements];
  h_data.sigma_y = new real_t[total_measurements];
  h_data.sigma_z = new real_t[total_measurements];
  h_data.sensor_dir_x = new real_t[total_measurements];
  h_data.sensor_dir_y = new real_t[total_measurements];
  h_data.sensor_dir_z = new real_t[total_measurements];
  h_data.plane_normal_x = new real_t[total_measurements];
  h_data.plane_normal_y = new real_t[total_measurements];
  h_data.plane_normal_z = new real_t[total_measurements];

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

      h_data.sigma_x[measurement_idx] =
          1 / (EPSILON + h_data.drift_radius[measurement_idx]);
      h_data.sigma_y[measurement_idx] = 1.0f;
      h_data.sigma_z[measurement_idx] = 1.0f;
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
      h_data.sigma_x[measurement_idx] = 0.1f;
      h_data.sigma_y[measurement_idx] = 1.0f;
      h_data.sigma_z[measurement_idx] = 1.0f;

      // Generate plane normal and strip direction
      Vector3 plane_normal(0, 0, 1);
      Vector3 sensor_dir(0, 1, 0);
      h_data.sensor_dir_x[measurement_idx] = sensor_dir.x();
      h_data.sensor_dir_y[measurement_idx] = sensor_dir.y();
      h_data.sensor_dir_z[measurement_idx] = sensor_dir.z();
      h_data.plane_normal_x[measurement_idx] = plane_normal.x();
      h_data.plane_normal_y[measurement_idx] = plane_normal.y();
      h_data.plane_normal_z[measurement_idx] = plane_normal.z();

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
  CUDA_CHECK(cudaMalloc(&d_data.sigma_x, num_measurements * sizeof(real_t)));
  CUDA_CHECK(cudaMalloc(&d_data.sigma_y, num_measurements * sizeof(real_t)));
  CUDA_CHECK(cudaMalloc(&d_data.sigma_z, num_measurements * sizeof(real_t)));
  CUDA_CHECK(cudaMalloc(&d_data.buckets, num_buckets * 3 * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_data.theta, num_buckets * sizeof(real_t)));
  CUDA_CHECK(cudaMalloc(&d_data.phi, num_buckets * sizeof(real_t)));
  CUDA_CHECK(cudaMalloc(&d_data.x0, num_buckets * sizeof(real_t)));
  CUDA_CHECK(cudaMalloc(&d_data.y0, num_buckets * sizeof(real_t)));
  CUDA_CHECK(cudaMalloc(&d_data.plane_normal_x, num_measurements * sizeof(real_t)));
  CUDA_CHECK(cudaMalloc(&d_data.plane_normal_y, num_measurements * sizeof(real_t)));
  CUDA_CHECK(cudaMalloc(&d_data.plane_normal_z, num_measurements * sizeof(real_t)));
  CUDA_CHECK(cudaMalloc(&d_data.sensor_dir_x, num_measurements * sizeof(real_t)));
  CUDA_CHECK(cudaMalloc(&d_data.sensor_dir_y, num_measurements * sizeof(real_t)));
  CUDA_CHECK(cudaMalloc(&d_data.sensor_dir_z, num_measurements * sizeof(real_t)));

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
  CUDA_CHECK(cudaMemcpy(d_data.sigma_x, h_data.sigma_x,
                        num_measurements * sizeof(real_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_data.sigma_y, h_data.sigma_y,
                        num_measurements * sizeof(real_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_data.sigma_z, h_data.sigma_z,
                        num_measurements * sizeof(real_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_data.buckets, h_data.buckets,
                        num_buckets * 3 * sizeof(int), cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMemcpy(d_data.plane_normal_x, h_data.plane_normal_x,
                        num_measurements * sizeof(real_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_data.plane_normal_y, h_data.plane_normal_y,
                        num_measurements * sizeof(real_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_data.plane_normal_z, h_data.plane_normal_z,
                        num_measurements * sizeof(real_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_data.sensor_dir_x, h_data.sensor_dir_x,
                        num_measurements * sizeof(real_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_data.sensor_dir_y, h_data.sensor_dir_y,
                        num_measurements * sizeof(real_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_data.sensor_dir_z, h_data.sensor_dir_z,
                        num_measurements * sizeof(real_t),
                        cudaMemcpyHostToDevice));

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
  delete[] h_data.sigma_x;
  delete[] h_data.sigma_y;
  delete[] h_data.sigma_z;
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
  cudaFree(d_data.sigma_x);
  cudaFree(d_data.sigma_y);
  cudaFree(d_data.sigma_z);
  cudaFree(d_data.buckets);
  cudaFree(d_data.theta);
  cudaFree(d_data.phi);
  cudaFree(d_data.x0);
  cudaFree(d_data.y0);
}

// Helper function to load and organize all data
std::tuple<std::vector<long int>, std::map<long int, std::vector<MDTHit>>,
           std::map<long int, std::vector<RPCHit>>,
           std::vector<BucketGroundTruth>>
load_and_organize_data(const std::string &mdt_filename,
                       const std::string &rpc_filename, int start, int end) {

  std::vector<MDTHit> mdt_hits;
  std::vector<RPCHit> rpc_hits;

  // Load all data
  for (int i = start; i < end; i++) {
    read_mdt_csv(mdt_filename, i, mdt_hits);
    read_rpc_csv(rpc_filename, i, rpc_hits);
  }

  std::vector<long int> bucket_ids = get_bucket_ids(mdt_hits, rpc_hits);
  std::cout << "Found " << bucket_ids.size() << " unique buckets in the data."
            << std::endl;

  // Group hits by bucket_id
  std::map<long int, std::vector<MDTHit>> mdt_bucket;
  std::map<long int, std::vector<RPCHit>> rpc_bucket;

  for (const auto &hit : mdt_hits) {
    mdt_bucket[hit.bucket_id].push_back(hit);
  }
  for (const auto &hit : rpc_hits) {
    rpc_bucket[hit.bucket_id].push_back(hit);
  }

  // Filter buckets with insufficient hits
  filter_buckets(bucket_ids, mdt_bucket, rpc_bucket);

  std::vector<BucketGroundTruth> bucket_ground_truths;

  return {bucket_ids, mdt_bucket, rpc_bucket, bucket_ground_truths};
}