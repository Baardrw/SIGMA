#include <Eigen/Dense>
#include <cassert>
#include <chrono>
#include <clocale>
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
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
#include "muon_segment.h"

#define DUPLICATIONS 10000

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
};

struct RPCHit {
  real_t strip_pos_x, strip_pos_y, strip_pos_z;
  real_t strip_dir_x, strip_dir_y, strip_dir_z;
  real_t strip_normal_x, strip_normal_y, strip_normal_z;
  real_t hit_pos_x, hit_pos_y, hit_pos_z;
  real_t poca_x, poca_y, poca_z;
  int event_id, volume_id;
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
std::vector<MDTHit> read_mdt_csv(const std::string &filename,
                                 int target_event_id) {
  std::vector<MDTHit> hits;
  std::ifstream file(filename);
  std::string line;

  if (!file.is_open()) {
    std::cerr << "Error: Could not open MDT file: " << filename << std::endl;
    return hits;
  }
  // Skip header line
  std::getline(file, line);

  while (std::getline(file, line)) {
    std::vector<std::string> columns = split_csv_line(line);
    if (columns.size() < 14)
      continue;

    MDTHit hit;
    hit.surface_pos_x = std::stof(columns[0]);
    hit.surface_pos_y = std::stof(columns[1]);
    hit.surface_pos_z = std::stof(columns[2]);
    hit.hit_pos_x = std::stof(columns[3]);
    hit.hit_pos_y = std::stof(columns[4]);
    hit.hit_pos_z = std::stof(columns[5]);
    hit.poca_x = std::stof(columns[6]);
    hit.poca_y = std::stof(columns[7]);
    hit.poca_z = std::stof(columns[8]);
    hit.hit_dir_x = std::stof(columns[9]);
    hit.hit_dir_y = std::stof(columns[10]);
    hit.hit_dir_z = std::stof(columns[11]);
    hit.event_id = std::stoi(columns[12]);
    hit.volume_id = std::stoi(columns[13]);

    // Filter by event ID
    if (hit.event_id == target_event_id) {
      hits.push_back(hit);
    }
  }

  file.close();
  std::cout << "Read " << hits.size() << " MDT hits for event "
            << target_event_id << std::endl;
  return hits;
}

// Function to read RPC data from CSV
std::vector<RPCHit> read_rpc_csv(const std::string &filename,
                                 int target_event_id) {
  std::vector<RPCHit> hits;
  std::ifstream file(filename);
  std::string line;

  if (!file.is_open()) {
    std::cerr << "Error: Could not open RPC file: " << filename << std::endl;
    return hits;
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
  std::cout << "Read " << hits.size() << " RPC hits for event "
            << target_event_id << std::endl;
  return hits;
}

// Function to get unique volume IDs from hits
std::vector<int> get_unique_volume_ids(const std::vector<MDTHit> &mdt_hits,
                                       const std::vector<RPCHit> &rpc_hits) {
  std::set<int> volume_set;

  for (const auto &hit : mdt_hits) {
    volume_set.insert(hit.volume_id);
  }
  for (const auto &hit : rpc_hits) {
    volume_set.insert(hit.volume_id);
  }

  std::vector<int> volume_ids(volume_set.begin(), volume_set.end());
  std::cout << "Found " << volume_ids.size() << " unique volumes: ";
  for (int vol : volume_ids) {
    std::cout << vol << " ";
  }
  std::cout << std::endl;

  return volume_ids;
}

std::vector<int> get_volumes_with_hits(const std::vector<int> &volume_ids,
                                       const std::vector<MDTHit> &mdt_hits,
                                       const std::vector<RPCHit> &rpc_hits) {
  std::set<int> volumes_with_hits;

  for (const auto &hit : mdt_hits) {
    if (std::find(volume_ids.begin(), volume_ids.end(), hit.volume_id) !=
        volume_ids.end()) {
      volumes_with_hits.insert(hit.volume_id);
    }
  }

  return std::vector<int>(volumes_with_hits.begin(), volumes_with_hits.end());
}

void load_csv_data(Data &h_data, const std::string &mdt_filename,
                   const std::string &rpc_filename, int event_id,
                   const std::vector<int> &volume_ids,
                   int duplications = 10000) {

  std::vector<MDTHit> mdt_hits = read_mdt_csv(mdt_filename, event_id);
  std::vector<RPCHit> rpc_hits = read_rpc_csv(rpc_filename, event_id);

  // Group hits by volume
  std::map<int, std::vector<MDTHit>> mdt_by_volume;
  std::map<int, std::vector<RPCHit>> rpc_by_volume;

  for (const auto &hit : mdt_hits) {
    mdt_by_volume[hit.volume_id].push_back(hit);
  }
  for (const auto &hit : rpc_hits) {
    rpc_by_volume[hit.volume_id].push_back(hit);
  }

  // Calculate total measurements needed
  int total_measurements = 0;
  for (int volume_id : volume_ids) {
    total_measurements +=
        mdt_by_volume[volume_id].size() + rpc_by_volume[volume_id].size();
  }

  std::cout << "Total measurements needed: " << total_measurements << std::endl;

  // Allocate host memory
  h_data.sensor_pos_x = new real_t[total_measurements * duplications];
  h_data.sensor_pos_y = new real_t[total_measurements * duplications];
  h_data.sensor_pos_z = new real_t[total_measurements * duplications];
  h_data.drift_radius = new real_t[total_measurements * duplications];
  h_data.sigma = new real_t[total_measurements * duplications];

  int measurement_idx = 0;

  // Fill data for each bucket (volume)
  for (size_t bucket = 0; bucket < volume_ids.size() * duplications; bucket++) {
    int volume_id = volume_ids[bucket % volume_ids.size()];
    const auto &mdt_volume_hits = mdt_by_volume[volume_id];
    const auto &rpc_volume_hits = rpc_by_volume[volume_id];

    // std::cout << "Bucket " << bucket << " (Volume " << volume_id
    //           << "): " << mdt_volume_hits.size() << " MDT hits, "
    //           << rpc_volume_hits.size() << " RPC hits" << std::endl;

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

      h_data.sigma[measurement_idx] = 0.1f; // Small sigma for all
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

void setup_buckets_from_csv(Data &h_data, const std::string &mdt_filename,
                            const std::string &rpc_filename, int event_id,
                            std::vector<int> &volume_ids,
                            int duplications = 10000) {

  // Read CSV data to determine bucket structure

  std::vector<MDTHit> mdt_hits = read_mdt_csv(mdt_filename, event_id);
  std::vector<RPCHit> rpc_hits = read_rpc_csv(rpc_filename, event_id);

  std::cout << "mdt hits: " << mdt_hits.size()
            << ", rpc hits: " << rpc_hits.size() << std::endl;

  // Get unique volume IDs
  volume_ids = get_unique_volume_ids(mdt_hits, rpc_hits);
  std::vector<int> volumes_with_hits =
      get_volumes_with_hits(volume_ids, mdt_hits, rpc_hits);
  volume_ids = volumes_with_hits;

  int num_buckets = volume_ids.size() * duplications;

  // Group hits by volume to determine bucket boundaries
  std::map<int, std::vector<MDTHit>> mdt_by_volume;
  std::map<int, std::vector<RPCHit>> rpc_by_volume;

  for (const auto &hit : mdt_hits) {
    mdt_by_volume[hit.volume_id].push_back(hit);
  }
  for (const auto &hit : rpc_hits) {
    rpc_by_volume[hit.volume_id].push_back(hit);
  }

  // Allocate bucket arrays
  h_data.buckets = new int[num_buckets * 3];
  h_data.theta = new real_t[num_buckets];
  h_data.phi = new real_t[num_buckets];
  h_data.x0 = new real_t[num_buckets];
  h_data.y0 = new real_t[num_buckets];

  int running_total = 0;

  // Set up bucket boundaries
  for (size_t i = 0; i < volume_ids.size() * duplications; i++) {
    int volume_id = volume_ids[i % volume_ids.size()];
    int mdt_count = mdt_by_volume[volume_id].size();
    int rpc_count = rpc_by_volume[volume_id].size();

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

    // std::cout << "Bucket " << i << " (Volume " << volume_id << "): "
    //           << "start=" << h_data.buckets[i * 3 + 0]
    //           << ", rpc_start=" << h_data.buckets[i * 3 + 1]
    //           << ", end=" << h_data.buckets[i * 3 + 2] << std::endl;
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
  CUDA_CHECK(cudaMemcpy(h_data.phi, d_data.phi,
                        num_buckets * sizeof(real_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_data.x0, d_data.x0,
                        num_buckets * sizeof(real_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_data.y0, d_data.y0,
                        num_buckets * sizeof(real_t), cudaMemcpyDeviceToHost));
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

bool benchmark_seed_line_with_duplicate_data() {
  std::cout << "Running benchmark_seed_line_with_duplicate_data..."
            << std::endl;

  // File paths - update these to your actual file paths
  std::string mdt_filename = "/shared/src/SIGMA/mdt_hits.csv";
  std::string rpc_filename = "/shared/src/SIGMA/rpc_hits.csv";
  int event_id = 16; // Choose which event to load

  Data h_data;
  std::vector<int> volume_ids;

  // Timing variables
  auto start_total = std::chrono::high_resolution_clock::now();
  auto start_setup = std::chrono::high_resolution_clock::now();

  // Set up buckets based on CSV data structure
  int duplications = DUPLICATIONS; // Number of duplications for each volume
  setup_buckets_from_csv(h_data, mdt_filename, rpc_filename, event_id,
                         volume_ids, duplications);

  auto end_setup = std::chrono::high_resolution_clock::now();

  int num_buckets = volume_ids.size() * duplications;

  if (num_buckets == 0) {
    std::cerr << "Error: No buckets found in CSV data" << std::endl;
    return false;
  }

  // Load actual CSV data
  auto start_load = std::chrono::high_resolution_clock::now();
  load_csv_data(h_data, mdt_filename, rpc_filename, event_id, volume_ids,
                duplications);
  auto end_load = std::chrono::high_resolution_clock::now();

  // Calculate total measurements
  int total_measurements =
      h_data.buckets[(num_buckets - 1) * 3 + 2]; // End of last bucket
  std::cout << "Total measurements to process: " << total_measurements
            << std::endl;
  std::cout << "Number of buckets: " << num_buckets << std::endl;
  std::cout << "Average measurements per bucket: "
            << (float)total_measurements / num_buckets << std::endl;

  // Copy to device with timing
  auto start_h2d = std::chrono::high_resolution_clock::now();
  Data d_data = copy_to_device(h_data, total_measurements, num_buckets);
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
  const int warps_per_block = 10;
  const int block_size = warpSize * warps_per_block;
  const int num_blocks =
      num_buckets / (warps_per_block * (warpSize / cg_group_size)) +
      (num_buckets % (warps_per_block * (warpSize / cg_group_size)) == 0 ? 0
                                                                         : 1);

  // Warm up run (optional - helps with consistent timing)
  seed_lines<<<num_blocks, block_size>>>(&d_data, num_buckets);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());
  fit_lines<<<num_blocks, block_size>>>(&d_data, num_buckets);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());

  // Timed runs
  const int num_runs = 10; // Run multiple times for better statistics
  std::vector<float> kernel_times;

  auto start_kernel_total = std::chrono::high_resolution_clock::now();

  for (int run = 0; run < num_runs; run++) {
    CUDA_CHECK(cudaEventRecord(kernel_start, 0));
    seed_lines<<<num_blocks, block_size>>>(&d_data, num_buckets);
    fit_lines<<<num_blocks, block_size>>>(&d_data, num_buckets);
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
  copy_results_to_host(h_data, d_data, num_buckets);
  CUDA_CHECK(cudaDeviceSynchronize()); // Ensure copy is complete
  auto end_d2h = std::chrono::high_resolution_clock::now();

  auto end_total = std::chrono::high_resolution_clock::now();

  // Calculate timing statistics
  auto setup_time =
      std::chrono::duration<double, std::milli>(end_setup - start_setup)
          .count();
  auto load_time =
      std::chrono::duration<double, std::milli>(end_load - start_load).count();
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
      (num_buckets * num_runs) / (kernel_total_time / 1000.0);

  // Memory bandwidth estimation (rough)
  size_t estimated_bytes_transferred =
      total_measurements *
          (sizeof(float) * 6 + sizeof(int)) + // Input data per measurement
      num_buckets * sizeof(float) * 4;        // Output data per bucket
  double memory_bandwidth_gb_s = (estimated_bytes_transferred * num_runs) /
                                 (kernel_total_time / 1000.0) /
                                 (1024 * 1024 * 1024);

  // Print detailed results
  std::cout << "\n=== BENCHMARK RESULTS ===" << std::endl;
  std::cout << std::fixed << std::setprecision(3);

  std::cout << "\n--- Timing Breakdown ---" << std::endl;
  std::cout << "Setup time:              " << setup_time << " ms" << std::endl;
  std::cout << "Data loading time:       " << load_time << " ms" << std::endl;
  std::cout << "Host to Device copy:     " << h2d_time << " ms" << std::endl;
  std::cout << "Device to Host copy:     " << d2h_time << " ms" << std::endl;
  std::cout << "Total time:              " << total_time << " ms" << std::endl;

  std::cout << "\n--- Kernel Performance (average of " << num_runs
            << " runs) ---" << std::endl;
  std::cout << "Average kernel time:     " << avg_kernel_time << " ms"
            << std::endl;
  std::cout << "Min kernel time:         " << min_kernel_time << " ms"
            << std::endl;
  std::cout << "Max kernel time:         " << max_kernel_time << " ms"
            << std::endl;
  std::cout << "Standard deviation:      " << std_dev << " ms" << std::endl;
  std::cout << "Coefficient of variation: " << (std_dev / avg_kernel_time * 100)
            << "%" << std::endl;

  std::cout << "\n--- Performance Metrics ---" << std::endl;
  std::cout << "Measurements per second: " << std::scientific
            << measurements_per_second << std::endl;
  std::cout << "Buckets per second:      " << buckets_per_second << std::endl;
  std::cout << "Est. memory bandwidth:   " << std::fixed
            << memory_bandwidth_gb_s << " GB/s" << std::endl;

  // Cleanup
  CUDA_CHECK(cudaEventDestroy(kernel_start));
  CUDA_CHECK(cudaEventDestroy(kernel_end));
  cleanup_host(h_data);
  cleanup_device(d_data);

  return true; // Return true if no errors occurred
}

int main() {
  std::cout << "Starting GPU kernel tests with CSV data..." << std::endl;

  // Initialize CUDA
  CUDA_CHECK(cudaSetDevice(0));

  bool all_passed = true;

  // Run CSV-based test
  all_passed &= benchmark_seed_line_with_duplicate_data();

  if (all_passed) {
    std::cout << "Test completed successfully!" << std::endl;
    return 0;
  } else {
    std::cout << "Test failed!" << std::endl;
    return 1;
  }
}