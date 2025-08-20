#pragma once
#include "config.h"

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
