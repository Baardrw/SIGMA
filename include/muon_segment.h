#pragma once
#include <cuda_runtime.h>

#include "config.h"
#include "data_structures.h"

__global__ void seed_lines(struct Data, int num_buckets);

__host__ __device__ inline void estimate_phi(struct Data *data, int rpc_start,
                                    int bucket_end, real_t &phi);