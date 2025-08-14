#pragma once
#include <cuda_runtime.h>

#include "config.h"
#include "data_structures.h"

__global__ void seed_lines(struct Data *data, int num_buckets);

__global__ void fit_lines(struct Data *data, int num_buckets);