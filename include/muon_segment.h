#pragma once
#include <cuda_runtime.h>

#include "config.h"
#include "data_structures.h"

__global__ void seed_lines(struct Data, int num_buckets);