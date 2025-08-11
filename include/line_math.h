#pragma once
#include <cuda_runtime.h>

#include "config.h"
#include "data_structures.h"
#include "math_utils.h"

// Helper function to compute D_ortho components
inline __device__ __host__ void
compute_Dortho_components(const line_t &line, const real3_t &W, real3_t &Dw,
                          real_t &norm_Dw, real3_t &D_ortho);

// Helper function to compute dD_ortho components
inline __device__ __host__ real3_t get_delta_D_ortho(const line_t &line,
                                                     int param_idx,
                                                     const real3_t &W);

// Helper function to compute ddD_ortho components
inline __device__ __host__ real3_t get_dd_D_ortho(const line_t &line,
                                                  int param1_idx,
                                                  int param2_idx,
                                                  const real3_t &W);
                                                
// Creates a line from the given parameters
inline __host__ __device__ void create_line(real_t x0, real_t y0, real_t phi,
                                            real_t theta, line_t &line);

inline __host__ __device__ void
update_derived_line_quantities(line_t &line, real3_t &sensor_direction);
