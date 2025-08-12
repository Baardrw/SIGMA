#include "config.h"
#include "line_math.h"
#include "residual_math.h"

extern __constant__ real_t W_components[3];

namespace residualMath {

/**
 * Computes the residuals for all measurements in the bucket
 *
 * @returns the residual for each measurment in the bucket, STORED ON ALL
 * THREADS THREADS WITH INVALID MEASUREMENTS WILL RETURN 0.0
 */
__device__ real_t compute_residual(struct Data *data, line_t &line,
                                   int bucket_start, int rpc_start,
                                   int bucket_end) {
  unsigned int mdt_measurments = rpc_start - bucket_start;

  real_t yz_res = 0.0f;
  if (threadIdx.x < mdt_measurments) {
    const Vector3 T = {data->sensor_pos_x[bucket_start + threadIdx.x],
                       data->sensor_pos_y[bucket_start + threadIdx.x],
                       data->sensor_pos_z[bucket_start + threadIdx.x]};
    const Vector3 K = T - line.S0;
    const Vector3 W = {W_components[0], W_components[1], W_components[2]};
    const real_t drift_radius = data->drift_radius[bucket_start + threadIdx.x];

    yz_res = abs(K.cross(line.Dortho).dot(W)) - drift_radius;
  }

  return yz_res;
}

/**
 * Computes the chi2 for the line given the measurements in the bucket
 *
 * @returns Chi2 value for the line, ONLY ON THREAD 0
 */
__device__ real_t compute_chi2(struct Data *data, line_t &line,
                               int bucket_start, int rpc_start,
                               int bucket_end) {
  real_t chi2 = 0.0f;

  real_t inverse_sigma_squared = 0.0f;
  if (threadIdx.x < bucket_end - bucket_start) {
    inverse_sigma_squared += 1.0f / (data->sigma[bucket_start + threadIdx.x] *
                                     data->sigma[bucket_start + threadIdx.x]);
  }

  real_t residual =
      compute_residual(data, line, bucket_start, rpc_start, bucket_end);
  real_t chi_val = residual * residual * inverse_sigma_squared;

  for (int i = warpSize / 2; i >= 1; i /= 2) {
    chi_val += __shfl_down_sync(FULL_MASK, chi_val, i);
  }

  chi2 = chi_val;
  return chi2;
}
} // namespace residualMath