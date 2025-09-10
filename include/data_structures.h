#pragma once

#define EIGEN_DONT_VECTORIZE
#define EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT
#include <Eigen/Core>
#include "config.h"

// Accessors
#define GET_THETA(line) (line.params[0])
#define GET_PHI(line) (line.params[1])
#define GET_X0(line) (line.params[2])
#define GET_Y0(line) (line.params[3])

#define EIGEN_VEC3(data, prefix, i)                                            \
  Vector3((data)->prefix##_x[i], (data)->prefix##_y[i], (data)->prefix##_z[i])

#define SENSOR_POS(data, i) EIGEN_VEC3(data, sensor_pos, i)
#define PLANE_NORMAL(data, i) EIGEN_VEC3(data, plane_normal, i)
#define SENSOR_DIRECTION(data, i) EIGEN_VEC3(data, sensor_dir, i)

#define SIGMA(data, i) EIGEN_VEC3(data, sigma, i)
#define DRIFT_RADIUS(data, i) (data)->drift_radius[i]

struct Data {
  real_t *sensor_pos_x;
  real_t *sensor_pos_y;
  real_t *sensor_pos_z;

  real_t *plane_normal_x;
  real_t *plane_normal_y;
  real_t *plane_normal_z;

  real_t *sensor_dir_x;
  real_t *sensor_dir_y;
  real_t *sensor_dir_z;

  real_t *to_next_d_x;
  real_t *to_next_d_y;
  real_t *to_next_d_z;

  real_t *drift_radius;
  real_t *sigma_x; // Sigma bending
  real_t *sigma_y; // Sigma non-bending
  real_t *sigma_z; // Sigma time

  real_t *time;
  int *buckets; // bucket[bucket_index * 2] -> start of mdt data
                // bucket[bucket_index * 2 + 1] -> start of rpc data
                // There is a last imaginary bucket such that the end bucket
                // knows where to end The end bucket is alwayds one adress
                // beyond the last real bucket

  // Seeds:
  real_t *x0;
  real_t *y0;
  real_t *phi;
  real_t *theta;

  // Fitted lines:
  real_t *fitted_x0;
  real_t *fitted_y0;
  real_t *fitted_phi;
  real_t *fitted_theta;
};

enum {
  // Parameters
  THETA = 0,
  PHI = 1,
  X0 = 2,
  Y0 = 3,

  D_THETA = 0,
  D_PHI = 1,
  D_X0 = 0,
  D_Y0 = 1,

  DD_THETA_THETA = 0,
  DD_PHI_PHI = 1,
  DD_THETA_PHI = 2,

  // Reisudal indexes
  BENDING = 0,
  NON_BENDING = 1,
  TIME = 2
};

typedef struct line {
  Vector3 D;
  Vector3 dD[2];
  Vector3 ddD[3]; // theta theta, phi phi, theta phi

  Vector3 S0;

  __host__ __device__ void update_line(real_t x0, real_t y0, real_t phi,
                                       real_t theta) {
    D << sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta);
    S0 << x0, y0, 0.0;
  }

  __host__ __device__ void update_derivatives(real_t theta, real_t phi) {
    dD[THETA] << cos(theta) * cos(phi), cos(theta) * sin(phi), -sin(theta);
    dD[PHI] << -sin(theta) * sin(phi), sin(theta) * cos(phi), 0.0;

    ddD[DD_THETA_THETA] << -sin(theta) * cos(phi), -sin(theta) * sin(phi),
        -cos(theta);
    ddD[DD_PHI_PHI] << sin(theta) * cos(phi), sin(theta) * sin(phi), 0.0;
    ddD[DD_THETA_PHI] << -cos(theta) * sin(phi), cos(theta) * cos(phi), 0.0;
  }

  __host__ __device__ Vector3 get_D_ortho() {
    Vector3 Dw = D;
    Dw[0] = 0;
    return (1 / Dw.norm()) * Dw;
  }

  __host__ __device__ void swap_to_Dortho() {

    // Compute D_ortho
    Vector3 Dw = D;
    Dw[0] = 0;
    real_t norm_Dw = Dw.norm();
    real_t D_dot_W = D[0]; // Required later so stored before D is overwritten
    D = (1 / norm_Dw) * Dw;

    // Compute dD_ortho
    real_t dD_dot_W[2]; // Required later so stored before dD is overwritten
    real_t norm_Dw_squared = norm_Dw * norm_Dw;
#pragma unroll
    for (int i = THETA; i <= PHI; i++) {
      dD_dot_W[i] = dD[i][0];

      dD[i] = (dD[i] - Vector3(dD_dot_W[i], 0, 0)) * (1 / norm_Dw);
      dD[i] -= (dD_dot_W[i] * D_dot_W) / norm_Dw_squared * D;
    }

// Compute ddD_ortho

// First compute terms that require the old ddD data
#pragma unroll
    for (int i = DD_THETA_THETA; i <= DD_THETA_PHI; i++) {
      real_t dd_D_dot_W = ddD[i][0];
      ddD[i] = (ddD[i] - Vector3(dd_D_dot_W, 0, 0)) * (1 / norm_Dw);
      ddD[i] += (dd_D_dot_W * D_dot_W) * (1 / norm_Dw_squared) * D;
    }

// Add the non cross terms
#pragma unroll
    for (int i = THETA; i <= PHI; i++) {
      ddD[i] += 2 * (D_dot_W * dD_dot_W[i]) / norm_Dw_squared * dD[i];
      ddD[i] += (dD_dot_W[i] * dD_dot_W[i]) / norm_Dw_squared * D;
    }

    // Add the cross terms
    ddD[DD_THETA_PHI] +=
        (D_dot_W * dD_dot_W[THETA]) / norm_Dw_squared * dD[PHI];
    ddD[DD_THETA_PHI] +=
        (D_dot_W * dD_dot_W[PHI]) / norm_Dw_squared * dD[THETA];
    ddD[DD_THETA_PHI] +=
        (dD_dot_W[THETA] * dD_dot_W[PHI]) / norm_Dw_squared * D;
  }

} line_t;

template <bool Overflow> struct residual_cache_t {
  Vector3 residual; // Residual for the measurement
  Vector3 delta_residual[4];
  Vector3 dd_residual[3]; // THETA_THETA, PHI_PHI, THETA_PHI
  Vector3 inverse_sigma_squared;
};

template <> struct residual_cache_t<true> {
  // Reisudla measurments for the non overflowed data
  Vector3 residual; // Residual for the measurement
  Vector3 delta_residual[4];
  Vector3 dd_residual[3]; // THETA_THETA, PHI_PHI, THETA_PHI
  Vector3 inverse_sigma_squared;

  // Residuals for the overflowed data
  Vector3 rpc_residual; // Residual for the measurement
  Vector3 rpc_delta_residual[4];
  Vector3 rpc_dd_residual[3]; // THETA_THETA, PHI_PHI, THETA_PHI
  Vector3 rpc_inverse_sigma_squared;
};

// In the general case (Overflow == false) there will be no overlap between rpc
// and mdt, so we can save memory by sharing the measurement cache
template <bool Overflow> struct measurement_cache_t {
  // MDT measurements
  real_t drift_radius; // MDT drift radius measurement

  // Shared
  Vector3 sensor_direction; // RPC sensor direction, or MDT wire direction
                            // (each thread is only assigned one type of
                            // measurement, so no conflicts will occour)
  Vector3 sensor_pos; // Sensor position, for RPC this gives the hit position,
                      // for mdt it gives the tube center

  // RPC measurements
  Vector3 plane_normal; // RPC plane normal
};

template <> struct measurement_cache_t<true> {
  // MDT measurements
  real_t drift_radius; // MDT drift radius measurement

  // Shared
  Vector3 sensor_direction; // MDT wire direction
  Vector3 sensor_pos; // Sensor position, for RPC this gives the hit position,
                      // for mdt it gives the tube center

  // RPC measurements
  Vector3 strip_direction;
  Vector3 strip_pos;
  Vector3 plane_normal; // RPC plane normal
};