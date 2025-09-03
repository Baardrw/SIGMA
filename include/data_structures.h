#pragma once
#include <Eigen/Dense>

#include "config.h"

// Accessors
#define GET_THETA(line) (line.params[0])
#define GET_PHI(line) (line.params[1])
#define GET_X0(line) (line.params[2])
#define GET_Y0(line) (line.params[3])

#define MDT_DIR Vector3(1.0f, 0.0f, 0.0f) // Default wire direction

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
  real_t params[4]; // theta, phi, x0, y0

  Vector3 D;
  Vector3 dD[2];
  Vector3 ddD[3]; // theta theta, phi phi, theta phi

  Vector3 S0;
  Vector3 dS0[2] = {{1, 0, 0}, {0, 1, 0}}; // x0, y0

  Vector3 D_ortho;
  Vector3 dD_ortho[2];  // theta, phi
  Vector3 ddD_ortho[3]; // theta theta, phi phi, theta phi
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
  Vector3 _connection_vector;

  real_t *drift_radius; // MDT drift radius measurement

  real_t *sensor_pos_x;
  real_t *sensor_pos_y;
  real_t *sensor_pos_z;

  real_t *sensor_dir_x;
  real_t *sensor_dir_y;
  real_t *sensor_dir_z;

  real_t *plane_normal_x;
  real_t *plane_normal_y;
  real_t *plane_normal_z;

  __host__ __device__ Vector3 connection_vector() { return _connection_vector; }

  __host__ __device__ Vector3 sensor_direction() {
    return Vector3(*sensor_dir_x, *sensor_dir_y, *sensor_dir_z);
  }

  __host__ __device__ Vector3 sensor_pos() {
    return Vector3(*sensor_pos_x, *sensor_pos_y, *sensor_pos_z);
  }

  __host__ __device__ Vector3 plane_normal() {
    return Vector3(*plane_normal_x, *plane_normal_y, *plane_normal_z);
  }
};

template <> struct measurement_cache_t<true> {

  Vector3 _connection_vector;

  real_t *drift_radius; // MDT drift radius measurement

  real_t *sensor_pos_x;
  real_t *sensor_pos_y;
  real_t *sensor_pos_z;

  real_t *sensor_dir_x;
  real_t *sensor_dir_y;
  real_t *sensor_dir_z;

  real_t *plane_normal_x;
  real_t *plane_normal_y;
  real_t *plane_normal_z;

  real_t *strip_direction_x;
  real_t *strip_direction_y;
  real_t *strip_direction_z;

  real_t *strip_pos_x;
  real_t *strip_pos_y;
  real_t *strip_pos_z;

  __host__ __device__ Vector3 connection_vector() { return _connection_vector; }

  __host__ __device__ Vector3 sensor_direction() {
    return Vector3(*sensor_dir_x, *sensor_dir_y, *sensor_dir_z);
  }

  __host__ __device__ Vector3 sensor_pos() {
    return Vector3(*sensor_pos_x, *sensor_pos_y, *sensor_pos_z);
  }

  __host__ __device__ Vector3 strip_direction() {
    return Vector3(*strip_direction_x, *strip_direction_y, *strip_direction_z);
  }

  __host__ __device__ Vector3 strip_pos() {
    return Vector3(*strip_pos_x, *strip_pos_y, *strip_pos_z);
  }

  __host__ __device__ Vector3 plane_normal() {
    return Vector3(*plane_normal_x, *plane_normal_y, *plane_normal_z);
  }
};