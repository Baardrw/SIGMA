#pragma once

#include "config.h"

// Accessors
#define GET_THETA(line) (line.params[0])
#define GET_PHI(line) (line.params[1])
#define GET_X0(line) (line.params[2])
#define GET_Y0(line) (line.params[3])

#define MDT_DIR Vector3(1.0f, 0.0f, 0.0f) // Default wire direction

#define VEC3(data, prefix, i)                                                  \
  Vector3((data)->prefix##_x[i], (data)->prefix##_y[i], (data)->prefix##_z[i])

#define SENSOR_POS(data, i) VEC3(data, sensor_pos, i)
#define PLANE_NORMAL(data, i) VEC3(data, plane_normal, i)
#define SENSOR_DIRECTION(data, i) VEC3(data, sensor_dir, i)

#define SIGMA(data, i) VEC3(data, sigma, i)
#define DRIFT_RADIUS(data, i) (data)->drift_radius[i]

struct Vector3 {
  union {
    real_t x, y, z;
    real_t v[3];
  };

  // Constructor for three components
  __host__ __device__ Vector3() = default;
  __host__ __device__ Vector3(real_t x, real_t y, real_t z) : v{x, y, z} {}

  // The << operator for convenient initialization
  __device__ __host__ Vector3 &operator<<(double val) {
    static int component = 0;
    switch (component % 3) {
    case 0:
      x = val;
      break;
    case 1:
      y = val;
      break;
    case 2:
      z = val;
      component = -1;
      break; // Reset after z
    }
    component++;
    return *this;
  }

  // Index operator
  __host__ __device__ real_t &operator[](int index) { return v[index]; }
  __host__ __device__ const real_t &operator[](int index) const {
    return v[index];
  }

  __host__ __device__ Vector3 operator-(const Vector3 &other) {
    return Vector3(v[0] - other.v[0], v[1] - other.v[1], v[2] - other.v[2]);
  }

  __host__ __device__ Vector3 operator-() {
    return Vector3(-v[0], -v[1], -v[2]);
  }

  __host__ __device__ Vector3 operator*(real_t scalar) {
    return Vector3(v[0] * scalar, v[1] * scalar, v[2] * scalar);
  }

  // Dot product
  __host__ __device__ real_t dot(const Vector3 &other) const {
    return v[0] * other.v[0] + v[1] * other.v[1] + v[2] * other.v[2];
  }

  __host__ __device__ real_t norm() const {
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  }

  __host__ __device__ Vector3 cross(const Vector3 &other) const {
    return Vector3(v[1] * other.v[2] - v[2] * other.v[1],
                   v[2] * other.v[0] - v[0] * other.v[2],
                   v[0] * other.v[1] - v[1] * other.v[0]);
  }
};

__host__ __device__ inline Vector3 operator*(real_t scalar, const Vector3 &v) {
  return Vector3(v.x * scalar, v.y * scalar, v.z * scalar);
}

__host__ __device__ inline Vector3 operator+(const Vector3 &a,
                                             const Vector3 &b) {
  return Vector3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline Vector3 operator-(const Vector3 &a,
                                             const Vector3 &b) {
  return Vector3(a.x - b.x, a.y - b.y, a.z - b.z);
}

struct Vector4 {
  union {
    real_t x, y, z, w;
    real_t v[4];
  };

  // Default constructor
  __host__ __device__ Vector4() = default;

  // Constructor for four components
  __host__ __device__ Vector4(real_t x, real_t y, real_t z, real_t w)
      : v{x, y, z, w} {}

  // Index operator
  __host__ __device__ real_t &operator[](int index) { return v[index]; }
  __host__ __device__ const real_t &operator[](int index) const {
    return v[index];
  }

  // Operator overloads to allow Vector3 arithmetic
  __host__ __device__ Vector4 operator+(const Vector4 &other) {
    return Vector4(v[0] + other.v[0], v[1] + other.v[1], v[2] + other.v[2],
                   v[3] + other.v[3]);
  }

  __host__ __device__ Vector4 operator+=(const Vector4 &other) {
    v[0] += other.v[0];
    v[1] += other.v[1];
    v[2] += other.v[2];
    v[3] += other.v[3];
    return *this;
  }

  __host__ __device__ Vector4 operator-(const Vector4 &other) {
    return Vector4(v[0] - other.v[0], v[1] - other.v[1], v[2] - other.v[2],
                   v[3] - other.v[3]);
  }

  __host__ __device__ real_t norm() {
    real_t norm = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]);
    return norm;
  }
};

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
  Vector3 connection_vector; // Vector from the z plane intersection to the
                             // sensor position
  real_t drift_radius;       // MDT drift radius measurement

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
  Vector3 connection_vector; // Vector from the z plane intersection to the
                             // sensor position
  real_t drift_radius;       // MDT drift radius measurement

  // Shared
  Vector3 sensor_direction; // MDT wire direction
  Vector3 sensor_pos; // Sensor position, for RPC this gives the hit position,
                      // for mdt it gives the tube center

  // RPC measurements
  Vector3 strip_direction;
  Vector3 strip_pos;
  Vector3 plane_normal; // RPC plane normal
};