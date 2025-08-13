#pragma once
#include <Eigen/Dense>

#include "config.h"

// Accessors
#define GET_THETA(line) (line.params[0])
#define GET_PHI(line) (line.params[1])
#define GET_X0(line) (line.params[2])
#define GET_Y0(line) (line.params[3])

#define EIGEN_VEC3(data, prefix, i)                                            \
  Vector3((data)->prefix##_x[i], (data)->prefix##_y[i],                \
                  (data)->prefix##_z[i])


#define SENSOR_POS(data, i) EIGEN_VEC3(data, sensor_pos, i)
#define HIT_POS(data, i) EIGEN_VEC3(data, hit_pos, i)
#define RPC_HIT_POS(data, i) EIGEN_VEC3(data, rpc_hit, i)
#define RPC_NORMAL(data, i) EIGEN_VEC3(data, rpc_normal, i)

struct Data {
  real_t *sensor_pos_x;
  real_t *sensor_pos_y;
  real_t *sensor_pos_z;

  real_t *plane_norm_x;
  real_t *plane_norm_y;
  real_t *plane_norm_z;

  real_t *sensor_dir_x;
  real_t *sensor_dir_y;
  real_t *sensor_dir_z;

  real_t *to_next_d_x;
  real_t *to_next_d_y;
  real_t *to_next_d_z;

  real_t *drift_radius;
  real_t *sigma; // Dirft radius uncertainty

  real_t *time;
  int *buckets; // bucket[bucket_index * 2] -> start of mdt data
                // bucket[bucket_index * 2 + 1] -> start of rpc data
                // There is a last imaginary bucket such that the end bucket
                // knows where to end The end bucket is alwayds one adress
                // beyond the last real bucket

  // Seeds:
  real_t *seed_x0;
  real_t *seed_y0;
  real_t *seed_phi;
  real_t *seed_theta;
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
  DD_THETA_PHI = 2
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