#include "config.h"

// Accessors
#define GET_THETA(line) (line.params[0])
#define GET_PHI(line) (line.params[1])
#define GET_X0(line) (line.params[2])
#define GET_Y0(line) (line.params[3])

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
  real_t *time;
  int *buckets; // bucket[bucket_index * 2] -> start of mdt data
                // bucket[bucket_index * 2 + 1] -> start of rpc data
                // There is a last imaginary bucket such that the end bucket
                // knows where to end The end bucket is alwayds one adress
                // beyond the last real bucket
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

  real3_t D;
  real3_t dD[2];
  real3_t ddD[3]; // theta theta, phi phi, theta phi

  real3_t S0;
  real3_t dS0[2] = {{1, 0, 0}, {0, 1, 0}}; // x0, y0

  real3_t Dortho;
  real3_t dDortho[2];  // theta, phi
  real3_t ddDortho[3]; // theta theta, phi phi, theta phi
} line_t;