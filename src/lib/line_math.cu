#include "line_math.h"

// Helper function to compute D_ortho components
inline __device__ __host__ void
compute_Dortho_components(const line_t &line, const real3_t &W, real3_t &Dw,
                          real_t &norm_Dw, real3_t &D_ortho) {
  Dw = line.D - dot(line.D, W) * W;
  norm_Dw = norm3(Dw);
  D_ortho = 1 / norm_Dw * Dw;
}

inline __device__ __host__ real3_t get_delta_D_ortho(const line_t &line,
                                                     int param_idx,
                                                     const real3_t &W) {
  // Get gradient for the parameter
  real3_t dD =
      line.dD[param_idx]; // Assuming param_idx is 0 for THETA, 1 for PHI

  // Compute D_ortho components
  real3_t Dw, D_ortho;
  real_t norm_Dw;
  compute_Dortho_components(line, W, Dw, norm_Dw, D_ortho);

  real_t dD_dot_W = dot(dD, W);
  real_t D_dot_W = dot(line.D, W);
  real_t norm_Dw_squared = norm_Dw * norm_Dw;

  real3_t term1 = (dD - dD_dot_W * W) * (1 / norm_Dw);
  real3_t term2 = (dD_dot_W * D_dot_W) / norm_Dw_squared * D_ortho;

  return term1 - term2;
}

// get_dd_D_ortho implementation
inline __device__ __host__ real3_t get_dd_D_ortho(const line_t &line,
                                                  int param1_idx,
                                                  int param2_idx,
                                                  const real3_t &W) {
  // Get second derivative
  int dd_idx;
  if (param1_idx == THETA && param2_idx == THETA) {
    dd_idx = DD_THETA_THETA;
  } else if (param1_idx == PHI && param2_idx == PHI) {
    dd_idx = DD_PHI_PHI;
  } else {
    dd_idx = DD_THETA_PHI; // Mixed derivative
  }
  real3_t dd_D = line.ddD[dd_idx];

  // Get first derivatives
  real3_t delta_D1 = line.dD[param1_idx];
  real3_t delta_D2 = line.dD[param2_idx];

  // Compute D_ortho components
  real3_t Dw, D_ortho;
  real_t norm_Dw;
  compute_Dortho_components(line, W, Dw, norm_Dw, D_ortho);

  // Precompute common terms
  real_t D_dot_W = dot(line.D, W);
  real_t delta_D1_dot_W = dot(delta_D1, W);
  real_t delta_D2_dot_W = dot(delta_D2, W);
  real_t dd_D_dot_W = dot(dd_D, W);
  real_t norm_Dw_squared = norm_Dw * norm_Dw;

  // Get delta_D_ortho for param1 and param2
  real3_t delta_D_ortho_1 = get_delta_D_ortho(line, param1_idx, W);
  real3_t delta_D_ortho_2 = get_delta_D_ortho(line, param2_idx, W);

  // Compute the 5 terms
  real3_t t1 = (dd_D - dd_D_dot_W * W) * (1 / norm_Dw);

  real3_t t2 = (D_dot_W * delta_D2_dot_W) / norm_Dw_squared * delta_D_ortho_1;

  real3_t t3 = (D_dot_W * delta_D1_dot_W) / norm_Dw_squared * delta_D_ortho_2;

  real3_t t4 = (dd_D_dot_W * D_dot_W) / norm_Dw_squared * D_ortho;

  real3_t t5 = (delta_D1_dot_W * delta_D2_dot_W) / norm_Dw_squared * D_ortho;

  return t1 + t2 + t3 + t4 + t5;
}

inline __host__ __device__ void create_line(real_t x0, real_t y0, real_t phi,
                                            real_t theta, line_t &line) {
  real3_t D = {sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)};
  real3_t S0 = {x0, y0, 0.0};

  line.D = D;
  line.S0 = S0;
}

inline __host__ __device__ void
update_derived_line_quantities(line_t &line, real3_t &sensor_direction) {
  real_t theta = GET_THETA(line);
  real_t phi = GET_PHI(line);

  // Compute derivatives of D
  real3_t dD_theta = {cos(theta) * cos(phi), cos(theta) * sin(phi),
                      -sin(theta)};
  real3_t dD_phi = {-sin(theta) * sin(phi), sin(theta) * cos(phi), 0.0};
  line.dD[THETA] = dD_theta;
  line.dD[PHI] = dD_phi;

  // Compute second derivatives of D
  real3_t ddD_theta_theta = {-sin(theta) * cos(phi), -sin(theta) * sin(phi),
                             -cos(theta)};
  real3_t ddD_phi_phi = {sin(theta) * cos(phi), sin(theta) * sin(phi), 0.0};
  real3_t ddD_theta_phi = {-cos(theta) * sin(phi), cos(theta) * cos(phi), 0.0};

  line.ddD[DD_THETA_THETA] = ddD_theta_theta;
  line.ddD[DD_PHI_PHI] = ddD_phi_phi;
  line.ddD[DD_THETA_PHI] = ddD_theta_phi;

  // Compute Dortho and its derivatives
  real3_t Dw, D_ortho;
  real_t norm_Dw;
  compute_Dortho_components(line, sensor_direction, Dw, norm_Dw, D_ortho);

  line.dDortho[THETA] = get_delta_D_ortho(line, THETA, sensor_direction);
  line.dDortho[PHI] = get_delta_D_ortho(line, PHI, sensor_direction);

  line.ddDortho[DD_THETA_THETA] =
      get_dd_D_ortho(line, THETA, THETA, sensor_direction);
  line.ddDortho[DD_PHI_PHI] = get_dd_D_ortho(line, PHI, PHI, sensor_direction);
  line.ddDortho[DD_THETA_PHI] =
      get_dd_D_ortho(line, THETA, PHI, sensor_direction);
}
