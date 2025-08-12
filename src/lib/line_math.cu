#include "config.h"
#include "line_math.h"

namespace lineMath {

// Helper function to compute D_ortho components
inline __host__ __device__ void
compute_Dortho_components(const line_t &line, const Vector3 &W, Vector3 &Dw,
                          real_t &norm_Dw, Vector3 &D_ortho) {
  Dw = line.D - line.D.dot(W) * W;
  norm_Dw = Dw.norm();
  D_ortho = (1 / norm_Dw) * Dw;
}

// Helper to avoid computing unnecessary values (used during seed_line)
__host__ __device__ void compute_Dortho(line_t &line, const Vector3 &W) {
  Vector3 Dw, D_ortho;
  real_t norm_Dw;
  compute_Dortho_components(line, W, Dw, norm_Dw, D_ortho);
  line.Dortho = D_ortho;
}

inline __host__ __device__ Vector3 get_delta_D_ortho(const line_t &line,
                                                     int param_idx,
                                                     const Vector3 &W) {
  // Get gradient for the parameter
  Vector3 dD =
      line.dD[param_idx]; // Assuming param_idx is 0 for THETA, 1 for PHI

  // Compute D_ortho components
  Vector3 Dw, D_ortho;
  real_t norm_Dw;
  compute_Dortho_components(line, W, Dw, norm_Dw, D_ortho);

  real_t dD_dot_W = dD.dot(W);
  real_t D_dot_W = line.D.dot(W);
  real_t norm_Dw_squared = norm_Dw * norm_Dw;

  Vector3 term1 = (dD - dD_dot_W * W) * (1 / norm_Dw);
  Vector3 term2 = (dD_dot_W * D_dot_W) / norm_Dw_squared * D_ortho;

  return term1 - term2;
}

// get_dd_D_ortho implementation
inline __host__ __device__ Vector3 get_dd_D_ortho(const line_t &line,
                                                  int param1_idx,
                                                  int param2_idx,
                                                  const Vector3 &W) {
  // Get second derivative
  int dd_idx;
  if (param1_idx == THETA && param2_idx == THETA) {
    dd_idx = DD_THETA_THETA;
  } else if (param1_idx == PHI && param2_idx == PHI) {
    dd_idx = DD_PHI_PHI;
  } else {
    dd_idx = DD_THETA_PHI; // Mixed derivative
  }
  Vector3 dd_D = line.ddD[dd_idx];

  // Get first derivatives
  Vector3 delta_D1 = line.dD[param1_idx];
  Vector3 delta_D2 = line.dD[param2_idx];

  // Compute D_ortho components
  Vector3 Dw, D_ortho;
  real_t norm_Dw;
  compute_Dortho_components(line, W, Dw, norm_Dw, D_ortho);

  // Precompute common terms
  real_t D_dot_W = line.D.dot(W);
  real_t delta_D1_dot_W = delta_D1.dot(W);
  real_t delta_D2_dot_W = delta_D2.dot(W);
  real_t dd_D_dot_W = dd_D.dot(W);
  real_t norm_Dw_squared = norm_Dw * norm_Dw;

  // Get delta_D_ortho for param1 and param2
  Vector3 delta_D_ortho_1 = get_delta_D_ortho(line, param1_idx, W);
  Vector3 delta_D_ortho_2 = get_delta_D_ortho(line, param2_idx, W);

  // Compute the 5 terms
  Vector3 t1 = (dd_D - dd_D_dot_W * W) * (1 / norm_Dw);

  Vector3 t2 = (D_dot_W * delta_D2_dot_W) / norm_Dw_squared * delta_D_ortho_1;

  Vector3 t3 = (D_dot_W * delta_D1_dot_W) / norm_Dw_squared * delta_D_ortho_2;

  Vector3 t4 = (dd_D_dot_W * D_dot_W) / norm_Dw_squared * D_ortho;

  Vector3 t5 = (delta_D1_dot_W * delta_D2_dot_W) / norm_Dw_squared * D_ortho;

  return t1 + t2 + t3 + t4 + t5;
}

__host__ __device__ void create_line(real_t x0, real_t y0, real_t phi,
                                            real_t theta, line_t &line) {
  // Initialize direction vector D
  line.D << sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta);

  // Initialize starting point S0
  line.S0 << x0, y0, 0.0;

  // Store parameters
  line.params[0] = theta;
  line.params[1] = phi;
  line.params[2] = x0;
  line.params[3] = y0;
}

inline __host__ __device__ void
update_derived_line_quantities(line_t &line, Vector3 &sensor_direction) {
  real_t theta = GET_THETA(line);
  real_t phi = GET_PHI(line);

  // Compute derivatives of D
  line.dD[THETA] << cos(theta) * cos(phi), cos(theta) * sin(phi), -sin(theta);
  line.dD[PHI] << -sin(theta) * sin(phi), sin(theta) * cos(phi), 0.0;

  // Compute second derivatives of D
  line.ddD[DD_THETA_THETA] << -sin(theta) * cos(phi), -sin(theta) * sin(phi),
      -cos(theta);
  line.ddD[DD_PHI_PHI] << sin(theta) * cos(phi), sin(theta) * sin(phi), 0.0;
  line.ddD[DD_THETA_PHI] << -cos(theta) * sin(phi), cos(theta) * cos(phi), 0.0;

  // Compute Dortho and its derivatives
  Vector3 Dw, D_ortho;
  real_t norm_Dw;
  compute_Dortho_components(line, sensor_direction, Dw, norm_Dw, D_ortho);

  // Store D_ortho in the line structure
  line.Dortho = D_ortho;

  // Compute first derivatives of D_ortho
  line.dDortho[THETA] = get_delta_D_ortho(line, THETA, sensor_direction);
  line.dDortho[PHI] = get_delta_D_ortho(line, PHI, sensor_direction);

  // Compute second derivatives of D_ortho
  line.ddDortho[DD_THETA_THETA] =
      get_dd_D_ortho(line, THETA, THETA, sensor_direction);
  line.ddDortho[DD_PHI_PHI] = get_dd_D_ortho(line, PHI, PHI, sensor_direction);
  line.ddDortho[DD_THETA_PHI] =
      get_dd_D_ortho(line, THETA, PHI, sensor_direction);
}
} // namespace lineMath