#include "config.h"
#include "data_structures.h"
#include "line_math.h"

namespace lineMath {

// Helper function to compute D_ortho components
inline __host__ __device__ void
compute_D_ortho_components(const line_t &line, Vector3 &Dw,
                           real_t &norm_Dw, Vector3 &D_ortho) {
  Dw = line.D - line.D.dot(MDT_DIR) * MDT_DIR;
  norm_Dw = Dw.norm();
  D_ortho = (1 / norm_Dw) * Dw;
}

// Helper to avoid computing unnecessary values (used during seed_line)
__host__ __device__ void compute_D_ortho(line_t &line) {
  Vector3 Dw, D_ortho;
  real_t norm_Dw;
  compute_D_ortho_components(line, Dw, norm_Dw, D_ortho);
  line.D_ortho = D_ortho;
}

inline __host__ __device__ Vector3 get_delta_D_ortho(const line_t &line,
                                                     int param_idx,
                                                     Vector3 D_ortho,
                                                     real_t norm_Dw) {
  // Get gradient for the parameter
  Vector3 dD =
      line.dD[param_idx]; // Assuming param_idx is 0 for THETA, 1 for PHI

  real_t dD_dot_W = dD.dot(MDT_DIR);
  real_t D_dot_W = line.D.dot(MDT_DIR);
  real_t norm_Dw_squared = norm_Dw * norm_Dw;

  Vector3 term1 = (dD - dD_dot_W * MDT_DIR) * (1 / norm_Dw);
  Vector3 term2 = (dD_dot_W * D_dot_W) / norm_Dw_squared * D_ortho;

  return term1 - term2;
}

// get_dd_D_ortho implementation
inline __host__ __device__ Vector3 get_dd_D_ortho(const line_t &line,
                                                  int param1_idx,
                                                  int param2_idx,
                                                  real_t norm_Dw) {
  // Get second derivative
  int dd_idx;
  if (param1_idx == THETA && param2_idx == THETA) {
    dd_idx = DD_THETA_THETA;
  } else if (param1_idx == PHI && param2_idx == PHI) {
    dd_idx = DD_PHI_PHI;
  } else {
    dd_idx = DD_THETA_PHI; // Mixed derivative
  }

  // Params from D
  // Get first derivatives
  Vector3 delta_D1 = line.dD[param1_idx];
  Vector3 delta_D2 = line.dD[param2_idx];
  Vector3 dd_D = line.ddD[dd_idx];

  // Params from D_ortho
  Vector3 D_ortho = line.D_ortho;
  Vector3 delta_D_ortho_1 = line.dD_ortho[param1_idx];
  Vector3 delta_D_ortho_2 = line.dD_ortho[param2_idx];

  // Precompute common terms
  real_t D_dot_W = line.D.dot(MDT_DIR);
  real_t delta_D1_dot_W = delta_D1.dot(MDT_DIR);
  real_t delta_D2_dot_W = delta_D2.dot(MDT_DIR);
  real_t dd_D_dot_W = dd_D.dot(MDT_DIR);
  real_t norm_Dw_squared = norm_Dw * norm_Dw;

  // Get delta_D_ortho for param1 and param2

  // Compute the 5 terms
  Vector3 t1 = (dd_D - dd_D_dot_W * MDT_DIR) * (1 / norm_Dw);

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
  line.params[THETA] = theta;
  line.params[PHI] = phi;
  line.params[X0] = x0;
  line.params[Y0] = y0;
}

__host__ __device__ void update_derivatives(line_t &line) {
  real_t theta = GET_THETA(line);
  real_t phi = GET_PHI(line);

  // ====== Derivatives of D ======
  line.dD[THETA] << cos(theta) * cos(phi), cos(theta) * sin(phi), -sin(theta);
  line.dD[PHI] << -sin(theta) * sin(phi), sin(theta) * cos(phi), 0.0;

  line.ddD[DD_THETA_THETA] << -sin(theta) * cos(phi), -sin(theta) * sin(phi),
      -cos(theta);
  line.ddD[DD_PHI_PHI] << sin(theta) * cos(phi), sin(theta) * sin(phi), 0.0;
  line.ddD[DD_THETA_PHI] << -cos(theta) * sin(phi), cos(theta) * cos(phi), 0.0;

  // ====== Derivatives of D ortho ======
  Vector3 Dw, D_ortho;
  real_t norm_Dw;
  compute_D_ortho_components(line, Dw, norm_Dw, D_ortho);

  // Store D_ortho in the line structure
  line.D_ortho = D_ortho;

  // Compute first derivatives of D_ortho
  line.dD_ortho[THETA] =
      get_delta_D_ortho(line, THETA, D_ortho, norm_Dw);
  line.dD_ortho[PHI] =
      get_delta_D_ortho(line, PHI, D_ortho, norm_Dw);

  // Compute second derivatives of D_ortho
  line.ddD_ortho[DD_THETA_THETA] =
      get_dd_D_ortho(line, THETA, THETA, norm_Dw);
  line.ddD_ortho[DD_PHI_PHI] =
      get_dd_D_ortho(line, PHI, PHI, norm_Dw);
  line.ddD_ortho[DD_THETA_PHI] =
      get_dd_D_ortho(line, THETA, PHI, norm_Dw);
}
} // namespace lineMath