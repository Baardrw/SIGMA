#include "data_structures.h"
#include "test_math_utils.h"

void calculate_bucket_answer(BucketGroundTruth &bucket_ground_truth,
                             const std::vector<MDTHit> &mdt_hits,
                             const std::vector<RPCHit> &rpc_hits) {

  std::vector<Eigen::Vector3d> real_hits;
  for (const auto &hit : mdt_hits) {
    real_hits.emplace_back(hit.poca_x, hit.poca_y, hit.poca_z);
  }
  // There is some faulty RPC data

  // for (const auto &hit : rpc_hits) {
  //   real_hits.emplace_back(hit.poca_x, hit.poca_y, hit.poca_z);
  // }

  // Calculate best fit line parameters
  if (real_hits.size() < 2) {
    bucket_ground_truth.theta = 0.0;
    bucket_ground_truth.phi = 0.0;
    bucket_ground_truth.x0 = 0.0;
    bucket_ground_truth.y0 = 0.0;
    return;
  }

  // Step 1: Calculate centroid
  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  for (const auto &point : real_hits) {
    centroid += point;
  }
  centroid /= static_cast<double>(real_hits.size());

  // Step 2: Center the points
  Eigen::MatrixXd centered_points(3, real_hits.size());
  for (size_t i = 0; i < real_hits.size(); ++i) {
    centered_points.col(i) = real_hits[i] - centroid;
  }

  // Step 3: Perform SVD
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      centered_points, Eigen::ComputeThinU | Eigen::ComputeThinV);

  // The first column of U gives the direction of maximum variance (best fit
  // line)
  Eigen::Vector3d direction = svd.matrixU().col(0);

  // Ensure consistent direction
  if (direction.x() < 0) {
    direction = -direction;
  }

  // Convert to spherical coordinates
  double dx = direction.x();
  double dy = direction.y();
  double dz = direction.z();

  bucket_ground_truth.theta = std::acos(dz / direction.norm());
  bucket_ground_truth.phi = std::atan2(dy, dx);

  // Calculate (x0, y0) - point on line at z=0
  if (std::abs(dz) > 1e-10) {
    double t = -centroid.z() / dz;
    Eigen::Vector3d point_at_z0 = centroid + t * direction;
    bucket_ground_truth.x0 = point_at_z0.x();
    bucket_ground_truth.y0 = point_at_z0.y();
  } else {
    bucket_ground_truth.x0 = centroid.x();
    bucket_ground_truth.y0 = centroid.y();
  }
}

// TODO: update to 3D
std::vector<real_t> calculate_chi2(Data &h_data, int num_buckets) {

  std::vector<real_t> chi2_values(num_buckets, 0.0f);
  for (int i = 0; i < num_buckets; i++) {
    real_t x0 = h_data.x0[i];
    real_t y0 = h_data.y0[i];
    real_t phi = h_data.phi[i];
    real_t theta = h_data.theta[i];

    std::vector<real_t> residuals;
    std::vector<real_t> inverse_sigma_squared;

    // Get the start and end indices for this bucket
    int start_idx = h_data.buckets[i * 3];
    int rpc_idx = h_data.buckets[i * 3 + 1];
    int end_idx = h_data.buckets[i * 3 + 2];

    // Create line
    line_t line;
    lineMath::create_line(x0, y0, phi, theta, line);
    lineMath::compute_D_ortho(line);

    // Compute K
    std::vector<Vector3> K;
    for (int j = start_idx; j < rpc_idx; j++) {
      Vector3 K_j;
      K_j << h_data.sensor_pos_x[j] - x0, h_data.sensor_pos_y[j] - y0,
          h_data.sensor_pos_z[j];

      K.push_back(K_j);
    }

    // Get drift radius for this bucket
    std::vector<real_t> drift_radius;
    for (int j = start_idx; j < rpc_idx; j++) {
      drift_radius.push_back(h_data.drift_radius[j]);
    }

    // get inverse sigma squared
    for (int j = start_idx; j < end_idx; j++) {
      inverse_sigma_squared.push_back(1.0f /
                                      (h_data.sigma_x[j] * h_data.sigma_x[j]));
    }

    residuals = residualMath::compute_residuals(
        line, K, drift_radius, rpc_idx - start_idx, end_idx - rpc_idx);

    real_t chi2 = residualMath::get_chi2(residuals, inverse_sigma_squared);
    chi2_values[i] = chi2;
  }

  return chi2_values;
}