#include "config.h"
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

    std::vector<Vector3> residuals;
    std::vector<Vector3> inverse_sigma_squared;

    // Get the start and end indices for this bucket
    int start_idx = h_data.buckets[i * 3];
    int rpc_idx = h_data.buckets[i * 3 + 1];
    int end_idx = h_data.buckets[i * 3 + 2];

    // Create line
    line_t line;
    lineMath::create_line(x0, y0, phi, theta, line);
    lineMath::compute_D_ortho(line);

    measurement_cache_t *measurement_cache =
        new measurement_cache_t[end_idx - start_idx];

    // Get MDT measurements
    for (int j = start_idx; j < rpc_idx; j++) {
      Vector3 K_j;
      K_j << h_data.sensor_pos_x[j] - x0, h_data.sensor_pos_y[j] - y0,
          h_data.sensor_pos_z[j];

      measurement_cache[j - start_idx].connection_vector = K_j;
      measurement_cache[j - start_idx].drift_radius = h_data.drift_radius[j];
    }

    // get inverse sigma squared
    for (int j = start_idx; j < end_idx; j++) {
      inverse_sigma_squared.push_back(Vector3(1/ h_data.sigma_x[j] / h_data.sigma_x[j],
                                               1/ h_data.sigma_y[j] / h_data.sigma_y[j],
                                               1/ h_data.sigma_z[j] / h_data.sigma_z[j]));
    }

    // Get RPC measurments
    for (int j = rpc_idx; j < end_idx; j++) {
      Vector3 P;
      P << h_data.sensor_pos_x[j], h_data.sensor_pos_y[j],
          h_data.sensor_pos_z[j];

      Vector3 sensor_direction;
      sensor_direction << h_data.sensor_dir_x[j], h_data.sensor_dir_y[j],
          h_data.sensor_dir_z[j];

      Vector3 plane_normal;
      plane_normal << h_data.plane_normal_x[j], h_data.plane_normal_y[j],
          h_data.plane_normal_z[j];

      measurement_cache[j - start_idx].sensor_pos = P;
      measurement_cache[j - start_idx].sensor_direction = sensor_direction;
      measurement_cache[j - start_idx].plane_normal = plane_normal;
    }

    residuals = residualMath::compute_residuals(
        line, measurement_cache, rpc_idx - start_idx, end_idx - rpc_idx);

    real_t chi2 = residualMath::get_chi2(residuals, inverse_sigma_squared);
    chi2_values[i] = chi2;
  }

  return chi2_values;
}