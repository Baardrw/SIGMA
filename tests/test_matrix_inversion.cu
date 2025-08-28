#include <iostream>
#include <vector>

#include "../include/config.h"
#include "../include/matrix_math.h"
#include "common/test_common_includes.h"
#include <Eigen/Dense>

// Test for 4x4 matrix inversion kernel
bool test_invert_4x4() {
  std::cout << "Running test_invert_4x4..." << std::endl;

  const int num_test_matrices = 100;
  const int block_size = 32;
  const int num_blocks = num_test_matrices;

  // Host arrays
  std::vector<Matrix4> h_input_matrices(num_test_matrices);
  std::vector<Matrix4> h_output_matrices(num_test_matrices);
  std::vector<int> h_results(num_test_matrices);
  std::vector<Matrix4> h_expected_inverses(num_test_matrices);

  // Generate test matrices and compute expected inverses using Eigen
  for (int i = 0; i < num_test_matrices; i++) {
    // Create test matrix with known properties
    Eigen::Matrix4f eigen_matrix;

    if (i < 10) {
      // Identity matrices (easy case)
      eigen_matrix = Eigen::Matrix4f::Identity();
    } else if (i < 20) {
      // Diagonal matrices
      eigen_matrix = Eigen::Matrix4f::Zero();
      eigen_matrix(0, 0) = 2.0f + i * 0.1f;
      eigen_matrix(1, 1) = 3.0f + i * 0.1f;
      eigen_matrix(2, 2) = 4.0f + i * 0.1f;
      eigen_matrix(3, 3) = 5.0f + i * 0.1f;
    } else if (i < 90) {
      // Random ill conditioned matrices
      eigen_matrix = Eigen::Matrix4f::Random() * 10.0f;

      // Print condition number for debugging
      Eigen::JacobiSVD<Eigen::Matrix4f> svd(eigen_matrix);
      float cond_number = svd.singularValues()(0) / svd.singularValues()(3);
      // std::cout << "Matrix " << i << " condition number: " << cond_number <<
      // std::endl;
    } else {
      // Near-singular matrices (should fail to invert)
      eigen_matrix = Eigen::Matrix4f::Random() * 1e-10f;
    }

    // Copy to Matrix4 format
    for (int row = 0; row < 4; row++) {
      for (int col = 0; col < 4; col++) {
        h_input_matrices[i](row, col) = eigen_matrix(row, col);
      }
    }

    // Compute expected inverse using Eigen
    if (i < 90) { // For non-singular matrices
      Eigen::Matrix4f inverse = eigen_matrix.inverse();
      for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
          h_expected_inverses[i](row, col) = inverse(row, col);
        }
      }
    }
  }

  // Device arrays
  Matrix4 *d_input_matrices, *d_output_matrices;
  int *d_results;

  CUDA_CHECK(
      cudaMalloc(&d_input_matrices, num_test_matrices * sizeof(Matrix4)));
  CUDA_CHECK(
      cudaMalloc(&d_output_matrices, num_test_matrices * sizeof(Matrix4)));
  CUDA_CHECK(cudaMalloc(&d_results, num_test_matrices * sizeof(int)));

  // Copy input data to device
  CUDA_CHECK(cudaMemcpy(d_input_matrices, h_input_matrices.data(),
                        num_test_matrices * sizeof(Matrix4),
                        cudaMemcpyHostToDevice));

  // Launch test kernel
  matrixMath::test_invert_4x4<<<num_blocks, block_size>>>(
      d_input_matrices, d_output_matrices, d_results, num_test_matrices);

  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());

  // Copy results back to host
  CUDA_CHECK(cudaMemcpy(h_output_matrices.data(), d_output_matrices,
                        num_test_matrices * sizeof(Matrix4),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_results.data(), d_results,
                        num_test_matrices * sizeof(int),
                        cudaMemcpyDeviceToHost));

  // Validate results
  bool all_tests_passed = true;
  int successful_inversions = 0;
  int failed_inversions = 0;
  int accuracy_failures = 0;

  const real_t tolerance = 1e-4f;

  for (int i = 0; i < num_test_matrices; i++) {
    if (i >= 90) {
      // Near-singular matrices should fail
      if (h_results[i]) {
        std::cout << "Warning: Matrix " << i
                  << " should have failed to invert (near-singular)"
                  << std::endl;
      } else {
        failed_inversions++;
      }
    } else {
      // Well-conditioned matrices should succeed
      if (!h_results[i]) {
        std::cout << "Error: Matrix " << i
                  << " failed to invert but should have succeeded" << std::endl;
        all_tests_passed = false;
        failed_inversions++;
        continue;
      }

      successful_inversions++;

      // Check accuracy by comparing with expected inverse
      real_t max_error = 0.0f;
      for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
          real_t error = std::abs(h_output_matrices[i](row, col) -
                                  h_expected_inverses[i](row, col));
          max_error = std::max(max_error, error);
        }
      }

      if (max_error > tolerance) {
        std::cout << "Accuracy error for matrix " << i
                  << ": max error = " << max_error << std::endl;
        accuracy_failures++;
        all_tests_passed = false;
      }

      // Verify that A * A^(-1) = I
      Matrix4 identity_check;
      for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
          identity_check(row, col) = 0.0f;
          for (int k = 0; k < 4; k++) {
            identity_check(row, col) +=
                h_input_matrices[i](row, k) * h_output_matrices[i](k, col);
          }
        }
      }

      // Check if result is close to identity
      real_t identity_error = 0.0f;
      for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
          real_t expected = (row == col) ? 1.0f : 0.0f;
          identity_error = std::max(
              identity_error, std::abs(identity_check(row, col) - expected));
        }
      }

      if (identity_error > tolerance) {
        std::cout << "Identity check failed for matrix " << i
                  << ": max error = " << identity_error << std::endl;
        all_tests_passed = false;
      }
    }
  }

  // Print statistics
  std::cout << "\n=== Matrix Inversion Test Results ===" << std::endl;
  std::cout << "Total test matrices: " << num_test_matrices << std::endl;
  std::cout << "Successful inversions: " << successful_inversions << std::endl;
  std::cout << "Failed inverses (near-singular): " << failed_inversions
            << std::endl;
  std::cout << "Accuracy failures: " << accuracy_failures << std::endl;
  std::cout << "Tolerance used: " << tolerance << std::endl;
  std::cout << "===================================\n" << std::endl;

  // Cleanup
  CUDA_CHECK(cudaFree(d_input_matrices));
  CUDA_CHECK(cudaFree(d_output_matrices));
  CUDA_CHECK(cudaFree(d_results));

  return all_tests_passed;
}

#include <iomanip>
#include <iostream>
#include <vector>

#include "../include/config.h"
#include "../include/matrix_math.h"
#include "common/test_common_includes.h"
#include <Eigen/Dense>

void print_matrix(const Matrix4 &mat, const std::string &name) {
  std::cout << std::fixed << std::setprecision(6);
  std::cout << name << ":" << std::endl;
  for (int i = 0; i < 4; i++) {
    std::cout << "  ";
    for (int j = 0; j < 4; j++) {
      std::cout << std::setw(12) << mat(i, j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void print_eigen_matrix(const Eigen::Matrix4f &mat, const std::string &name) {
  std::cout << std::fixed << std::setprecision(6);
  std::cout << name << ":" << std::endl;
  for (int i = 0; i < 4; i++) {
    std::cout << "  ";
    for (int j = 0; j < 4; j++) {
      std::cout << std::setw(12) << mat(i, j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

bool test_single_known_matrix() {
  std::cout << "=== Testing Single Known Matrix ===" << std::endl;

  // Create a simple, well-conditioned test matrix
  Eigen::Matrix4f eigen_matrix;
  // clang-format off
  eigen_matrix << 7094196363.602517127990723, 513750587.747360944747925, 152919.733894120698096, -1841049.430075737880543,
                  513750587.747360944747925, 400233044.300575911998749, -213336.136733991035726, -197513.861572516208980,
                  152919.733894120698096, -213336.136733991035726, 144.000000999999941, 0.000000000000000,
                 -1841049.430075737880543, -197513.861572516208980, 0.000000000000000, 492.090051914652463;
  // clang-format on

  // Print the test matrix
  print_eigen_matrix(eigen_matrix, "Input Matrix");

  // Check condition number
  Eigen::JacobiSVD<Eigen::Matrix4f> svd(eigen_matrix);
  float cond_number = svd.singularValues()(0) / svd.singularValues()(3);
  std::cout << "Condition number: " << cond_number << std::endl;
  std::cout << "Determinant: " << eigen_matrix.determinant() << std::endl
            << std::endl;

  // Compute expected inverse using Eigen
  Eigen::Matrix4f eigen_inverse = eigen_matrix.inverse();
  print_eigen_matrix(eigen_inverse, "Expected Inverse (Eigen)");

  // Verify Eigen's inverse: A * A^(-1) should = I
  Eigen::Matrix4f identity_check = eigen_matrix * eigen_inverse;
  print_eigen_matrix(identity_check, "A * A^(-1) (should be identity)");

  // Expected cofactor matrix
  Eigen::Matrix4f cofactor_matrix;
  // clang-format off
  cofactor_matrix << 52.0f, -19.0f, 5.0f, -1.0f,
                    -19.0f, 38.0f, -10.0f, 2.0f,
                     5.0f, -10.0f, 25.0f, -5.0f,
                    -1.0f, 2.0f, -5.0f, 18.0f;
  // clang-format on
  print_eigen_matrix(cofactor_matrix, "Expected Cofactor Matrix (Eigen)");

  // Convert to Matrix4 format
  Matrix4 h_input_matrix, h_output_matrix;
  for (int row = 0; row < 4; row++) {
    for (int col = 0; col < 4; col++) {
      h_input_matrix(row, col) = eigen_matrix(row, col);
    }
  }

  // Device arrays
  Matrix4 *d_input_matrix, *d_output_matrix;
  int *d_result;

  CUDA_CHECK(cudaMalloc(&d_input_matrix, sizeof(Matrix4)));
  CUDA_CHECK(cudaMalloc(&d_output_matrix, sizeof(Matrix4)));
  CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));

  // Copy input to device
  CUDA_CHECK(cudaMemcpy(d_input_matrix, &h_input_matrix, sizeof(Matrix4),
                        cudaMemcpyHostToDevice));

  // Launch kernel with just 1 thread block, 32 threads
  matrixMath::test_invert_4x4<<<1, 32>>>(d_input_matrix, d_output_matrix,
                                         d_result, 1);

  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());

  // Copy result back
  int result;
  CUDA_CHECK(cudaMemcpy(&h_output_matrix, d_output_matrix, sizeof(Matrix4),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost));

  std::cout << "Inversion result: " << (result ? "SUCCESS" : "FAILED")
            << std::endl
            << std::endl;

  if (!result) {
    std::cout << "Matrix inversion failed on GPU!" << std::endl;
    return false;
  }

  // Print GPU result
  print_matrix(h_output_matrix, "GPU Inverse Result");

  // Compare with expected result
  std::cout << "=== Comparison with Expected ===" << std::endl;
  real_t max_error = 0.0f;
  for (int row = 0; row < 4; row++) {
    for (int col = 0; col < 4; col++) {
      real_t expected = eigen_inverse(row, col);
      real_t actual = h_output_matrix(row, col);
      real_t error = std::abs(actual - expected);
      max_error = std::max(max_error, error);

      std::cout << "(" << row << "," << col << "): Expected=" << std::setw(12)
                << expected << ", Got=" << std::setw(12) << actual
                << ", Error=" << std::setw(12) << error << std::endl;
    }
  }
  std::cout << "Maximum error: " << max_error << std::endl << std::endl;

  // Verify identity: A * GPU_inverse = I
  std::cout << "=== Identity Check: A * GPU_Inverse ===" << std::endl;
  Matrix4 gpu_identity_check;
  real_t max_identity_error = 0.0f;

  for (int row = 0; row < 4; row++) {
    for (int col = 0; col < 4; col++) {
      gpu_identity_check(row, col) = 0.0f;
      for (int k = 0; k < 4; k++) {
        gpu_identity_check(row, col) +=
            h_input_matrix(row, k) * h_output_matrix(k, col);
      }

      real_t expected = (row == col) ? 1.0f : 0.0f;
      real_t error = std::abs(gpu_identity_check(row, col) - expected);
      max_identity_error = std::max(max_identity_error, error);

      std::cout << "(" << row << "," << col << "): " << std::setw(12)
                << gpu_identity_check(row, col) << " (should be " << expected
                << ", error=" << error << ")" << std::endl;
    }
  }

  std::cout << "Maximum identity error: " << max_identity_error << std::endl
            << std::endl;

  // Test with different tolerance levels
  const real_t tolerance = 1e-4f;
  bool accuracy_ok = max_error < tolerance;
  bool identity_ok = max_identity_error < tolerance;

  std::cout << "=== Final Results ===" << std::endl;
  std::cout << "Accuracy test (tolerance " << tolerance
            << "): " << (accuracy_ok ? "PASS" : "FAIL") << std::endl;
  std::cout << "Identity test (tolerance " << tolerance
            << "): " << (identity_ok ? "PASS" : "FAIL") << std::endl;
  std::cout << "Overall: "
            << (accuracy_ok && identity_ok ? "SUCCESS" : "FAILURE")
            << std::endl;

cleanup:
  CUDA_CHECK(cudaFree(d_input_matrix));
  CUDA_CHECK(cudaFree(d_output_matrix));
  CUDA_CHECK(cudaFree(d_result));

  return result && (max_error < tolerance) && (max_identity_error < tolerance);
}

bool test_identity_matrix() {
  std::cout << "\n=== Testing Identity Matrix (Should be Trivial) ==="
            << std::endl;

  Matrix4 h_input_matrix = Matrix4::Identity();
  Matrix4 h_output_matrix;

  print_matrix(h_input_matrix, "Identity Matrix Input");

  Matrix4 *d_input_matrix, *d_output_matrix;
  int *d_result;

  CUDA_CHECK(cudaMalloc(&d_input_matrix, sizeof(Matrix4)));
  CUDA_CHECK(cudaMalloc(&d_output_matrix, sizeof(Matrix4)));
  CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));

  CUDA_CHECK(cudaMemcpy(d_input_matrix, &h_input_matrix, sizeof(Matrix4),
                        cudaMemcpyHostToDevice));

  matrixMath::test_invert_4x4<<<1, 32>>>(d_input_matrix, d_output_matrix,
                                         d_result, 1);

  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());

  int result;
  CUDA_CHECK(cudaMemcpy(&h_output_matrix, d_output_matrix, sizeof(Matrix4),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost));

  std::cout << "Identity inversion result: " << (result ? "SUCCESS" : "FAILED")
            << std::endl;

  if (result) {
    print_matrix(h_output_matrix, "GPU Result (should be identity)");

    real_t max_error = 0.0f;
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        real_t expected = (i == j) ? 1.0f : 0.0f;
        real_t error = std::abs(h_output_matrix(i, j) - expected);
        max_error = std::max(max_error, error);
      }
    }
    std::cout << "Max error from identity: " << max_error << std::endl;
  }

  CUDA_CHECK(cudaFree(d_input_matrix));
  CUDA_CHECK(cudaFree(d_output_matrix));
  CUDA_CHECK(cudaFree(d_result));

  return result;
}

int main() {
  std::cout << "Starting focused matrix inversion tests..." << std::endl;
  CUDA_CHECK(cudaSetDevice(0));

  bool all_passed = true;

  // Test identity matrix first (should be trivial)
  all_passed &= test_identity_matrix();

  // Test a single known matrix with detailed analysis
  all_passed &= test_single_known_matrix();

  if (all_passed) {
    std::cout << "\nAll focused tests passed!" << std::endl;
    return 0;
  } else {
    std::cout << "\nSome focused tests failed!" << std::endl;
    return 1;
  }
}
