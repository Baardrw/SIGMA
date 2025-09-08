#include "config.h"
#include "data_structures.h"
#include "matrix_math.h"

namespace matrixMath {

__device__ Vector4 matrix_vector_mult(Matrix4 mat, Vector4 vec) {
  Vector4 result;
  for (int i = 0; i < 4; ++i) {
    result[i] = mat(i, 0) * vec[0] + mat(i, 1) * vec[1] + mat(i, 2) * vec[2] +
                mat(i, 3) * vec[3];
  }
  return result;
}

__device__ bool invert_2x2(const Matrix2 &input, Matrix2 &output) {
  float det = input.determinant();
  if (fabsf(det) < 1e-7) {
    return false; // Matrix is singular, cannot invert
  }

  output(0, 0) = input(1, 1) / det;
  output(0, 1) = -input(0, 1) / det;
  output(1, 0) = -input(1, 0) / det;
  output(1, 1) = input(0, 0) / det;

  return true;
}

__device__ __forceinline__ bool
adjugate_inversion(cg::thread_block_tile<TILE_SIZE> &bucket_tile,
                   const Matrix4 &input_matrix, Matrix4 &inv_out) {
  // Initialize variables for ALL threads (prevents garbage in shuffles)
  real_t cofactor = 0.0f;
  real_t inv_element = 0.0f;
  bool success = true;

  int row = bucket_tile.thread_rank() / 4;
  int col = bucket_tile.thread_rank() % 4;

  // ==== Matrix of minors ====
  real_t minor = 0.0f;
  Matrix3 minor_matrix;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      if (i == row || j == col)
        continue;
      int minor_row = (i < row) ? i : i - 1;
      int minor_col = (j < col) ? j : j - 1;
      minor_matrix(minor_row, minor_col) = input_matrix(i, j);
    }
  }
  minor = minor_matrix.determinant();

  // ==== Cofactor matrix ====
  cofactor = ((row + col) % 2 == 0) ? minor : -minor;

  // ==== Determinant calculation - ALL threads participate in shuffles ====
  real_t det = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    bucket_tile.sync();
    real_t cofactor_0j = bucket_tile.shfl(cofactor, j);
    det += input_matrix(0, j) * cofactor_0j;
  }

  // Only bucket 0 has a valid determinant, broadcast to all threads
  bucket_tile.sync();
  det = bucket_tile.shfl(det, 0);

  // Check singularity
  if (fabsf(det) < 1e-20) {
    success = false;
    if (bucket_tile.thread_rank() == 0) {
      printf("Matrix is singular, cannot invert. Det: %.15e\n", det);
    }
  }

  // ==== Adjugate and inversion - threads 0-15 only ====
  real_t adjugate =
      bucket_tile.shfl(cofactor, col * 4 + row); // ALL participate
  inv_element = adjugate / det;

// ==== Result collection - ALL threads participate ====
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    real_t shfl = bucket_tile.shfl(inv_element, i); // ALL participate
    if (bucket_tile.thread_rank() == 0) {
      inv_out(i / 4, i % 4) = shfl;
    }
  }

  // Broadcast success flag
  bucket_tile.sync();
  success = bucket_tile.shfl(success, 0);
  return success;
}

__device__ bool invert_4x4(cg::thread_block_tile<TILE_SIZE> &bucket_tile,
                           Matrix4 &input, Matrix4 &output) {

  input += Matrix4::Identity() * 1e-6;
  bool status = adjugate_inversion(bucket_tile, input, output);
  bucket_tile.sync();

  return status;
}

// Declare template instantiations
__global__ void test_invert_4x4(Matrix4 *input_matrices,
                                Matrix4 *output_matrices, int *results,
                                int num_matrices) {
  // Calculate which matrix this thread block should process
  int matrix_idx = blockIdx.x;
  if (matrix_idx >= num_matrices) {
    printf("matrix_idx %d >= num_matrices %d\n", matrix_idx, num_matrices);
    return;
  }

  cg::thread_block_tile<16> bucket_tile =
      cg::tiled_partition<16>(cg::this_thread_block());

  // Each thread block processes one matrix
  Matrix4 input = input_matrices[matrix_idx];
  Matrix4 output;

  // Call your invert function
  bool success = invert_4x4(bucket_tile, input, output);

  // Store results (only one thread per block needs to write)
  if (bucket_tile.thread_rank() == 0) {
    output_matrices[matrix_idx] = output;
    if (success)
      results[matrix_idx] = 1;
    else {
      results[matrix_idx] = 0;
      printf("Matrix %d inversion failed.\n", matrix_idx);
    }
  }
}

} // namespace matrixMath