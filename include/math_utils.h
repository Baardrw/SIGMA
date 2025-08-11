#include "config.h"

// TODO optimize as vectorized operation?
inline __host__ __device__ real_t dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Define multiplication operators for real3_t
__device__ __host__ inline real3_t operator*(real_t scalar, const real3_t& vec) {
    return {scalar * vec.x, scalar * vec.y, scalar * vec.z};
}

__device__ __host__ inline real3_t operator*(const real3_t& vec, real_t scalar) {
    return {vec.x * scalar, vec.y * scalar, vec.z * scalar};
}

// In-place multiplication
__device__ __host__ inline real3_t& operator*=(real3_t& vec, real_t scalar) {
    vec.x *= scalar;
    vec.y *= scalar;
    vec.z *= scalar;
    return vec;
}

// Vector Addition Operators
__device__ __host__ inline real3_t operator+(const real3_t& a, const real3_t& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ __host__ inline real3_t& operator+=(real3_t& a, const real3_t& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

// Vector Subtraction Operators
__device__ __host__ inline real3_t operator-(const real3_t& a, const real3_t& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ __host__ inline real3_t& operator-=(real3_t& a, const real3_t& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}

// Unary Negation Operator
__device__ __host__ inline real3_t operator-(const real3_t& a) {
    return {-a.x, -a.y, -a.z};
}

