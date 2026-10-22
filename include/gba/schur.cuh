#pragma once

// Batched small-block linear algebra for the Schur complement, written
// in-repo (no cuBLAS batched dependency): one thread per block, matrices
// row-major. The camera reduced camera system in bundle adjustment is a
// block-diagonal 9x9 SPD Cholesky; the point eliminator needs a batched
// 3x3 SPD inverse. Both fail loudly (return false) on non-SPD input —
// no silent garbage.
//
// v1 solves the dense reduced camera system S on the host with Eigen
// (C <= 49 in the test datasets). These device routines cover the
// per-block pieces (V^-1 used by lm.cu; the 9x9 factor/solve exercised by
// tests/test_schur.cu) and are the building blocks for a future
// device-side sparse S.

#include <cuda_runtime.h>

namespace gba::dev {

// In-place-style Cholesky factor of a symmetric 9x9 (only the lower
// triangle of A is read). L (9x9, lower triangle valid, upper zeroed).
// Returns false if A is not positive definite.
__device__ inline bool chol9_factor(const double* A, double* L) {
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = A[i * 9 + j];
            for (int k = 0; k < j; ++k) s -= L[i * 9 + k] * L[j * 9 + k];
            if (i == j) {
                if (!(s > 0.0)) return false;
                L[i * 9 + i] = sqrt(s);
            } else {
                L[i * 9 + j] = s / L[j * 9 + j];
            }
        }
        for (int j = i + 1; j < 9; ++j) L[i * 9 + j] = 0.0;
    }
    return true;
}

// Solve L L^T x = b given L from chol9_factor.
__device__ inline void chol9_solve(const double* L, const double* b, double* x) {
    double y[9];
    for (int i = 0; i < 9; ++i) {
        double s = b[i];
        for (int k = 0; k < i; ++k) s -= L[i * 9 + k] * y[k];
        y[i] = s / L[i * 9 + i];
    }
    for (int i = 8; i >= 0; --i) {
        double s = y[i];
        for (int k = i + 1; k < 9; ++k) s -= L[k * 9 + i] * x[k];
        x[i] = s / L[i * 9 + i];
    }
}

// Factor A (symmetric 9x9) and solve A x = b. Returns false if not SPD.
__device__ inline bool chol9_factor_solve(const double* A, const double* b, double* x) {
    double L[81];
    if (!chol9_factor(A, L)) return false;
    chol9_solve(L, b, x);
    return true;
}

// Inverse of a symmetric positive definite 3x3 (lower triangle of A read)
// via Cholesky: A^-1 = L^-T L^-1. Full 3x3 written to Ainv.
// Returns false if A is not positive definite.
__device__ inline bool inverse3_spd(const double* A, double* Ainv) {
    double L[9];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = A[i * 3 + j];
            for (int k = 0; k < j; ++k) s -= L[i * 3 + k] * L[j * 3 + k];
            if (i == j) {
                if (!(s > 0.0)) return false;
                L[i * 3 + i] = sqrt(s);
            } else {
                L[i * 3 + j] = s / L[j * 3 + j];
            }
        }
    }
    double Li[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int i = 0; i < 3; ++i) {
        Li[i * 3 + i] = 1.0 / L[i * 3 + i];
        for (int j = 0; j < i; ++j) {
            double s = 0.0;
            for (int k = j; k < i; ++k) s -= L[i * 3 + k] * Li[k * 3 + j];
            Li[i * 3 + j] = s / L[i * 3 + i];
        }
    }
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double s = 0.0;
            for (int k = 0; k < 3; ++k) s += Li[k * 3 + i] * Li[k * 3 + j];
            Ainv[i * 3 + j] = s;
        }
    return true;
}

}  // namespace gba::dev

namespace gba {

// Batched: factor each of n symmetric 9x9 blocks and solve A_i x_i = b_i.
// ok[i] is set to 0 where a block is not SPD. One thread per block.
__global__ void batch_chol9_factor_solve(const double* A, const double* b, double* x,
                                         int* ok, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ok[i] = gba::dev::chol9_factor_solve(A + 81 * i, b + 9 * i, x + 9 * i) ? 1 : 0;
}

// Batched: invert each of n SPD 3x3 blocks. ok[i] = 0 where not SPD.
__global__ void batch_inverse3_spd(const double* A, double* Ainv, int* ok, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ok[i] = gba::dev::inverse3_spd(A + 9 * i, Ainv + 9 * i) ? 1 : 0;
}

}  // namespace gba
