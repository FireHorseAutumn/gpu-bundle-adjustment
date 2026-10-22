// test_schur — validates the batched small-block routines in schur.cuh
// against an Eigen reference: 9x9 SPD Cholesky factor+solve and 3x3 SPD
// inverse, one thread per block, random well-conditioned inputs.

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <cstdio>
#include <random>
#include <vector>

#include "gba/schur.cuh"

namespace {

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

}  // namespace

int main() {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> unif(-1.0, 1.0);

    constexpr int N = 64;  // blocks per batch

    // --- 9x9 Cholesky factor + solve ---
    std::vector<double> h_A(81 * N), h_b(9 * N), h_x(9 * N);
    std::vector<double> ref_x(9 * N);
    for (int i = 0; i < N; ++i) {
        Eigen::Matrix<double, 9, 9> M = Eigen::Matrix<double, 9, 9>::NullaryExpr(
            [&]() { return unif(rng); });
        Eigen::Matrix<double, 9, 9> A = M.transpose() * M +
                                        9.0 * Eigen::Matrix<double, 9, 9>::Identity();
        Eigen::Matrix<double, 9, 1> b =
            Eigen::Matrix<double, 9, 1>::NullaryExpr([&]() { return unif(rng); });
        Eigen::Matrix<double, 9, 1> x = A.llt().solve(b);
        for (int r = 0; r < 9; ++r) {
            h_b[9 * i + r] = b[r];
            ref_x[9 * i + r] = x[r];
            for (int c = 0; c < 9; ++c) h_A[81 * i + r * 9 + c] = A(r, c);
        }
    }

    double *d_A, *d_b, *d_x;
    int* d_ok;
    check_cuda(cudaMalloc(&d_A, 81 * N * sizeof(double)), "malloc A");
    check_cuda(cudaMalloc(&d_b, 9 * N * sizeof(double)), "malloc b");
    check_cuda(cudaMalloc(&d_x, 9 * N * sizeof(double)), "malloc x");
    check_cuda(cudaMalloc(&d_ok, N * sizeof(int)), "malloc ok");
    check_cuda(cudaMemcpy(d_A, h_A.data(), 81 * N * sizeof(double), cudaMemcpyHostToDevice),
               "upload A");
    check_cuda(cudaMemcpy(d_b, h_b.data(), 9 * N * sizeof(double), cudaMemcpyHostToDevice),
               "upload b");
    gba::batch_chol9_factor_solve<<<1, N>>>(d_A, d_b, d_x, d_ok, N);
    check_cuda(cudaGetLastError(), "launch chol9");
    check_cuda(cudaMemcpy(h_x.data(), d_x, 9 * N * sizeof(double), cudaMemcpyDeviceToHost),
               "download x");
    std::vector<int> h_ok(N);
    check_cuda(cudaMemcpy(h_ok.data(), d_ok, N * sizeof(int), cudaMemcpyDeviceToHost),
               "download ok");

    double worst = 0.0;
    for (int i = 0; i < N; ++i) {
        if (!h_ok[i]) {
            std::fprintf(stderr, "block %d reported non-SPD for an SPD input\n", i);
            return 1;
        }
        for (int r = 0; r < 9; ++r)
            worst = std::max(worst, std::abs(h_x[9 * i + r] - ref_x[9 * i + r]));
    }
    if (worst > 1e-9) {
        std::fprintf(stderr, "chol9 solve max abs error %.3g exceeds 1e-9\n", worst);
        return 1;
    }
    std::printf("chol9 factor+solve: max abs error %.3g over %d blocks\n", worst, N);

    // --- 3x3 SPD inverse ---
    std::vector<double> h_A3(9 * N), h_inv(9 * N), ref_inv(9 * N);
    for (int i = 0; i < N; ++i) {
        Eigen::Matrix3d M = Eigen::Matrix3d::NullaryExpr([&]() { return unif(rng); });
        Eigen::Matrix3d A = M.transpose() * M + 3.0 * Eigen::Matrix3d::Identity();
        Eigen::Matrix3d Ai = A.inverse();
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) {
                h_A3[9 * i + r * 3 + c] = A(r, c);
                ref_inv[9 * i + r * 3 + c] = Ai(r, c);
            }
    }
    double *d_A3, *d_inv;
    check_cuda(cudaMalloc(&d_A3, 9 * N * sizeof(double)), "malloc A3");
    check_cuda(cudaMalloc(&d_inv, 9 * N * sizeof(double)), "malloc inv");
    check_cuda(cudaMemcpy(d_A3, h_A3.data(), 9 * N * sizeof(double), cudaMemcpyHostToDevice),
               "upload A3");
    gba::batch_inverse3_spd<<<1, N>>>(d_A3, d_inv, d_ok, N);
    check_cuda(cudaGetLastError(), "launch inverse3");
    check_cuda(cudaMemcpy(h_inv.data(), d_inv, 9 * N * sizeof(double), cudaMemcpyDeviceToHost),
               "download inv");
    check_cuda(cudaMemcpy(h_ok.data(), d_ok, N * sizeof(int), cudaMemcpyDeviceToHost),
               "download ok3");

    worst = 0.0;
    for (int i = 0; i < N; ++i) {
        if (!h_ok[i]) {
            std::fprintf(stderr, "3x3 block %d reported non-SPD for an SPD input\n", i);
            return 1;
        }
        for (int r = 0; r < 9; ++r)
            worst = std::max(worst, std::abs(h_inv[9 * i + r] - ref_inv[9 * i + r]));
    }
    if (worst > 1e-9) {
        std::fprintf(stderr, "inverse3 max abs error %.3g exceeds 1e-9\n", worst);
        return 1;
    }
    std::printf("inverse3_spd: max abs error %.3g over %d blocks\n", worst, N);

    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_ok);
    cudaFree(d_A3);
    cudaFree(d_inv);
    std::printf("PASS\n");
    return 0;
}
