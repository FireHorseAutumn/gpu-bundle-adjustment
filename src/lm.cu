// lm.cu — GPU Levenberg-Marquardt for BAL problems.
//
// Structure per outer iteration:
//   1. edge kernel: per observation, residual + analytic 2x12 Jacobian,
//      atomically accumulated into per-camera U (9x9), per-point V (3x3),
//      per-edge W (9x3), rhs b = -J^T r, and the total squared error.
//   2. per trial: damp + invert V blocks on device (batched 3x3 from
//      schur.cuh), assemble the dense reduced camera system
//      S = U - W V^-1 W^T on the host (9C x 9C, Eigen LLT),
//      back-substitute the point step, evaluate the trial cost on device.
//   3. Nielsen gain-ratio accept/reject with Marquardt diagonal damping.
//
// v1 deliberately keeps S dense and host-side. That made LLT trivial on
// the original C <= 49 tests, but larger BAL runs hit the dense assembly
// and factorization wall. Follow-up: sparse S (block CSR over the camera
// graph) + device-side solve — the assembly loop below is the exact place
// to swap it in.
//
// Anomaly policy: non-finite residuals/Jacobians and non-SPD V blocks are
// counted on device, never silently dropped. The counter is returned in
// LMResult and asserted to be zero in tests.

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "gba/lm.hpp"
#include "gba/schur.cuh"

#define GBA_CUDA_CHECK(call)                                                     \
    do {                                                                         \
        cudaError_t err_ = (call);                                               \
        if (err_ != cudaSuccess)                                                 \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ +  \
                                     ":" + std::to_string(__LINE__) + ": " +     \
                                     cudaGetErrorString(err_));                  \
    } while (0)

namespace gba {
namespace {

constexpr int kBlock = 256;

// Residual only, formula identical to the Ceres bal example (first-order
// branch for theta -> 0).
__host__ __device__ inline void edge_residual(const double* cam, const double* X,
                                              const double* obs, double* r) {
    const double* a = cam;
    double P[3];
    double theta2 = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
    if (theta2 > 1e-24) {
        double theta = sqrt(theta2);
        double w[3] = {a[0] / theta, a[1] / theta, a[2] / theta};
        double c = cos(theta), s = sin(theta);
        double wx[3] = {w[1] * X[2] - w[2] * X[1], w[2] * X[0] - w[0] * X[2],
                        w[0] * X[1] - w[1] * X[0]};
        double q = w[0] * X[0] + w[1] * X[1] + w[2] * X[2];
        for (int i = 0; i < 3; ++i)
            P[i] = X[i] * c + wx[i] * s + w[i] * q * (1.0 - c) + cam[3 + i];
    } else {
        double ax[3] = {a[1] * X[2] - a[2] * X[1], a[2] * X[0] - a[0] * X[2],
                        a[0] * X[1] - a[1] * X[0]};
        for (int i = 0; i < 3; ++i) P[i] = X[i] + ax[i] + cam[3 + i];
    }
    double u = -P[0] / P[2], v = -P[1] / P[2];
    double r2 = u * u + v * v;
    double dist = 1.0 + cam[7] * r2 + cam[8] * r2 * r2;
    r[0] = cam[6] * dist * u - obs[0];
    r[1] = cam[6] * dist * v - obs[1];
}

// Residual + analytic Jacobian. Jc is 2x9 row-major (columns: rodrigues[3],
// t[3], f, k1, k2), Jp is 2x3 row-major.
//
// Derivation (main branch, w = a/theta, c = cos theta, s = sin theta):
//   RX   = c X + s (w x X) + (1-c) (w.X) w
//   dRX/dw = -s [X]x + (1-c) ((w.X) I + w X^T)
//   dRX/da = (dRX/dw)(I - w w^T)/theta + (dRX/dtheta) w^T
//   d(u,v)/dP = [[-1/Pz, 0, Px/Pz^2], [0, -1/Pz, Py/Pz^2]]
// then chain through the distortion 2x2 D and projection.
__host__ __device__ inline void edge_residual_jacobian(const double* cam, const double* X,
                                                       const double* obs, double* r,
                                                       double* Jc, double* Jp) {
    edge_residual(cam, X, obs, r);

    const double* a = cam;
    double theta2 = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
    double R[9];    // d(rotated point)/dX
    double Dra[9];  // d(rotated point)/d(rodrigues), rows = point coords
    double Pp[3];

    if (theta2 > 1e-24) {
        double theta = sqrt(theta2);
        double w[3] = {a[0] / theta, a[1] / theta, a[2] / theta};
        double c = cos(theta), s = sin(theta);
        double wx[3] = {w[1] * X[2] - w[2] * X[1], w[2] * X[0] - w[0] * X[2],
                        w[0] * X[1] - w[1] * X[0]};
        double q = w[0] * X[0] + w[1] * X[1] + w[2] * X[2];
        for (int i = 0; i < 3; ++i) Pp[i] = X[i] * c + wx[i] * s + w[i] * q * (1.0 - c);

        // R = c I + s [w]x + (1-c) w w^T
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double wxm = (i == 0 ? (j == 1 ? -w[2] : (j == 2 ? w[1] : 0.0))
                              : i == 1 ? (j == 0 ? w[2] : (j == 2 ? -w[0] : 0.0))
                                       : (j == 0 ? -w[1] : (j == 1 ? w[0] : 0.0)));
                R[i * 3 + j] = (i == j ? c : 0.0) + s * wxm + (1.0 - c) * w[i] * w[j];
            }

        double dth[3];  // dRX/dtheta
        for (int i = 0; i < 3; ++i) dth[i] = -s * X[i] + c * wx[i] + s * q * w[i];

        // A = dRX/dw = -s [X]x + (1-c) (q I + w X^T)
        double A[9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double xx = (i == 0 ? (j == 1 ? -X[2] : (j == 2 ? X[1] : 0.0))
                             : i == 1 ? (j == 0 ? X[2] : (j == 2 ? -X[0] : 0.0))
                                      : (j == 0 ? -X[1] : (j == 1 ? X[0] : 0.0)));
                A[i * 3 + j] = -s * xx + (1.0 - c) * ((i == j ? q : 0.0) + w[i] * X[j]);
            }
        double Aw[3] = {0.0, 0.0, 0.0};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) Aw[i] += A[i * 3 + j] * w[j];
        // dRX/da = (A - Aw w^T)/theta + dth w^T
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                Dra[i * 3 + j] = (A[i * 3 + j] - Aw[i] * w[j]) / theta + dth[i] * w[j];
    } else {
        // First-order model RX = X + a x X: dRX/da = -[X]x, R = I.
        double ax[3] = {a[1] * X[2] - a[2] * X[1], a[2] * X[0] - a[0] * X[2],
                        a[0] * X[1] - a[1] * X[0]};
        for (int i = 0; i < 3; ++i) Pp[i] = X[i] + ax[i];
        for (int i = 0; i < 9; ++i) R[i] = (i % 4 == 0) ? 1.0 : 0.0;
        double nx[9] = {0.0, X[2], -X[1], -X[2], 0.0, X[0], X[1], -X[0], 0.0};
        for (int i = 0; i < 9; ++i) Dra[i] = nx[i];
    }

    double P[3] = {Pp[0] + cam[3], Pp[1] + cam[4], Pp[2] + cam[5]};
    double u = -P[0] / P[2], v = -P[1] / P[2];
    double r2 = u * u + v * v;
    double f = cam[6], k1 = cam[7], k2 = cam[8];
    double dist = 1.0 + k1 * r2 + k2 * r2 * r2;
    double L = k1 + 2.0 * k2 * r2;  // d(dist)/d(r2)

    // D = d(pred)/d(u,v), symmetric 2x2
    double D00 = f * (dist + 2.0 * L * u * u);
    double D01 = 2.0 * f * L * u * v;
    double D11 = f * (dist + 2.0 * L * v * v);
    // M = d(u,v)/dP (2x3); DM = D * M (2x3)
    double iz = 1.0 / P[2];
    double M[6] = {-iz, 0.0, P[0] * iz * iz, 0.0, -iz, P[1] * iz * iz};
    double DM[6];
    for (int j = 0; j < 3; ++j) {
        DM[j] = D00 * M[j] + D01 * M[3 + j];
        DM[3 + j] = D01 * M[j] + D11 * M[3 + j];
    }

    // Jc columns: rodrigues (DM * Dra), t (DM), f, k1, k2.
    for (int row = 0; row < 2; ++row) {
        for (int j = 0; j < 3; ++j) {
            double s = 0.0;
            for (int k = 0; k < 3; ++k) s += DM[row * 3 + k] * Dra[k * 3 + j];
            Jc[row * 9 + j] = s;
        }
        for (int j = 0; j < 3; ++j) Jc[row * 9 + 3 + j] = DM[row * 3 + j];
        double p = row == 0 ? u : v;
        Jc[row * 9 + 6] = dist * p;
        Jc[row * 9 + 7] = f * r2 * p;
        Jc[row * 9 + 8] = f * r2 * r2 * p;
    }
    // Jp = DM * R
    for (int row = 0; row < 2; ++row)
        for (int j = 0; j < 3; ++j) {
            double s = 0.0;
            for (int k = 0; k < 3; ++k) s += DM[row * 3 + k] * R[k * 3 + j];
            Jp[row * 3 + j] = s;
        }
}

__device__ inline double block_reduce(double v, double* shared) {
    shared[threadIdx.x] = v;
    __syncthreads();
    for (int s = kBlock / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
        __syncthreads();
    }
    return shared[0];
}

// Accumulates J^T J (U, V), J^T(-r) (bc, bp), per-edge W = Jc^T Jp, and the
// total squared error. U/V lower triangles only; mirrored on the host.
__global__ void edge_kernel(const double* cams, const double* pts, const int* obs_cam,
                            const int* obs_pt, const double* obs_px, int num_obs,
                            double* U, double* V, double* W, double* bc, double* bp,
                            double* sse, unsigned long long* counters) {
    __shared__ double s_reduce[kBlock];
    __shared__ unsigned long long s_anom[kBlock];
    double local_sse = 0.0;
    unsigned long long local_anom = 0;
    int stride = gridDim.x * blockDim.x;
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < num_obs; e += stride) {
        const double* cam = cams + 9 * obs_cam[e];
        const double* X = pts + 3 * obs_pt[e];
        double r[2], Jc[18], Jp[6];
        edge_residual_jacobian(cam, X, obs_px + 2 * e, r, Jc, Jp);
        bool finite = isfinite(r[0]) && isfinite(r[1]);
        for (int j = 0; j < 18; ++j) finite = finite && isfinite(Jc[j]);
        for (int j = 0; j < 6; ++j) finite = finite && isfinite(Jp[j]);
        if (!finite) {
            ++local_anom;
            continue;
        }
        local_sse += r[0] * r[0] + r[1] * r[1];
        int c = obs_cam[e], p = obs_pt[e];
        double* We = W + 27 * e;
        for (int i = 0; i < 9; ++i)
            for (int j = 0; j < 3; ++j)
                We[i * 3 + j] = Jc[i] * Jp[j] + Jc[9 + i] * Jp[3 + j];
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j <= i; ++j)
                atomicAdd(&U[81 * c + i * 9 + j], Jc[i] * Jc[j] + Jc[9 + i] * Jc[9 + j]);
            atomicAdd(&bc[9 * c + i], -(Jc[i] * r[0] + Jc[9 + i] * r[1]));
        }
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j <= i; ++j)
                atomicAdd(&V[9 * p + i * 3 + j], Jp[i] * Jp[j] + Jp[3 + i] * Jp[3 + j]);
            atomicAdd(&bp[3 * p + i], -(Jp[i] * r[0] + Jp[3 + i] * r[1]));
        }
    }
    block_reduce(local_sse, s_reduce);
    if (threadIdx.x == 0) atomicAdd(sse, s_reduce[0]);
    s_anom[threadIdx.x] = local_anom;
    __syncthreads();
    for (int s = kBlock / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_anom[threadIdx.x] += s_anom[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0 && s_anom[0] > 0) atomicAdd(counters, s_anom[0]);
}

// Squared-error-only pass for trial steps.
__global__ void cost_kernel(const double* cams, const double* pts, const int* obs_cam,
                            const int* obs_pt, const double* obs_px, int num_obs,
                            double* sse, unsigned long long* counters) {
    __shared__ double s_reduce[kBlock];
    __shared__ unsigned long long s_anom[kBlock];
    double local_sse = 0.0;
    unsigned long long local_anom = 0;
    int stride = gridDim.x * blockDim.x;
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < num_obs; e += stride) {
        double r[2];
        edge_residual(cams + 9 * obs_cam[e], pts + 3 * obs_pt[e], obs_px + 2 * e, r);
        if (!isfinite(r[0]) || !isfinite(r[1])) {
            ++local_anom;
            continue;
        }
        local_sse += r[0] * r[0] + r[1] * r[1];
    }
    block_reduce(local_sse, s_reduce);
    if (threadIdx.x == 0) atomicAdd(sse, s_reduce[0]);
    s_anom[threadIdx.x] = local_anom;
    __syncthreads();
    for (int s = kBlock / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_anom[threadIdx.x] += s_anom[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0 && s_anom[0] > 0) atomicAdd(counters, s_anom[0]);
}

// V' = V + lambda * diag(V) per point block, inverted in place into Vinv.
// Non-SPD blocks increment counters[1] (the host escalates lambda).
__global__ void damp_invert_v_kernel(const double* V, double lambda, double* Vinv,
                                     int num_points, unsigned long long* counters) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= num_points) return;
    const double* Vp = V + 9 * p;
    double d0 = Vp[0] > 0.0 ? Vp[0] : 1e-12;
    double d1 = Vp[4] > 0.0 ? Vp[4] : 1e-12;
    double d2 = Vp[8] > 0.0 ? Vp[8] : 1e-12;
    double A[9] = {Vp[0] + lambda * d0, Vp[3], Vp[6],
                   Vp[3], Vp[4] + lambda * d1, Vp[7],
                   Vp[6], Vp[7], Vp[8] + lambda * d2};
    if (!dev::inverse3_spd(A, Vinv + 9 * p)) atomicAdd(&counters[1], 1ULL);
}

struct DeviceProblem {
    double* cams = nullptr;
    double* pts = nullptr;
    int* obs_cam = nullptr;
    int* obs_pt = nullptr;
    double* obs_px = nullptr;
    double* U = nullptr;    // 81 * C (lower triangle written)
    double* V = nullptr;    // 9 * P (lower triangle)
    double* Vinv = nullptr; // 9 * P (full)
    double* W = nullptr;    // 27 * E
    double* bc = nullptr;   // 9 * C
    double* bp = nullptr;   // 3 * P
    double* sse = nullptr;  // 1
    unsigned long long* counters = nullptr;  // [0] edge anomalies, [1] non-SPD V

    DeviceProblem(const BALProblem& p) {
        size_t C = p.num_cameras, P = p.num_points, E = p.num_observations;
        GBA_CUDA_CHECK(cudaMalloc(&cams, 9 * C * sizeof(double)));
        GBA_CUDA_CHECK(cudaMalloc(&pts, 3 * P * sizeof(double)));
        GBA_CUDA_CHECK(cudaMalloc(&obs_cam, E * sizeof(int)));
        GBA_CUDA_CHECK(cudaMalloc(&obs_pt, E * sizeof(int)));
        GBA_CUDA_CHECK(cudaMalloc(&obs_px, 2 * E * sizeof(double)));
        GBA_CUDA_CHECK(cudaMalloc(&U, 81 * C * sizeof(double)));
        GBA_CUDA_CHECK(cudaMalloc(&V, 9 * P * sizeof(double)));
        GBA_CUDA_CHECK(cudaMalloc(&Vinv, 9 * P * sizeof(double)));
        GBA_CUDA_CHECK(cudaMalloc(&W, 27 * E * sizeof(double)));
        GBA_CUDA_CHECK(cudaMalloc(&bc, 9 * C * sizeof(double)));
        GBA_CUDA_CHECK(cudaMalloc(&bp, 3 * P * sizeof(double)));
        GBA_CUDA_CHECK(cudaMalloc(&sse, sizeof(double)));
        GBA_CUDA_CHECK(cudaMalloc(&counters, 2 * sizeof(unsigned long long)));
        GBA_CUDA_CHECK(cudaMemcpy(obs_cam, p.obs_camera.data(), E * sizeof(int),
                                  cudaMemcpyHostToDevice));
        GBA_CUDA_CHECK(cudaMemcpy(obs_pt, p.obs_point.data(), E * sizeof(int),
                                  cudaMemcpyHostToDevice));
        GBA_CUDA_CHECK(cudaMemcpy(obs_px, p.obs_pixel.data(), 2 * E * sizeof(double),
                                  cudaMemcpyHostToDevice));
    }
    ~DeviceProblem() {
        cudaFree(cams);
        cudaFree(pts);
        cudaFree(obs_cam);
        cudaFree(obs_pt);
        cudaFree(obs_px);
        cudaFree(U);
        cudaFree(V);
        cudaFree(Vinv);
        cudaFree(W);
        cudaFree(bc);
        cudaFree(bp);
        cudaFree(sse);
        cudaFree(counters);
    }
};

int grid_for(int n) {
    int blocks = (n + kBlock - 1) / kBlock;
    return blocks < 1024 ? blocks : 1024;
}

}  // namespace

void edge_residual_jacobian_host(const double* camera9, const double* point3,
                                 const double* observed2, double* out_residual2,
                                 double* out_Jc, double* out_Jp) {
    edge_residual_jacobian(camera9, point3, observed2, out_residual2, out_Jc, out_Jp);
}

LMResult solve_bal_gpu(BALProblem& prob, const LMOptions& opts) {
    const int C = prob.num_cameras, P = prob.num_points, E = prob.num_observations;
    auto t0 = std::chrono::steady_clock::now();

    DeviceProblem d(prob);
    const int grid_e = grid_for(E), grid_p = grid_for(P);

    // Point -> observing edges adjacency, for the Schur assembly and the
    // point back-substitution.
    std::vector<std::vector<int>> point_edges(P);
    for (int e = 0; e < E; ++e) point_edges[prob.obs_point[e]].push_back(e);
    const int nthreads =
        std::max(1, std::min(C, static_cast<int>(std::thread::hardware_concurrency())));

    std::vector<double> h_U(81 * C), h_V(9 * P), h_Vinv(9 * P), h_W(27 * E);
    std::vector<double> h_bc(9 * C), h_bp(3 * P);
    std::vector<double> h_sse(1);
    std::vector<unsigned long long> h_counters(2);

    LMResult result;

    // Phase timers (verbose only): the memcpy calls already synchronize, so
    // wall-clock between phases is accurate.
    auto now = []() { return std::chrono::steady_clock::now(); };
    double t_accum = 0.0, t_dampinv = 0.0, t_assembly = 0.0, t_solve = 0.0,
           t_trial_cost = 0.0;

    // Full J^T J accumulation at the current (device) parameters.
    auto accumulate = [&]() {
        auto ta = now();
        GBA_CUDA_CHECK(cudaMemsetAsync(d.U, 0, 81 * C * sizeof(double)));
        GBA_CUDA_CHECK(cudaMemsetAsync(d.V, 0, 9 * P * sizeof(double)));
        GBA_CUDA_CHECK(cudaMemsetAsync(d.bc, 0, 9 * C * sizeof(double)));
        GBA_CUDA_CHECK(cudaMemsetAsync(d.bp, 0, 3 * P * sizeof(double)));
        GBA_CUDA_CHECK(cudaMemsetAsync(d.sse, 0, sizeof(double)));
        edge_kernel<<<grid_e, kBlock>>>(d.cams, d.pts, d.obs_cam, d.obs_pt, d.obs_px, E,
                                        d.U, d.V, d.W, d.bc, d.bp, d.sse, d.counters);
        GBA_CUDA_CHECK(cudaGetLastError());
        GBA_CUDA_CHECK(cudaMemcpy(h_U.data(), d.U, 81 * C * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        GBA_CUDA_CHECK(cudaMemcpy(h_V.data(), d.V, 9 * P * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        GBA_CUDA_CHECK(cudaMemcpy(h_W.data(), d.W, 27 * E * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        GBA_CUDA_CHECK(cudaMemcpy(h_bc.data(), d.bc, 9 * C * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        GBA_CUDA_CHECK(cudaMemcpy(h_bp.data(), d.bp, 3 * P * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        GBA_CUDA_CHECK(cudaMemcpy(h_sse.data(), d.sse, sizeof(double),
                                  cudaMemcpyDeviceToHost));
        t_accum += std::chrono::duration<double>(now() - ta).count();
    };

    GBA_CUDA_CHECK(cudaMemsetAsync(d.counters, 0, 2 * sizeof(unsigned long long)));
    GBA_CUDA_CHECK(cudaMemcpy(d.cams, prob.cameras.data(), 9 * C * sizeof(double),
                              cudaMemcpyHostToDevice));
    GBA_CUDA_CHECK(cudaMemcpy(d.pts, prob.points.data(), 3 * P * sizeof(double),
                              cudaMemcpyHostToDevice));
    accumulate();
    GBA_CUDA_CHECK(cudaDeviceSynchronize());

    double sse = h_sse[0];
    result.initial_cost = sse / 2.0;
    double lambda = opts.initial_lambda;

    std::vector<double> trial_cams = prob.cameras, trial_pts = prob.points;

    Eigen::MatrixXd S(9 * C, 9 * C);
    Eigen::VectorXd rhs(9 * C), dc(9 * C);
    std::vector<double> dp(3 * P);
    std::vector<double> Tbuf(27 * E);  // per-edge T_e = W_e V'^-1, 9x3 row-major
    // Privatized Schur accumulators: each thread accumulates a disjoint set
    // of points into its own S/rhs copy (hot in its own L2), then a merge
    // pass sums the copies. Shared-S accumulation measured ~5x slower due
    // to cross-core cache-line ping-pong on the 9x9 block RMWs.
    const int dim = 9 * C;
    // Cap privatized copies so the total stays cache-resident (~16 MB);
    // beyond that, private copies spill L2/L3 and the merge cost dominates.
    const int max_workers_by_mem =
        std::max(1, (16 << 20) / std::max(1, dim * dim * 8));
    const int nworkers = std::max(1, std::min({nthreads, P, max_workers_by_mem}));
    std::vector<double> S_priv(static_cast<size_t>(nworkers) * dim * dim);
    std::vector<double> rhs_priv(static_cast<size_t>(nworkers) * dim);

    bool converged = false, failed = false;
    double rel_decrease = 0.0;

    while (result.iterations < opts.max_iterations && !converged && !failed) {
        // Inner trust-region loop: escalate lambda until a step decreases
        // the cost with a positive gain ratio.
        double nu = 2.0;
        bool accepted = false;
        for (int trial = 0; trial < opts.max_inner_trials; ++trial) {
            GBA_CUDA_CHECK(cudaMemcpy(h_counters.data(), d.counters,
                                      2 * sizeof(unsigned long long),
                                      cudaMemcpyDeviceToHost));
            std::vector<unsigned long long> before = h_counters;
            auto ta = now();
            damp_invert_v_kernel<<<grid_p, kBlock>>>(d.V, lambda, d.Vinv, P, d.counters);
            GBA_CUDA_CHECK(cudaGetLastError());
            GBA_CUDA_CHECK(cudaMemcpy(h_Vinv.data(), d.Vinv, 9 * P * sizeof(double),
                                      cudaMemcpyDeviceToHost));
            GBA_CUDA_CHECK(cudaMemcpy(h_counters.data(), d.counters,
                                      2 * sizeof(unsigned long long),
                                      cudaMemcpyDeviceToHost));
            t_dampinv += std::chrono::duration<double>(now() - ta).count();
            if (h_counters[1] != before[1]) {  // non-SPD V block: raise damping
                lambda *= nu;
                nu *= 2.0;
                continue;
            }

            // S = U + lambda*diag(U) on the diagonal blocks, minus
            // W V'^-1 W^T accumulated per point over edge pairs. Raw pointer
            // loops: Eigen's small-product path measured ~20x slower here.
            // Only the lower block triangle is filled (pairs with ce >= cf);
            // LLT<Lower> reads just that half.
            ta = now();
            std::memset(S_priv.data(), 0, S_priv.size() * sizeof(double));
            std::memset(rhs_priv.data(), 0, rhs_priv.size() * sizeof(double));

            // Phase 1: T_e = W_e V'^-1 for all edges, parallel over ranges.
            {
                std::vector<std::thread> pool;
                const int chunk = (E + nthreads - 1) / nthreads;
                for (int t = 0; t < nthreads; ++t) {
                    const int e0 = t * chunk, e1 = std::min(E, e0 + chunk);
                    pool.emplace_back([&, e0, e1] {
                        for (int e = e0; e < e1; ++e) {
                            const double* We = h_W.data() + 27 * e;
                            const double* Vi = h_Vinv.data() + 9 * prob.obs_point[e];
                            double* Te = Tbuf.data() + 27 * e;
                            for (int i = 0; i < 9; ++i) {
                                const double w0 = We[3 * i], w1 = We[3 * i + 1],
                                             w2 = We[3 * i + 2];
                                Te[3 * i] = w0 * Vi[0] + w1 * Vi[3] + w2 * Vi[6];
                                Te[3 * i + 1] = w0 * Vi[1] + w1 * Vi[4] + w2 * Vi[7];
                                Te[3 * i + 2] = w0 * Vi[2] + w1 * Vi[5] + w2 * Vi[8];
                            }
                        }
                    });
                }
                for (auto& th : pool) th.join();
            }
            // Phase 2: each worker accumulates a strided subset of points
            // into its private S/rhs copy — no shared writes, no coherence
            // traffic on the 9x9 block RMWs.
            {
                std::vector<std::thread> pool;
                for (int t = 0; t < nworkers; ++t) {
                    pool.emplace_back([&, t] {
                        double* Sp = S_priv.data() + static_cast<size_t>(t) * dim * dim;
                        double* rp = rhs_priv.data() + static_cast<size_t>(t) * dim;
                        for (int k = t; k < P; k += nworkers) {
                            const auto& edges = point_edges[k];
                            if (edges.empty()) continue;
                            const double bp0 = h_bp[3 * k], bp1 = h_bp[3 * k + 1],
                                         bp2 = h_bp[3 * k + 2];
                            for (int e : edges) {
                                const int ce = prob.obs_camera[e];
                                const double* Te = Tbuf.data() + 27 * e;
                                double* rseg = rp + 9 * ce;
                                for (int i = 0; i < 9; ++i)
                                    rseg[i] -= Te[3 * i] * bp0 + Te[3 * i + 1] * bp1 +
                                               Te[3 * i + 2] * bp2;
                                for (int f : edges) {
                                    const int cf = prob.obs_camera[f];
                                    if (cf > ce) continue;
                                    const double* Wf = h_W.data() + 27 * f;
                                    // S(9ce+i, 9cf+j) -= sum_l Te(i,l) Wf(j,l);
                                    // column-major S: fixed j -> contiguous i.
                                    for (int j = 0; j < 9; ++j) {
                                        const double w0 = Wf[3 * j],
                                                     w1 = Wf[3 * j + 1],
                                                     w2 = Wf[3 * j + 2];
                                        double* col =
                                            Sp + (9 * cf + j) * dim + 9 * ce;
                                        for (int i = 0; i < 9; ++i)
                                            col[i] -= Te[3 * i] * w0 +
                                                      Te[3 * i + 1] * w1 +
                                                      Te[3 * i + 2] * w2;
                                    }
                                }
                            }
                        }
                    });
                }
                for (auto& th : pool) th.join();
            }
            // Merge: S = sum of private copies + damped U diagonal blocks.
            {
                std::vector<std::thread> pool;
                const int total = dim * dim;
                const int chunk = (total + nthreads - 1) / nthreads;
                double* Spriv = S_priv.data();
                double* Sd = S.data();
                for (int t = 0; t < nthreads; ++t) {
                    const int i0 = t * chunk, i1 = std::min(total, i0 + chunk);
                    pool.emplace_back([&, i0, i1] {
                        for (int i = i0; i < i1; ++i) {
                            double s = 0.0;
                            for (int w = 0; w < nworkers; ++w)
                                s += Spriv[static_cast<size_t>(w) * total + i];
                            Sd[i] = s;
                        }
                    });
                }
                for (auto& th : pool) th.join();
            }
            for (int i = 0; i < dim; ++i) {
                double s = h_bc[i];
                for (int w = 0; w < nworkers; ++w) s += rhs_priv[w * dim + i];
                rhs[i] = s;
            }
            for (int c = 0; c < C; ++c) {
                for (int i = 0; i < 9; ++i) {
                    double diag = h_U[81 * c + i * 9 + i];
                    for (int j = 0; j <= i; ++j) {
                        double v = h_U[81 * c + i * 9 + j];
                        if (i == j) v += lambda * (diag > 0.0 ? diag : 1e-12);
                        S(9 * c + i, 9 * c + j) += v;
                    }
                }
            }

            t_assembly += std::chrono::duration<double>(now() - ta).count();
            ta = now();
            Eigen::LLT<Eigen::MatrixXd, Eigen::Lower> llt(S);
            if (llt.info() != Eigen::Success) {
                lambda *= nu;
                nu *= 2.0;
                continue;
            }
            dc = llt.solve(rhs);

            // Point back-substitution: dp = V'^-1 (bp - W^T dc).
            double denom = 0.0;  // delta^T (lambda D delta + b), for the gain ratio
            for (int c = 0; c < C; ++c)
                for (int i = 0; i < 9; ++i) {
                    double diag = h_U[81 * c + i * 9 + i];
                    double dci = dc[9 * c + i];
                    denom += dci * (lambda * (diag > 0.0 ? diag : 1e-12) * dci +
                                    h_bc[9 * c + i]);
                }
            for (int k = 0; k < P; ++k) {
                double m0 = h_bp[3 * k], m1 = h_bp[3 * k + 1], m2 = h_bp[3 * k + 2];
                for (int e : point_edges[k]) {
                    const double* We = h_W.data() + 27 * e;
                    const double* dce = dc.data() + 9 * prob.obs_camera[e];
                    for (int i = 0; i < 9; ++i) {
                        m0 -= We[3 * i] * dce[i];
                        m1 -= We[3 * i + 1] * dce[i];
                        m2 -= We[3 * i + 2] * dce[i];
                    }
                }
                const double* Vi = h_Vinv.data() + 9 * k;
                const double dp0 = Vi[0] * m0 + Vi[1] * m1 + Vi[2] * m2;
                const double dp1 = Vi[3] * m0 + Vi[4] * m1 + Vi[5] * m2;
                const double dp2 = Vi[6] * m0 + Vi[7] * m1 + Vi[8] * m2;
                dp[3 * k] = dp0;
                dp[3 * k + 1] = dp1;
                dp[3 * k + 2] = dp2;
                const double di0 = h_V[9 * k], di1 = h_V[9 * k + 4],
                             di2 = h_V[9 * k + 8];
                denom += dp0 * (lambda * (di0 > 0.0 ? di0 : 1e-12) * dp0 + h_bp[3 * k]) +
                         dp1 * (lambda * (di1 > 0.0 ? di1 : 1e-12) * dp1 + h_bp[3 * k + 1]) +
                         dp2 * (lambda * (di2 > 0.0 ? di2 : 1e-12) * dp2 + h_bp[3 * k + 2]);
            }
            if (!(denom > 0.0)) {
                lambda *= nu;
                nu *= 2.0;
                continue;
            }
            t_solve += std::chrono::duration<double>(now() - ta).count();
            ta = now();

            for (size_t i = 0; i < trial_cams.size(); ++i)
                trial_cams[i] = prob.cameras[i] + dc[static_cast<Eigen::Index>(i)];
            for (size_t i = 0; i < trial_pts.size(); ++i)
                trial_pts[i] = prob.points[i] + dp[i];
            GBA_CUDA_CHECK(cudaMemcpy(d.cams, trial_cams.data(), 9 * C * sizeof(double),
                                      cudaMemcpyHostToDevice));
            GBA_CUDA_CHECK(cudaMemcpy(d.pts, trial_pts.data(), 3 * P * sizeof(double),
                                      cudaMemcpyHostToDevice));
            GBA_CUDA_CHECK(cudaMemsetAsync(d.sse, 0, sizeof(double)));
            cost_kernel<<<grid_e, kBlock>>>(d.cams, d.pts, d.obs_cam, d.obs_pt,
                                            d.obs_px, E, d.sse, d.counters);
            GBA_CUDA_CHECK(cudaGetLastError());
            GBA_CUDA_CHECK(cudaMemcpy(h_sse.data(), d.sse, sizeof(double),
                                      cudaMemcpyDeviceToHost));
            GBA_CUDA_CHECK(cudaMemcpy(h_counters.data(), d.counters,
                                      2 * sizeof(unsigned long long),
                                      cudaMemcpyDeviceToHost));
            double trial_sse = h_sse[0];
            t_trial_cost += std::chrono::duration<double>(now() - ta).count();
            if (h_counters[0] != before[0] || !std::isfinite(trial_sse)) {
                lambda *= nu;  // non-finite trial: reject
                nu *= 2.0;
                continue;
            }

            double rho = (sse - trial_sse) / (0.5 * denom);
            if (opts.verbose)
                std::fprintf(stderr,
                             "  iter %d trial: lambda %.3g cost %.9g rho %.3g %s\n",
                             result.iterations, lambda, trial_sse / 2.0, rho,
                             rho > 0.0 ? "accept" : "reject");
            if (rho > 0.0 && trial_sse < sse) {
                prob.cameras = trial_cams;
                prob.points = trial_pts;
                rel_decrease = (sse - trial_sse) / std::max(sse, 1.0);
                sse = trial_sse;
                lambda *= std::max(1.0 / 3.0, 1.0 - std::pow(2.0 * rho - 1.0, 3.0));
                accepted = true;
                break;
            }
            lambda *= nu;
            nu *= 2.0;
        }

        if (!accepted) {
            failed = true;
            break;
        }
        ++result.iterations;
        if (rel_decrease < opts.rel_decrease_tol) {
            converged = true;
            break;
        }
        // Re-accumulate J^T J at the accepted parameters.
        accumulate();
        GBA_CUDA_CHECK(cudaDeviceSynchronize());
    }

    result.final_cost = sse / 2.0;
    GBA_CUDA_CHECK(cudaMemcpy(h_counters.data(), d.counters,
                              2 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    result.anomalies = h_counters[0];
    if (converged)
        result.termination = "CONVERGENCE";
    else if (failed)
        result.termination = "FAILURE";
    else
        result.termination = "MAX_ITERATIONS";

    auto t1 = std::chrono::steady_clock::now();
    result.total_time_s = std::chrono::duration<double>(t1 - t0).count();
    if (opts.verbose)
        std::fprintf(stderr,
                     "  phases: edge-accumulate %.3fs, damp+Vinv %.3fs, "
                     "S assembly %.3fs, LLT+backsub %.3fs, trial cost %.3fs\n",
                     t_accum, t_dampinv, t_assembly, t_solve, t_trial_cost);
    return result;
}

}  // namespace gba
