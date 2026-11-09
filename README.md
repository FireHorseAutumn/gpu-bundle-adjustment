# gpu-bundle-adjustment

GPU bundle adjustment at BAL scale. Per-observation edge kernels compute
residuals and analytic Jacobians in FP64 and accumulate into U/V/W blocks
with atomics; camera blocks invert with a batched 9x9 Cholesky written
in-repo. J is never materialized. The reduced camera system S is dense
(9C x 9C), assembled and factored on the host with Eigen LLT.

## Status

- Vertical slice complete: BAL reader, Ceres SPARSE_SCHUR oracle, GPU LM
  solver, benchmark harness, honest win/loss tables.
- CTest 3/3: batched Cholesky vs Eigen, cost/iteration gates vs the
  Ceres oracle, launch-boundary regression at P > 262144.
- FP8/FP4 problem-data storage measured and rejected; fp16 storage
  survives. Full study in `docs/RUNGS.md`.

## Headline numbers

vs Ceres SPARSE_SCHUR, same machine (RTX 5060 Laptop, FP64 throughout).
Small problems: interleaved best-of-7, unpinned (powersave governor;
absolute numbers move ~2x with CPU frequency state). Large problems:
median of 7 interleaved pairs, governor + platform profile pinned.

| dataset | cameras | Ceres s | gba-gpu-lm s | verdict |
|---|---:|---:|---:|---|
| problem-16 | 16 | 1.509 | 0.336 | GPU wins 4.5x (best) |
| ladybug-49 | 49 | 1.777 | 0.780 | GPU wins 2.3x (best) |
| trafalgar-257 | 257 | 17.654 | 30.646 | Ceres wins 1.74x (median) |
| dubrovnik-356 | 356 | 25.782 | 71.735 | Ceres wins 2.78x (median) |

The win is the edge kernel; the loss is host dense-S assembly + LLT,
60 to 80% of solver time, dominant from C=257. The GPU reaches the lower
final cost on both large problems: its LM trajectory is better, its host
linear algebra is slower.

## Honest losses

- The GPU loses at scale: 1.74x on Trafalgar-257, 2.78x on
  Dubrovnik-356, medians of 7 pinned pairs. The dense 9C x 9C host path
  was estimated to break near C~400; measured, it already loses at
  C=257.
- Every Trafalgar Ceres run hit NO_CONVERGENCE at 51 iterations, cost
  flat at 96656.0529, reproduced on a quiet machine: solver behavior,
  not contention.
- e4m3 problem-data storage lands at 245x the oracle final cost, e5m2
  at 922x, fp4 e2m1 never converges. fp16 survives at +0.2 to 1.5%.
- The small-problem table is unpinned; its medians (0.91 vs 2.23 on
  problem-16, 1.66 vs 2.24 on ladybug-49) are much closer than its
  bests.

## Build and test

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build
```

No sibling repos; everything comes from public sources: CUDA toolkit,
Eigen, Ceres, BAL. CMake prefers `g++-15` as the CUDA host compiler when
the default g++ is too new for nvcc, and points Ceres's FindBLAS at the
OpenBLAS runtime library when no BLAS dev package is installed.

Data: `data/fetch_bal.sh` (`smoke` | dataset names | `all` | URL).
Benchmarks: `build/bench_ceres` (oracle, JSON out) and `build/bench_bal`
(same schema; `--verbose` prints the phase breakdown). Raw runs land in
`data/`.

## Layout

- `include/gba/bal_problem.hpp`: SoA problem definition, Snavely
  residual.
- `include/gba/schur.cuh`: batched 9x9 Cholesky factor/solve + 3x3 SPD
  inverse; non-SPD fails loudly.
- `src/lm.cu`, `include/gba/lm.hpp`: the LM driver: edge kernel,
  Marquardt damping with Nielsen accept/reject, device anomaly counter.
- `src/bal_reader.cpp`: BAL .txt/.bz2 reader.
- `tools/`: `quantize_bal.cu` and `fp8_residual_probe.cu`, the
  storage-precision study.
- `tests/`, `benchmarks/`: gates and timing harnesses.

## Deeper reading

- `docs/RUNGS.md`: the full record: the precision experiment, both BAL
  tables with every caveat, the measured rejections, the fix path.
