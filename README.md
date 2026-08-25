# fast-deblur — C++20 Adaptive Blind Deblurring

High-performance C++20 implementation of the algorithms maintained in [`vtavakkoli/AdaptiveBlindDeblur`](https://github.com/vtavakkoli/AdaptiveBlindDeblur), including a strict MATLAB-parity path and the newer guarded Python methods.

The goal is **speed without hiding algorithmic differences**: the legacy-compatible baseline, robust baseline, Annealed PnP refinement, and Dual-Extreme refinement are separate modes and separate report columns.

## What is implemented

| Mode | CLI | Description |
|---|---|---|
| MATLAB-parity baseline | `baseline --matlab-parity` | Audited legacy math: sequential dark channel, Liu boundary wrapping, Levin pyramid, Cho FFT sizing, legacy gradient/PSF details, TV/L0 or Whyte RL final restoration |
| Robust C++ baseline | `baseline` | Same blind core plus retained weak PSF structure, structural cleanup, kernel retry and artifact guards |
| Annealed PnP | `annealed-pnp` | NLM prior + Gaussian annealing + FFT blur-consistency projection + artifact-safe candidate selection |
| Dual Extreme | `extreme-channel` | Dark/bright local-extrema confidence + detail recovery + FFT consistency + artifact guard |

See [`docs/METHODS.md`](docs/METHODS.md) for the numerical mapping.

## Performance-oriented design

- C++20, no Python runtime in production or benchmark execution.
- OpenCV native FFT/DFT, filtering, connected components, NLM and image I/O.
- FFTW DST-I for the Liu minimum-Laplacian boundary solver.
- Parallel target discovery for dark-channel minima, while the mathematically order-dependent assignment remains sequential by design.
- Native-resolution processing; no hidden benchmark resize/crop.
- `Release` builds use `-O3`. `-ffast-math` is intentionally **not** enabled because numerical precision/parity matters.
- Optional `-DFAST_DEBLUR_NATIVE=ON` enables `-march=native` for a local CPU build.

## Build locally

Requirements: CMake 3.22+, C++20 compiler, OpenCV 4.x development files, FFTW3 and nlohmann-json.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFAST_DEBLUR_NATIVE=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## Docker Compose

Unit/integration tests:

```bash
docker compose build test
docker compose run --rm test
```

Three-image report smoke test:

```bash
docker compose run --rm benchmark-preview
```

Full native-resolution 23-image benchmark:

```bash
docker compose run --rm benchmark
```

The full report is written to `report/index.html`, with raw metrics in `report/metrics.csv` and generated images under `report/images/`.

The test-data fetcher is pinned to the inspected upstream AdaptiveBlindDeblur commit, downloads all 23 observed benchmark images plus the compact historical MATLAB/Python regression assets, and records the source commit in `testdata/upstream/UPSTREAM_COMMIT.txt`.

## CLI

```bash
./build/fast-deblur \
  --input blurry.png \
  --output restored.png \
  --kernel-output estimated_kernel.png \
  --method baseline \
  --kernel-size 65
```

Strict translated legacy path:

```bash
./build/fast-deblur --input blurry.png --output legacy_cpp.png \
  --method baseline --kernel-size 65 --matlab-parity
```

Refinement example:

```bash
./build/fast-deblur --input blurry.png --output pnp.png \
  --method annealed-pnp --kernel-size 65
```

Use `--fast` only for previews/tests. It caps iteration counts and is intentionally not the quality benchmark configuration.

## HTML benchmark report

`fast-deblur-benchmark` uses `benchmark_profiles.json`, matching the 23 image-specific supports/settings from the upstream benchmark. For every image it generates side-by-side outputs for strict MATLAB-parity C++, robust C++ baseline, Annealed PnP using the robust PSF, and Dual-Extreme using the same PSF.

The report includes runtime, reblur RMSE and reference-free artifact diagnostics. Refinement runtime is the incremental refinement time; the baseline PSF-estimation time is shown separately in the robust-baseline column.

## Reproducibility and interpretation

The benchmark does **not** feed legacy MATLAB/Python result pixels or kernels into inference. They are evaluation/history assets only. Blind deblurring has no unique solution in general, so visual sharpness alone is not treated as correctness; reblur consistency and artifact-growth diagnostics are shown together.

This is experimental research software. For publication-quality comparisons, run the full benchmark on fixed hardware and preserve `metrics.csv`, the HTML report, compiler/container metadata, and the pinned upstream commit.
