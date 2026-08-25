# fast-deblur

[![C++ CI](https://github.com/vtavakkoli/fast-deblur/actions/workflows/ci.yml/badge.svg)](https://github.com/vtavakkoli/fast-deblur/actions/workflows/ci.yml)
![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-5C3EE8?logo=opencv)
![Docker](https://img.shields.io/badge/Docker-reproducible-2496ED?logo=docker)
![Status](https://img.shields.io/badge/status-experimental-orange)

**High-performance C++20 blind image deblurring with a MATLAB-parity path, robust PSF estimation, artifact-aware restoration, and reproducible native-resolution benchmarking.**

`fast-deblur` is the C++ implementation companion to [`vtavakkoli/AdaptiveBlindDeblur`](https://github.com/vtavakkoli/AdaptiveBlindDeblur). It is designed for experiments where runtime matters, but numerical differences must remain visible and auditable rather than being hidden behind a single opaque “best” mode.

> **Research status:** experimental. The repository is intended for reproducible research, algorithm engineering, and comparative evaluation. Blind deblurring is ill-posed; visual sharpness alone is not treated as correctness.

## Highlights

- **C++20 production path** — no Python runtime is required for CLI or benchmark execution.
- **MATLAB-parity mode** — preserves the translated legacy numerical path for regression and fidelity studies.
- **Robust blind baseline** — adds PSF structure retention, structural cleanup, retry logic, and artifact guards.
- **Two guarded refinements** — Annealed PnP and Dual-Extreme use the same independently estimated robust PSF.
- **Native-resolution benchmark** — no hidden resize or crop in the quality benchmark.
- **Reference-free diagnostics** — reblur consistency plus edge/noise/high-frequency/clipping growth.
- **Reproducible containers** — Docker Compose services for tests, preview benchmarking, and the full benchmark.
- **Reusable CMake library** — installable headers, library, CLI, and exported CMake package target.

## Methods

| Method | CLI | Purpose |
|---|---|---|
| Robust baseline | `--method baseline` | Multi-scale blind PSF estimation plus guarded final restoration |
| MATLAB-parity baseline | `--method baseline --matlab-parity` | Strict translated legacy path with robust selection/guards disabled |
| Annealed PnP | `--method annealed-pnp` | Gaussian annealing, NLM prior, FFT blur-consistency projection, artifact-safe acceptance |
| Dual-Extreme | `--method extreme-channel` | Dark/bright local-extrema guidance, detail recovery, FFT consistency, artifact guard |

See [`docs/METHODS.md`](docs/METHODS.md) for the numerical mapping and design rationale.

## Quick start

### Docker Compose

The quickest reproducible validation is:

```bash
docker compose build test
docker compose run --rm test
```

Generate a three-image smoke report:

```bash
docker compose run --rm benchmark-preview
```

Run the full native-resolution benchmark:

```bash
docker compose run --rm benchmark
```

Outputs are written to:

```text
report/
├── index.html
├── metrics.csv
└── images/
```

The benchmark consumes the checked-in `dataset/` read-only. Historical/reference assets are evaluation-only and are never supplied as inference inputs.

### Local CMake build

Requirements:

- CMake 3.22+
- C++20 compiler (GCC or Clang recommended)
- OpenCV 4.5+
- FFTW3
- nlohmann-json (benchmark executable)
- pkg-config
- Ninja recommended

Ubuntu/Debian dependencies:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake ninja-build pkg-config \
  libopencv-dev libfftw3-dev nlohmann-json3-dev
```

Using the provided presets:

```bash
cmake --preset release
cmake --build --preset release
ctest --preset release
```

For a machine-specific local build:

```bash
cmake --preset native
cmake --build --preset native
ctest --preset native
```

`native` enables `-march=native`; do not use it for portable binaries or cross-machine benchmark comparisons.

## CLI

Basic restoration:

```bash
./build/release/fast-deblur \
  --input blurry.png \
  --output restored.png \
  --kernel-output estimated-kernel.png \
  --method baseline \
  --kernel-size 65
```

Strict translated legacy path:

```bash
./build/release/fast-deblur \
  --input blurry.png \
  --output legacy-cpp.png \
  --method baseline \
  --kernel-size 65 \
  --matlab-parity
```

Refinement example:

```bash
./build/release/fast-deblur \
  --input blurry.png \
  --output pnp.png \
  --method annealed-pnp \
  --kernel-size 65
```

Useful options:

```text
--kernel-size N
--gamma X
--lambda-dark X
--lambda-grad X
--lambda-tv X
--lambda-l0 X
--weight-ring X
--saturated
--saturation-iterations N
--threads N
--fast
```

`--fast` is a preview/testing shortcut. It intentionally caps optimization work and is **not** the full-quality benchmark configuration.

## C++ library API

```cpp
#include <fast_deblur/fast_deblur.hpp>

fast_deblur::DeblurConfig config;
config.kernel_size = 65;
config.threads = 0;

auto result = fast_deblur::deblur(
    rgb_float_image,
    config,
    fast_deblur::Method::Baseline
);

// result.image  : restored CV_32FC3 image in [0, 1]
// result.kernel : normalized estimated PSF
// result.latent : last grayscale latent estimate
```

After installation, downstream CMake projects can use:

```cmake
find_package(fast_deblur CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE fast_deblur::fast_deblur)
```

Install locally with:

```bash
cmake --install build/release --prefix ./install
```

## Benchmark contract

`fast-deblur-benchmark` reads `benchmark_profiles.json` and evaluates the repository’s native-resolution benchmark set. The report separates:

1. strict MATLAB-parity C++ baseline;
2. robust C++ baseline;
3. Annealed PnP refinement;
4. Dual-Extreme refinement.

The report includes runtime, reblur RMSE, and reference-free artifact diagnostics. Reference/legacy pixels and kernels are used only for historical comparison where available; they are never used to choose a PSF or restoration candidate.

See [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md) for reproducibility rules and interpretation guidance.

## Architecture

```text
apps/                  CLI and benchmark executables
include/fast_deblur/   public C++ API
src/                   numerical implementation
tests/                 C++ unit/regression tests
dataset/               benchmark and historical assets
scripts/               data/reproducibility helpers
docs/                  method, architecture, benchmark documentation
```

The main numerical flow is:

```text
image
  ↓
coarse-to-fine blind PSF estimation
  ↓
latent/gradient optimization ↔ PSF update
  ↓
PSF cleanup / optional robust retry
  ↓
full-quality restoration
  ↓
optional PnP or Dual-Extreme refinement
  ↓
artifact guard + reblur diagnostics
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for module-level details.

## Performance philosophy

The project optimizes the implementation without deliberately changing the audited mathematics:

- Release builds use `-O3` on GCC/Clang.
- `-ffast-math` is intentionally not enabled because numerical fidelity matters.
- OpenMP is used when available.
- FFT/DFT-heavy operations use OpenCV and FFTW.
- `FAST_DEBLUR_NATIVE=ON` is explicitly opt-in.

Benchmark results should always include the repository revision, compiler/container metadata, and hardware used. Do not compare raw runtimes from different machines as if they were algorithmic speedups.

## Documentation

- [`docs/METHODS.md`](docs/METHODS.md) — algorithms and numerical modes
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — code and data flow
- [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md) — benchmark contract and interpretation
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — development workflow
- [`SECURITY.md`](SECURITY.md) — security reporting
- [`CITATION.cff`](CITATION.cff) — software citation metadata

## Reproducibility

For publication-quality experiments:

1. use a fixed repository commit;
2. preserve `benchmark_profiles.json`;
3. use the full benchmark, not `--fast` or preview mode;
4. record CPU, compiler, OpenCV, FFTW, container image, and thread count;
5. archive `report/index.html`, `report/metrics.csv`, and generated outputs;
6. inspect visual results together with reblur and artifact diagnostics.

## Contributing

Contributions are welcome when they preserve reproducibility and clearly identify mathematical changes. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request.

## Citation

If this software contributes to research, please cite the repository metadata in [`CITATION.cff`](CITATION.cff) and, when relevant, cite the original algorithms reproduced or translated by your experiment.

## Acknowledgment and lineage

This repository is the high-performance C++ companion to [`AdaptiveBlindDeblur`](https://github.com/vtavakkoli/AdaptiveBlindDeblur). The project intentionally separates **parity**, **robustness**, and **refinement** modes so improvements can be measured without rewriting history.
