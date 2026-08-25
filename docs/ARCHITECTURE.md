# Architecture

This document describes how the C++ implementation is organized and where numerical responsibilities live.

## Repository layout

```text
fast-deblur/
├── apps/
│   ├── fast-deblur.cpp       CLI
│   └── benchmark.cpp         HTML/CSV benchmark runner
├── include/fast_deblur/
│   ├── config.hpp            public configuration and method enum
│   └── fast_deblur.hpp       public library API
├── src/
│   ├── boundary.cpp          boundary handling
│   ├── dark_channel.cpp      dark-channel operations
│   ├── deblur.cpp            top-level blind pipeline and guarded restoration
│   ├── fft.cpp               FFT/PSF utility path
│   ├── kernel.cpp            gradient thresholding, PSF estimation and cleanup
│   ├── optimization.cpp      latent/final restoration optimization
│   ├── quality.cpp           reference-free artifact diagnostics
│   ├── refinement.cpp        PnP and Dual-Extreme refinements
│   └── saturation.cpp        saturation-aware restoration path
├── tests/                    C++ regression tests
├── dataset/                  benchmark/reference assets
├── benchmark_profiles.json   explicit benchmark configuration
└── docker-compose.yml        reproducible test/benchmark services
```

## Public API boundary

Public consumers should include only headers under `include/fast_deblur/`.

The principal API is:

```cpp
DeblurResult deblur(
    const cv::Mat& image,
    const DeblurConfig& config = {},
    Method method = Method::Baseline
);
```

`DeblurResult` returns:

- `image` — final restored float image clipped to `[0, 1]`;
- `kernel` — normalized estimated PSF;
- `latent` — last grayscale latent estimate from blind estimation;
- `restoration_path` — the selected robust/fallback path for observability.

Additional public helpers expose refinement, reblurring, artifact diagnostics, RMSE, PSNR, and SSIM.

## Blind deblurring data flow

```text
RGB image
   │
   ├─→ MATLAB-compatible grayscale conversion
   │
   ▼
coarse-to-fine pyramid
   │
   ▼
┌─────────────────────────────────────────────┐
│ per-scale blind optimization                │
│                                             │
│ observed gradients                          │
│       │                                     │
│       ▼                                     │
│ latent update ──→ thresholded gradients     │
│       ▲                    │                │
│       │                    ▼                │
│       └──────────── PSF estimation/pruning  │
└─────────────────────────────────────────────┘
   │
   ▼
PSF centering + robust structural cleanup
   │
   ▼
final restoration
   │
   ├─ suspicious? ─→ conservative / TV-safe candidate
   │
   ├─ saturated?  ─→ guarded Whyte-style checkpoint selection
   │
   └─ PSF suspect? ─→ independent gradient-only blind retry
   │
   ▼
optional refinement (PnP or Dual-Extreme)
   │
   ▼
artifact guard + final result
```

## Numerical layers

### Boundary and FFT layer

Boundary treatment and FFT sizing are isolated from the higher-level estimator so translated numerical behavior can be audited separately. FFT-heavy operations use OpenCV and FFTW where appropriate.

### Blind PSF estimation

`src/deblur.cpp` coordinates the scale pyramid and alternating latent/PSF updates. Robust mode changes support retention and post-estimation PSF validation without erasing the strict parity path.

### Final restoration

The configured restoration is treated as a candidate, not automatically as truth. Reference-free diagnostics can trigger conservative variants when ringing/noise/clipping growth is suspicious.

### Refinement layer

`src/refinement.cpp` implements optional post-baseline methods. Refinements share the already estimated robust PSF and are guarded by blur consistency and artifact checks.

### Quality layer

`src/quality.cpp` centralizes reference-free diagnostics. This prevents each method from inventing a different acceptance rule and keeps safety behavior reviewable.

## Parallelism

OpenMP is used when available. OpenCV thread control is exposed through `DeblurConfig::threads` and the CLI `--threads` option.

Some operations are mathematically order-dependent. The implementation does not parallelize such steps merely to increase throughput if doing so would change the audited result.

## Build targets

CMake defines:

- `fast_deblur` / `fast_deblur::fast_deblur` — reusable library;
- `fast-deblur` — CLI;
- `fast-deblur-benchmark` — report generator;
- `fast-deblur-tests` — regression test executable.

The CLI and benchmark targets can be disabled independently through CMake options. The installed package exports `fast_deblur::fast_deblur` for downstream CMake projects.

## Design principles

1. **Parity is a mode, not a moving target.**
2. **Robustness must be reference-free during inference.**
3. **Benchmark data must not silently become inference data.**
4. **Native-resolution quality runs must not hide resize/crop shortcuts.**
5. **Performance changes must preserve the intended mathematics unless explicitly documented.**
6. **A sharper result is not automatically a better blind-deblurring result.**
