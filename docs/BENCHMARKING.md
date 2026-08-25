# Benchmarking and Reproducibility

The benchmark is designed to make quality, runtime, and historical comparisons explicit without allowing reference assets to influence inference.

## Services

### Unit/integration tests

```bash
docker compose build test
docker compose run --rm test
```

### Preview benchmark

```bash
docker compose run --rm benchmark-preview
```

The preview is intentionally small and exists to validate the report pipeline quickly. It is not a publication-quality benchmark.

### Full benchmark

```bash
docker compose run --rm benchmark
```

The full service evaluates the checked-in native-resolution benchmark set and writes:

```text
report/
├── index.html
├── metrics.csv
└── images/
```

## Benchmark inputs

- `dataset/image/` contains the source blurry images used by the benchmark.
- `benchmark_profiles.json` contains explicit per-image configuration.
- Historical/reference assets under `dataset/` are available for evaluation and provenance but are not used to estimate a PSF or choose a restoration candidate.

The Docker Compose benchmark mount is read-only for `dataset/`.

## No hidden resize contract

The full benchmark is intended to run at source resolution. A performance optimization that changes the benchmark image dimensions must not be introduced silently.

If a future method intentionally evaluates a resized approximation, it should be reported as a separate mode with its own metrics rather than replacing the native-resolution result.

## Methods in the report

The report keeps conceptually different paths separate:

1. strict MATLAB-parity baseline;
2. robust C++ baseline;
3. Annealed PnP refinement using the robust PSF;
4. Dual-Extreme refinement using the same robust PSF.

This prevents a robust heuristic from being misreported as a parity improvement, and prevents refinement time from being confused with blind PSF-estimation time.

## Metrics

### Runtime

Runtime is useful only when hardware, thread count, compiler, build flags, and repository revision are controlled. Do not compare runtimes from different machines as if they were algorithmic speedups.

### Reblur RMSE

A restored candidate `x` and estimated PSF `k` are checked against the observed blurry image `y` through the forward model:

```text
k * x ≈ y
```

Reblur RMSE measures this consistency. Lower is better, but a low value alone does not guarantee a visually plausible latent image.

### Artifact diagnostics

Reference-free diagnostics track suspicious growth in:

- edges;
- noise;
- high-frequency energy;
- clipping.

These signals complement reblur consistency and help identify over-deconvolution or ripple artifacts.

### Historical/reference metrics

When compatible historical outputs exist, comparison metrics can be useful for regression/fidelity studies. They are not clean-image ground truth unless the underlying asset is explicitly documented as such.

## Publication-quality checklist

For a result intended for a paper, report, or external comparison:

- [ ] record the exact Git commit;
- [ ] use a Release build;
- [ ] keep `FAST_DEBLUR_NATIVE=OFF` unless all compared runs use the same machine and build policy;
- [ ] record CPU model and logical/physical core count;
- [ ] record compiler and version;
- [ ] record OpenCV and FFTW versions;
- [ ] record thread settings;
- [ ] preserve `benchmark_profiles.json`;
- [ ] run the full benchmark, not preview/`--fast` mode;
- [ ] archive `index.html`, `metrics.csv`, and generated images;
- [ ] inspect representative images visually in addition to aggregate metrics.

## CI policy

Pull requests run native CMake build/test/install validation and a Docker smoke benchmark. The full benchmark can also be triggered manually through GitHub Actions when an expensive end-to-end result is warranted.

A green preview CI run proves that the project builds, tests, and generates a report. It does not by itself prove that a numerical change improves the full dataset.

## Fair performance comparison

For before/after optimization measurements:

1. use the same input data and benchmark profiles;
2. use the same build type and compiler;
3. pin or report thread counts;
4. warm caches consistently, or explicitly report cold-start behavior;
5. run multiple repetitions when timing variance matters;
6. report median/dispersion rather than only the single fastest run;
7. verify output quality did not regress while optimizing runtime.

## Interpreting blind deblurring results

Blind deblurring is underdetermined. A candidate may look sharp while using an implausible PSF, or may reblur accurately while containing visible ringing. For that reason, the repository treats visual inspection, forward-model consistency, PSF structure, and artifact diagnostics as complementary evidence rather than declaring one metric a perceptual oracle.
