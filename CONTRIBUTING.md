# Contributing to fast-deblur

Thank you for improving `fast-deblur`. This project values **numerical transparency, reproducibility, and measurable performance** over opaque changes that only look better on a few images.

## Development principles

Please keep these boundaries explicit:

1. **Parity fixes** should improve fidelity to the audited legacy path without silently enabling robust heuristics.
2. **Robustness changes** should remain reference-free during inference.
3. **Performance changes** should preserve output quality and intended mathematics unless the PR explicitly proposes an algorithmic change.
4. **Benchmark/reference assets** must not become hidden inference inputs.
5. **Native-resolution benchmark behavior** must not be changed by an undocumented resize or crop.

## Build and test

Ubuntu/Debian dependencies:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake ninja-build pkg-config \
  libopencv-dev libfftw3-dev nlohmann-json3-dev
```

Configure, build, and test:

```bash
cmake --preset release
cmake --build --preset release
ctest --preset release
```

Install validation:

```bash
cmake --install build/release --prefix /tmp/fast-deblur-install
```

Docker validation:

```bash
docker compose build test
docker compose run --rm test
docker compose run --rm benchmark-preview
```

## Pull request checklist

Before opening a PR:

- [ ] describe whether the change is parity, robustness, performance, API, benchmark, or documentation work;
- [ ] explain the numerical impact when touching `src/`;
- [ ] add or update regression tests for behavior changes;
- [ ] run CMake build + CTest;
- [ ] run the Docker test service;
- [ ] run the preview benchmark when report or numerical code changes;
- [ ] keep unrelated formatting/refactoring out of numerical PRs when possible;
- [ ] document new CLI or public API behavior;
- [ ] do not claim a quality/speed improvement without benchmark evidence.

## Code style

The project uses modern C++20 and enables the following GCC/Clang warnings on the core library:

```text
-Wall -Wextra -Wpedantic -Wconversion -Wshadow
```

Prefer:

- RAII and value semantics;
- small functions with explicit numerical intent;
- `const` where it improves reasoning;
- standard-library facilities over custom utility code;
- comments that explain *why* a numerical choice exists rather than restating the code.

Avoid enabling `-ffast-math` in the default project because it can undermine numerical parity and reproducibility.

## Tests

Tests belong under `tests/`. Numerical tests should use explicit tolerances and explain what regression they protect.

When fixing a reported image failure, prefer a general invariant (PSF structure, artifact bound, forward-model consistency) over filename-specific behavior.

## Benchmark changes

See [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md).

Changes to any of the following should be called out prominently in the PR:

- `benchmark_profiles.json`;
- dataset selection;
- image dimensions;
- method definitions;
- metric definitions;
- thread policy;
- compiler optimization flags.

## Commit and PR scope

Short, reviewable commits are preferred. A useful PR description includes:

- problem statement;
- algorithmic or engineering approach;
- files/components affected;
- test evidence;
- benchmark evidence when relevant;
- known limitations.

## Reporting bugs

For ordinary correctness or reproducibility bugs, open a GitHub issue with:

- input characteristics (do not upload confidential images);
- command line or API call;
- OS/compiler/OpenCV versions;
- exact repository commit;
- error output or minimal reproduction;
- whether the issue appears in Docker.

For security-sensitive issues, follow [`SECURITY.md`](SECURITY.md) instead of posting exploit details publicly.
