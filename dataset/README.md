# Repository benchmark dataset

The repository contains **23 source images** under `dataset/image`, originating from the supplied CVPR 2016 dark-channel deblurring release.

Docker benchmark services use this checked-in data directly. The complete `dataset/` directory is mounted read-only at `/work/dataset`; no network download or temporary `testdata/upstream` copy is required.

Run the 3-image smoke benchmark with:

```bash
docker compose run --rm benchmark-preview
```

Run the full native-resolution 23-image benchmark with:

```bash
docker compose run --rm benchmark
```

For each source image the benchmark performs an independent MATLAB-parity blind deblur and a robust blind dark-channel PSF estimation, then evaluates the robust baseline plus the Annealed PnP and Dual-Extreme refinements. The two refinement methods reuse the robust baseline PSF so their restoration priors are compared under the same estimated blur model.

Generated outputs are written to `report-preview/` or `report/`; the committed source dataset is never modified. Other folders under `dataset/` are preserved as historical/reference assets and remain available to benchmark and validation code through the read-only dataset mount.
