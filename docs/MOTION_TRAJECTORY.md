# Motion-Trajectory-Prior PSF (MTP-PSF)

MTP-PSF is an experimental, training-free constraint for blind motion deblurring.

A conventional blind PSF update solves for a non-negative 2-D kernel that explains image gradients. MTP-PSF adds a physical prior: during one exposure, ordinary camera or object motion should usually form a mostly continuous path through displacement space rather than an arbitrary cloud of unrelated PSF pixels.

## Pipeline

1. Threshold low-amplitude PSF support and keep the connected component with the largest probability mass.
2. Estimate a weighted principal motion axis from the PSF mass.
3. Fit an ordered principal curve with iterative nearest-point assignment and second-order smoothing.
4. Project PSF probability mass onto that curve and rasterize a narrow finite-width trajectory tube.
5. Estimate trajectory confidence from path coverage and PSF anisotropy.
6. Fuse the original PSF with the trajectory projection using the confidence value.
7. Suppress off-trajectory mass, preserve the strongest core, normalize, recenter, and run the existing structural PSF cleanup.
8. Restore the image using the constrained PSF. If the result has ripple risk or a substantially worse reference-free restoration score, fall back to the robust baseline.

The method is different from generic kernel denoising: it tries to preserve longitudinal motion geometry while reducing branches, isolated blobs, and diffuse off-path energy.

## CLI

```bash
./build/release/fast-deblur \
  --input blurry.png \
  --output trajectory-restored.png \
  --kernel-output trajectory-kernel.png \
  --method trajectory \
  --kernel-size 65
```

No additional tuning parameters are exposed. Trajectory strength is selected internally from the estimated PSF geometry.

## C++ API

```cpp
fast_deblur::DeblurConfig config;
config.kernel_size = 65;
auto result = fast_deblur::deblurMotionTrajectory(image, config);
```

## Scientific status

MTP-PSF is a new experimental method in this repository. It is not a trained model, has no learned weights, and does not use legacy/reference kernels as inference inputs.

The prior is intentionally restrictive. Defocus, strongly spatially varying blur, multiple independent motions, or complex self-intersecting trajectories may violate its assumptions, so benchmark and visual validation remain necessary.
