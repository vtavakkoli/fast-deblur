# Methods

`fast-deblur` deliberately exposes distinct numerical modes instead of hiding them behind one label. This makes parity, robustness, and refinement changes measurable and reviewable.

## 1. Robust blind baseline

CLI:

```bash
fast-deblur --method baseline ...
```

The baseline performs coarse-to-fine blind kernel estimation and final restoration. At each pyramid scale it alternates between a latent-image update and a PSF update.

Main stages:

1. RGB input is converted to MATLAB-compatible grayscale coefficients.
2. A Levin-style scale pyramid is built for the requested PSF support.
3. The image boundary is wrapped for the blind optimization domain.
4. The latent image is updated with dark-channel + gradient regularization, or gradient-only regularization when `lambda_dark == 0`.
5. Latent gradients are thresholded and used to estimate the PSF.
6. The PSF is pruned, normalized, centered, and propagated to the next scale.
7. Robust mode carries weaker PSF support through the pyramid, then applies structural PSF cleanup.
8. Final restoration is evaluated with reference-free artifact diagnostics.

When a result is suspicious, robust mode can automatically evaluate more conservative TV/L0 restoration settings. Long-support solutions can also trigger an independent gradient-only PSF retry. The retry is accepted only when reference-free quality and artifact checks justify it.

For saturated scenes, the implementation uses the Whyte-style restoration path and evaluates earlier checkpoints when late iterations amplify noise or halos.

## 2. MATLAB-parity baseline

CLI:

```bash
fast-deblur --method baseline --matlab-parity ...
```

Parity mode keeps the translated legacy behavior separate from later robustness heuristics. `DeblurConfig::matlabParity()` disables robust selection, PSF retry, and conservative restoration selection.

Use this mode for:

- numerical regression against the translated legacy implementation;
- investigating a discrepancy between C++ and historical MATLAB behavior;
- measuring the effect of later robust heuristics independently.

Do not use parity mode as shorthand for “best perceptual result.” Its purpose is fidelity to the audited legacy path.

## 3. Annealed PnP refinement

CLI:

```bash
fast-deblur --method annealed-pnp ...
```

Annealed PnP starts from the robust baseline and the same independently estimated PSF. It applies an annealed image prior and projects candidate estimates back toward blur consistency.

The C++ path uses:

- Gaussian perturbation/annealing;
- OpenCV non-local means as the image prior;
- FFT-domain blur-consistency projection;
- reference-free artifact-aware candidate acceptance.

A refinement is not accepted merely because it is sharper. Reblur error and artifact growth are considered together.

## 4. Dual-Extreme refinement

CLI:

```bash
fast-deblur --method extreme-channel ...
```

Dual-Extreme also starts from the robust baseline and uses the same PSF. It derives confidence from local dark and bright extrema, applies restrained detail recovery, and projects the prior back toward the observed blur model.

The final artifact guard prevents a small consistency improvement from buying excessive ringing, clipping, or high-frequency amplification.

## Reference-free quality diagnostics

The repository treats blind deblurring as an ill-posed inverse problem. No clean image is assumed to be available during inference.

The core diagnostics compare the observed image with a candidate restoration using quantities such as:

- **reblur consistency** — does `PSF * restored` reproduce the observed image?
- **edge growth** — did deconvolution amplify gradients implausibly?
- **noise growth** — did local high-frequency noise increase excessively?
- **high-pass growth** — is the candidate dominated by oscillatory/ringing structure?
- **clipping growth** — did restoration create excessive zeros/ones?

These metrics are used as safety signals, not as a claim that one scalar score fully represents perceptual quality.

## PSF behavior in robust mode

Robust PSF estimation differs from strict parity in two important ways:

- weaker kernel support is retained during the multi-scale estimation so curved/compound motion is less likely to be destroyed prematurely;
- the final PSF is passed through structural cleanup that suppresses implausible weak branches while preserving meaningful connected support.

The goal is to improve physical plausibility without using historical/legacy kernels as inference inputs.

## Choosing a mode

| Goal | Recommended mode |
|---|---|
| Best default research result | Robust baseline |
| Legacy numerical comparison | MATLAB parity |
| Denoising/detail prior experiment | Annealed PnP |
| Dark/bright local-extrema detail experiment | Dual-Extreme |

For publication-quality comparisons, run all relevant modes with the same repository revision and benchmark profile rather than changing parameters interactively between methods.
