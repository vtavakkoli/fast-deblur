#!/usr/bin/env bash
set -euo pipefail

DEST="${1:-testdata/upstream}"
UPSTREAM_REPO="vtavakkoli/AdaptiveBlindDeblur"
UPSTREAM_COMMIT="876ec51c1025e9a5cdca14e7b7a0e0546952d828"
BASE="https://raw.githubusercontent.com/${UPSTREAM_REPO}/${UPSTREAM_COMMIT}"
mkdir -p "${DEST}/dataset/image" "${DEST}/examples/real_img2/matlab_reference" "${DEST}/examples/real_img2/python"

images=(
  "26.blurred.jpg" "26.png" "7_patch_use.png" "IMG_0650_small_patch.png"
  "IMG_0664_small_patch.png" "IMG_1240_blur.png" "IMG_4355_small.png"
  "IMG_4548_small.png" "IMG_4561.JPG" "blurry_2_small.png" "blurry_7.png"
  "boat.jpg" "flower.jpg" "flower_blurred.png" "las_vegas_saturated.png"
  "my_test_car6.png" "postcard.png" "real_blur_img3.png" "real_img2.png"
  "real_leaffiltered.png" "summerhouse.jpg" "toy.png" "wall.png"
)

fetch() {
  local rel="$1" out="${DEST}/$1"
  mkdir -p "$(dirname "$out")"
  if [[ -s "$out" ]]; then return; fi
  echo "Fetching ${rel}"
  curl --fail --location --retry 4 --retry-all-errors --silent --show-error \
    "${BASE}/${rel}" --output "${out}.tmp"
  mv "${out}.tmp" "$out"
}

for name in "${images[@]}"; do fetch "dataset/image/${name}"; done
fetch "dataset/benchmark_profiles.json"
fetch "examples/real_img2/input_compare.jpg"
fetch "examples/real_img2/matlab_reference/kernel.png"
fetch "examples/real_img2/matlab_reference/result_compare.jpg"
fetch "examples/real_img2/python/full_kernel.png"
fetch "examples/real_img2/python/full_result_compare.jpg"
fetch "examples/real_img2/python/fast_kernel.png"
fetch "examples/real_img2/python/fast_result_compare.jpg"
fetch "examples/real_img2/metrics.json"
printf '%s\n' "$UPSTREAM_COMMIT" > "${DEST}/UPSTREAM_COMMIT.txt"
echo "Upstream test data ready at ${DEST} (${UPSTREAM_COMMIT})"
