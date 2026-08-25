## Summary

<!-- What problem does this PR solve? -->

## Change type

- [ ] Numerical parity
- [ ] Robustness / artifact safety
- [ ] Performance optimization
- [ ] New method / algorithm
- [ ] Public API / packaging
- [ ] Benchmark / metrics
- [ ] Docker / CI
- [ ] Documentation

## Numerical impact

<!-- Required for src/, benchmark profile, or algorithm changes. Explain what mathematics/behavior changes and what intentionally stays unchanged. -->

## Validation

- [ ] `cmake --preset release`
- [ ] `cmake --build --preset release`
- [ ] `ctest --preset release`
- [ ] `docker compose build test`
- [ ] `docker compose run --rm test`
- [ ] `docker compose run --rm benchmark-preview` when numerical/report code changes

## Reproducibility and inference integrity

- [ ] No legacy/reference pixels or kernels are used as hidden inference inputs.
- [ ] No native-resolution benchmark resize/crop was introduced silently.
- [ ] New CLI/public API behavior is documented.
- [ ] New numerical behavior has regression coverage.

## Evidence

<!-- Add benchmark numbers, report artifact links, screenshots, or a short explanation of why they are not applicable. Avoid claiming speed/quality improvements without evidence. -->

## Known limitations

<!-- What should reviewers know before merging? -->
