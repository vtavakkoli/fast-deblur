# Changelog

All notable repository-level changes are documented here. The project follows semantic versioning for public API and packaging changes where practical.

## [0.2.0] - 2026-08-25

### Added

- installable/exported CMake package target `fast_deblur::fast_deblur`;
- CMake release/native presets;
- downstream package-consumer validation;
- method, architecture, and benchmarking documentation;
- contribution, security, citation, issue, and pull-request metadata;
- Docker build-context exclusions;
- manual full-benchmark GitHub Actions workflow path.

### Changed

- CI now validates a native CMake build, tests, package installation, downstream `find_package`, and Docker benchmark smoke report;
- README redesigned as the project landing page with reproducibility and library usage guidance.

### Numerical behavior

- No numerical algorithm source is intentionally changed by the 0.2.0 repository-professionalization release.

## [0.1.0]

Initial C++20 implementation with MATLAB-parity baseline, robust baseline, Annealed PnP refinement, Dual-Extreme refinement, Docker Compose validation, and native-resolution benchmark reporting.
