# Security Policy

`fast-deblur` is experimental research software for image processing. It is not currently presented as a security-hardened service or safety-critical component.

## Supported versions

Security fixes are applied to the current `main` branch. Older commits, experimental branches, and generated benchmark artifacts are not maintained as separate supported releases.

## Reporting a vulnerability

Please **do not open a public issue with exploit details** for a vulnerability that could affect users of the CLI, library, Docker image, or build pipeline.

Preferred reporting path:

1. use GitHub private vulnerability reporting / Security Advisories for this repository when available;
2. otherwise contact the repository maintainer privately through the contact information on the GitHub profile.

Please include:

- affected commit/version;
- affected component;
- reproduction steps or proof of concept;
- expected impact;
- suggested mitigation if known.

## Scope

Examples of security-relevant reports include:

- memory-safety issues triggered by crafted image input;
- path traversal or unsafe file handling in CLI/report generation;
- dependency or container configuration vulnerabilities with practical impact;
- command execution or build-pipeline injection;
- denial-of-service behavior caused by malformed inputs when it is disproportionate to expected image-processing cost.

Pure image-quality regressions, benchmark disagreements, and numerical instability should be reported as normal bugs rather than security vulnerabilities.

## Responsible disclosure

Please allow reasonable time for investigation and remediation before public disclosure. Acknowledgment and coordinated disclosure can be arranged with the reporter when appropriate.
