# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- pytest suite for the network modules: shared blocks, both regression networks
  (fc and conv heads), and the DDPM denoiser and diffusion process.
- GitHub Actions workflow running the tests on Python 3.11 and 3.12, plus a ruff lint job.
- ruff linting/formatting configuration and pre-commit hooks.
- `pyproject.toml` with dependencies and a `[dev]` extra.
- Pull request and issue templates, `LICENSE` (MIT), `CITATION.cff` and this changelog.
- Updated use_model.py to be able to run inferences with updated repo structure.
