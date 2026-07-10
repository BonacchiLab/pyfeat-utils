# pyfeat-utils Package Cleanup Design

## Context

`pyfeat-utils` is intended to provide a reproducible CLI/toolkit for batch py-feat processing of images and videos, plus standardized descriptive analysis of resulting facial emotion and Action Unit outputs.

The current repository is small and script-oriented. `pyfeat_processor.py` performs batch image/video detection and writes CSVs. `descriptive_statistics.py` reads generated CSV/TXT files and prints plots/statistics. `template_config.json` stores default data paths and processing settings.

The package configuration needs modernization because py-feat's installation guidance has changed. The official py-feat installation page now says py-feat requires Python 3.11+, with 3.11, 3.12, and 3.13 tested, and recommends `uv venv --python 3.13` followed by `uv pip install py-feat`.

Local investigation found:

- `.python-version` and `pyproject.toml` already have uncommitted edits moving from Python 3.9 to Python 3.13.
- `uv lock --check` succeeds outside the sandbox.
- `uv sync --dry-run` succeeds outside the sandbox and would install the current lock.
- The first local uv failure was caused by a broken default uv cache path at `C:\Users\nico\AppData\Local\uv\cache`.
- Sandbox restrictions also blocked uv from inspecting managed Python directories and executing the existing `.venv` interpreter until commands were escalated.
- The lock currently resolves `py-feat==0.6.2`, which should be verified against the intended package source before finalizing dependency policy.

## Goal

Make the project a clean, maintainable Python package first, while also repairing the uv/Python setup enough that future reliability and feature work has a stable base.

Priority order:

1. Clean Python package structure, dependency management, CLI, tests, and maintainability.
2. Reliable real-world batch processing and analysis workflow.
3. Later feature growth such as richer reports, dashboards, batch configs, and annotation workflows.

## Chosen Approach

Use a package-first cleanup.

This avoids doing only a minimal environment patch and avoids mixing feature expansion into the same step. The cleanup should make package metadata, command entry points, config loading, file discovery, processing orchestration, and statistics code testable before expanding user-facing features.

## Dependency And Environment Policy

`pyproject.toml` should become the primary source of truth for package metadata and dependencies.

Recommended Python support:

- Use `requires-python = ">=3.11"` to match py-feat's documented supported range.
- Keep `.python-version` at `3.13` for this local repository so uv creates a current development environment by default.

Recommended dependency handling:

- Keep `py-feat` as a normal runtime dependency.
- Remove `pip` from runtime dependencies unless there is a concrete runtime reason for importing or invoking it.
- Keep development tools in a dev dependency group: `pytest`, `ruff`, and `ipykernel`.
- Decide during implementation whether to keep generated `requirements*.txt` files. If kept, document them as exports from uv rather than hand-maintained sources.
- Regenerate `uv.lock` after metadata changes using a workspace-local cache workaround when needed, for example `UV_CACHE_DIR=.uv-cache`.

The implementation should verify whether the intended source is the current PyPI `py-feat` release or the GitHub development install path documented by py-feat. The default should be PyPI unless there is a clear incompatibility or required unreleased feature.

## CLI Design

Add package entry points instead of asking users to run module files directly.

Recommended commands:

- `pyfeat-utils init`: create a user data/config area and write a starter config.
- `pyfeat-utils process`: process configured image/video inputs and write prediction CSVs.
- `pyfeat-utils stats`: read py-feat outputs and generate descriptive statistics.

The CLI should be thin. It should parse arguments, load config, call testable library functions, and report results. Heavy model imports should happen only inside processing execution paths, not at top-level import or help-command time.

## Internal Structure

Split current scripts into focused modules with stable boundaries:

- `config.py`: config schema/defaults, config file loading, path expansion, validation.
- `files.py`: input discovery, media type filtering, output path construction.
- `processing.py`: detector creation, image/video processing, CSV writing.
- `statistics.py`: loading generated outputs, emotion/AU summaries, plot/report preparation.
- `cli.py`: command-line interface and exit behavior.

Each module should expose functions that can be tested without requiring py-feat model downloads unless the test explicitly opts into integration behavior.

## Config And Data Flow

The package should stop treating the package's `template_config.json` as the active user config. Instead:

1. `pyfeat-utils init` creates a user-facing data directory and config file.
2. Commands accept an explicit `--config` path.
3. If no config is supplied, commands look for a documented default config location.
4. Paths in config are expanded consistently, including `~`.
5. Processing commands discover supported files, skip generated outputs, and write deterministic output filenames.

This keeps installed package data read-only and makes the user's working configuration explicit.

## Error Handling

Commands should fail with actionable messages for:

- Missing config file.
- Invalid config keys or unsupported process types.
- Missing input directory.
- Empty input directory or no matching files.
- Missing expected py-feat output columns during statistics.
- py-feat import/model initialization failures.

Errors should avoid raw stack traces for expected user problems. Unexpected exceptions can still surface during development.

## Testing Strategy

Add focused unit tests before broad integration tests:

- Config loading and path expansion.
- Default config generation.
- File discovery for images/videos/CSVs.
- Output filename construction.
- Statistics calculations from tiny fixture CSVs.
- CLI smoke tests for `--help` and invalid/missing config behavior.

Avoid requiring py-feat model downloads in ordinary tests. Add an optional integration test marker later for real py-feat processing when model availability and runtime cost are acceptable.

## Documentation Updates

Update the README to match the current packaging workflow:

- State that py-feat requires Python 3.11+ and that this repo develops against Python 3.13.
- Recommend `uv sync` for this project.
- Include a fallback/manual setup based on py-feat's official installation page.
- Explain the CLI commands rather than direct module invocation.
- Add a troubleshooting note for uv cache issues on Windows, using a project-local cache:

```powershell
$env:UV_CACHE_DIR = ".\.uv-cache"
uv sync
```

## Verification

The cleanup is complete when:

- `uv lock --check` passes.
- `uv sync --dry-run` passes.
- CLI help runs without importing or initializing py-feat models.
- Unit tests pass without model downloads.
- README commands match the implemented CLI.
- Existing user behavior is preserved at the workflow level: process media files, produce CSVs, and compute descriptive summaries.

## Out Of Scope For This Step

- Rich report generation.
- Dashboards.
- Annotation interfaces.
- New py-feat analysis features beyond preserving current image/video processing and descriptive statistics behavior.
- GPU/device optimization beyond keeping the design compatible with future `device` configuration.
