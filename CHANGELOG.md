# Changelog

## 3.0.0 - 2026-06-08

### Changed

- Renamed the project to MatchTrace CV.
- Renamed the Python package and CLI command to `matchtrace`.
- Renamed the detector environment variable to `MATCHTRACE_MODEL_PATH`.
- Renamed the repository slug to `matchtrace-cv`.

Reinstall editable environments with `pip install -e .`.

## 2.0.0 - 2026-06-08

### Changed

- Reorganized project naming and package metadata before the final MatchTrace CV
  identity was selected.

## 1.1.0 - 2026-06-08

### Changed

- Replaced nearest-centroid matching with velocity prediction and combined
  distance/IoU association.
- Batched player crops before team-classifier inference.
- Weighted team votes by classifier confidence.
- Smoothed projected positions and ignored implausible movement jumps.
- Added possession hold and switch hysteresis.
- Added model, configuration, input, and output SHA-256 values to run summaries.
- Rejected overlapping input, output, and summary paths.
- Refactored video orchestration into reusable `AnalysisPaths` and
  `VideoAnalyzer` components.

### Tests

- Added coverage for short occlusions, batch inference, possession hold,
  movement jump rejection, artifact hashes, and path collisions.

## 1.0.0 - 2026-06-08

- Initial package layout, synthetic data generator, CNN training and evaluation,
  streaming inference, CLI, API, model export, and system tests.
