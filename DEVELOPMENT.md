# Development Notes

## Scope

This repository is a runnable reference implementation, not a validated
commercial match-analysis product. The synthetic example is useful for testing
the software path from data generation through video output. It does not measure
accuracy on broadcast footage.

## Design Decisions

- **Streaming video:** frames are read and written one at a time to keep memory
  use independent of video duration.
- **Small built-in tracker:** the motion/IoU tracker avoids another runtime
  dependency and is adequate for the generated example. Crowded real footage
  needs a stronger tracker or appearance model.
- **Synthetic team data:** this keeps tests reproducible and avoids shipping
  unlicensed media. Real deployment needs licensed, representative data.
- **TOML configuration:** Python 3.11 can read TOML without another parser
  dependency.
- **Synchronous API:** appropriate for local demonstrations. Long-running jobs
  should move to a worker queue in a service deployment.

Longer decision records are kept in [`docs/decisions`](docs/decisions).

## Known Limitations

- The default color detector only understands the generated demo palette.
- Pitch projection uses configured dimensions rather than automatic field-line
  calibration.
- Team classification supports two teams.
- The API does not implement uploads, authentication, queues, or job state.
- Synthetic metrics should not be compared with results from real match data.

## Provenance And Tooling

The implementation was developed with AI-assisted coding and then exercised
through automated tests, packaging checks, API startup checks, and visual output
review. Functional inspiration and attribution for the adjacent project are
recorded in [ATTRIBUTION.md](ATTRIBUTION.md).

Contributors should describe substantial generated or externally sourced code
in pull requests and preserve applicable attribution and licenses.
