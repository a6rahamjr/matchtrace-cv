# 0001: Use Generated Data For The Default Example

**Status:** Accepted

## Context

The repository needs an example that runs in CI and on a new developer machine.
Bundling broadcast footage, cached detections, or third-party weights creates
licensing and reproducibility problems.

## Decision

Generate a small two-team dataset and match video from a fixed seed. Keep the
real-video detector behind an optional adapter.

## Consequences

- Tests run without network access or private assets.
- Generated accuracy measures integration, not real-match performance.
- Real deployments must supply licensed data and an approved detector.

