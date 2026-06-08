# 0002: Stream Video Frames

**Status:** Accepted

## Context

Loading every frame before processing makes memory use proportional to video
duration and complicates long-running jobs.

## Decision

Read, analyze, render, and write one frame at a time. Temporal components keep
only their own bounded state.

## Consequences

- Memory use is roughly independent of duration.
- The pipeline cannot freely revisit earlier frames.
- Multi-pass algorithms must be implemented as explicit preprocessing stages.

