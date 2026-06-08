# Original vs New Project Comparison

## Architecture Differences

| Area | Original project | MatchTrace CV |
| --- | --- | --- |
| Layout | Feature folders around one script | Installable `src/` package with data, model, training, evaluation, inference, utility, test, and API boundaries |
| Configuration | Sample-specific paths and constants | Validated TOML configuration and environment override |
| Video processing | Full video retained in memory | Streaming read-process-write pipeline |
| Interfaces | Script entry point | Unified CLI plus FastAPI service |
| Extensibility | Detector and analytics tightly coupled | Detector protocol and typed component contracts |

## ML Pipeline Improvements

- Adds an independently implemented CNN training loop from random
  initialization.
- Uses deterministic seeds and fixed train, validation, and test splits.
- Saves the best validation checkpoint with preprocessing metadata.
- Adds held-out loss, accuracy, and confusion-matrix evaluation.
- Adds batched CNN inference and confidence-weighted track voting.
- Adds motion-and-IoU association with short-occlusion recovery.
- Adds position smoothing, jump rejection, and possession hysteresis.
- Adds saliency explainability and TorchScript export.

The original custom detector may outperform the bundled synthetic detector on
real football footage. MatchTrace's detector abstraction is an architectural
improvement, not an unsupported claim that the demo detector is more accurate.

## Dataset Handling

The original workflow depends on external training data, a custom model, sample
media, and cached pickle outputs. The new default workflow generates its own
licensed-safe dataset and video from a seed, with explicit split boundaries and
ground-truth metadata.

Real production training still requires a licensed, representative dataset.

## Performance Improvements

- Streaming inference changes video memory growth from duration-dependent to
  approximately constant per frame.
- Team crops are classified in one frame-level batch and capped per track.
- The verified version 1.1 synthetic demo processes 640x360 video at 69.3 FPS
  on the verification CPU.
- The classifier reaches 100% accuracy on the generated held-out split.

These figures are not directly comparable to the original 1920x1080 broadcast
sample because resolution, detector backend, content, and hardware workload
differ. No real-footage accuracy improvement is claimed without a shared
benchmark.

## Feature Enhancements

- Reproducible synthetic dataset and demo generation
- REST API with path controls
- TorchScript model export
- Saliency-map explainability
- Structured JSON logging and run summaries
- SHA-256 fingerprints and temporal-quality counters
- Offline end-to-end demo
- Optional production detector adapter

## Code Quality Improvements

- Typed dataclasses replace loosely shaped nested dictionaries at component
  boundaries.
- Modules have focused ownership and testable public functions.
- Paths and thresholds are centralized.
- Errors identify missing artifacts and invalid configuration.
- Tests cover data, optimization, evaluation, inference, API security,
  occlusion recovery, batching, temporal stability, and audit fingerprints.
- Documentation distinguishes measured facts from future claims.
