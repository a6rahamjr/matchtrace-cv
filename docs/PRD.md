# MatchTrace CV Product Requirements Document

## Product Identity

**Name:** MatchTrace CV

**Description:** A Python computer-vision pipeline that converts
football video into player tracks, team assignments, possession estimates,
physical movement metrics, and an annotated output video.

**Tagline:** From footage to match intelligence.

## Problem Statement

Football-analysis prototypes commonly combine model loading, video processing,
tracking, analytics, and rendering in one process with hard-coded paths and
sample-specific assumptions. This makes them difficult to test, deploy, extend,
or reproduce without private model weights and media.

MatchTrace CV separates streaming inference, training, evaluation,
configuration, and API serving behind explicit data contracts. Its default
example uses generated assets.

## Objective

Build a runnable football-video analytics platform that:

- processes frames incrementally instead of holding an entire video in memory;
- separates detection, tracking, classification, geometry, analytics, and
  rendering;
- trains a team-appearance classifier from a seeded dataset;
- produces measurable evaluation artifacts;
- supports both a self-contained synthetic detector and an optional external
  YOLO detector;
- exposes CLI and REST interfaces;
- remains testable on CPU without network access.

## Target Users

- ML engineers developing football-analysis models
- analysts prototyping match-intelligence workflows
- backend engineers integrating video analytics into services
- students learning production ML project organization
- researchers who need reproducible synthetic baselines

## ML Task Type

The MVP combines:

- object detection for players and the ball;
- multi-object tracking;
- supervised image classification for team appearance;
- geometric projection from image coordinates to pitch coordinates;
- temporal analytics for speed, distance, and possession.

The bundled trainable model is a compact convolutional neural network trained
from scratch for two-team jersey classification. Real-footage object detection
is provided through a pluggable model adapter.

## Input Specification

### Training

- Seeded synthetic player crops stored as compressed NumPy arrays
- RGB images with configurable size
- Integer team labels: `0` and `1`

### Inference

- MP4, AVI, or other OpenCV-readable video
- TOML configuration
- Trained team-classifier checkpoint
- Detector backend:
  - `synthetic_color` for generated demo footage
  - `ultralytics` for approved real-footage weights

## Output Specification

- Annotated video preserving source resolution and frame rate
- JSON run summary with frame count, throughput, possession, distances,
  quality counters, and artifact fingerprints
- Evaluation JSON with accuracy, loss, and confusion matrix
- PyTorch checkpoint and optional TorchScript export
- Structured application logs

## System Workflow

1. Load and validate TOML configuration.
2. Seed Python, NumPy, and PyTorch.
3. Open the input video and initialize a streaming writer.
4. Detect players and the ball through the configured detector.
5. Associate player detections with stable track IDs.
6. Classify each player crop and aggregate team predictions by track.
7. Estimate camera motion and compensate tracked positions.
8. Project image positions into approximate pitch coordinates.
9. Calculate speed, cumulative distance, and possession.
10. Render overlays and write each frame immediately.
11. Save a machine-readable summary and structured logs.

## MVP Features

- Reproducible synthetic dataset and match-video generation
- CNN team classifier trained from scratch
- Training, validation, and held-out evaluation
- Streaming video inference
- Color-based demo detector
- Optional Ultralytics detector adapter
- Motion-and-IoU tracking with configurable gating, aging, and velocity
- Batched team classification with confidence-weighted track voting
- Camera-motion compensation
- Pitch-coordinate projection
- Smoothed speed, validated distance, and hysteretic possession analytics
- Annotated MP4 output
- CLI and health/analyze REST endpoints
- Unit and end-to-end tests

## Advanced Features

- Add appearance ReID or a ByteTrack/OC-SORT backend for crowded scenes
- Fine-tune a detector on licensed football footage
- Add field-line calibration and automatic homography estimation
- Add event detection for passes, shots, and turnovers
- Add asynchronous job queues and object storage
- Add Prometheus metrics and distributed experiment tracking
- Add multi-camera identity reconciliation
- Add human review and annotation workflows

## Success Metrics

- Synthetic team-classifier test accuracy of at least 95%
- Deterministic dataset generation for the same seed
- End-to-end synthetic demo completes without network access
- Output frame count equals input frame count
- Inference memory does not grow with video duration
- All automated tests pass on CPU
- Invalid paths and malformed configuration fail with actionable messages

Real-video accuracy claims require a licensed, representative evaluation
dataset and are outside the synthetic MVP acceptance criteria.

## Constraints

- No original project source modules, cached pickle stubs, notebooks, model
  weights, or media are copied into the new repository.
- Synthetic data is intentionally a functional baseline, not evidence of
  real-match accuracy.
- Physical speed depends on calibration quality.
- The lightweight motion-aware tracker prioritizes portability; crowded-scene
  production deployments should add appearance ReID or a stronger backend.
- External model and media licenses remain the deployer's responsibility.

## Technology Stack

- Python 3.11
- PyTorch for model training and export
- OpenCV for video processing, geometry, and rendering
- NumPy for numerical operations and synthetic data
- FastAPI and Uvicorn for the REST layer
- Pydantic for API schemas
- TOML for dependency-free configuration parsing
- `unittest` for CPU-compatible automated tests
