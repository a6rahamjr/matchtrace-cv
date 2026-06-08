# Inference Guide

## Synthetic Demo

Run every stage:

```bash
matchtrace demo
```

Or run inference after data generation and training:

```bash
matchtrace analyze
```

## Custom Paths

```bash
matchtrace analyze \
  --input artifacts/data/demo_match.mp4 \
  --output outputs/custom.mp4 \
  --summary outputs/custom.json
```

## Processing Stages

1. Open the source video and read metadata.
2. Process one frame at a time.
3. Detect player and ball candidates.
4. Associate players with motion prediction, distance, and box overlap.
5. Batch-classify team appearance and aggregate confidence by track.
6. Estimate and accumulate camera movement.
7. Convert feet and ball positions into metric pitch coordinates.
8. Smooth positions, reject impossible jumps, and update possession, speed,
   and distance.
9. Draw overlays and immediately write the frame.
10. Record configuration, model, input, and output fingerprints.

## Real-Footage Backend

Install the optional dependency and configure:

```toml
[detector]
backend = "ultralytics"
confidence = 0.25
model_path = "models/approved-football-model.pt"
```

Then run:

```bash
matchtrace --config configs/production.toml analyze --input match.mp4
```

Recalibrate detector thresholds, tracker distance, pitch geometry, possession
distance, switch margin, smoothing, and speed limits for the camera angle and
competition.

## Output Verification

Confirm:

- output frame count matches input frame count;
- output FPS and resolution match the source;
- players retain IDs through ordinary motion;
- team colors are correct;
- possession changes are plausible;
- physical speed is plausible for the chosen calibration.

## Troubleshooting

**Checkpoint missing:** run `matchtrace train`.

**Video missing:** run `matchtrace generate-data` or pass `--input`.

**No real-video detections:** verify model class names and confidence.

**Writer failure:** change the four-character codec in the TOML configuration.

**Unrealistic speed:** improve pitch calibration and camera stabilization.
