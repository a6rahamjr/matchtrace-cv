# Training Guide

## Dataset Strategy

The project is not based on a standard public benchmark. Its default dataset is
generated locally with a seeded script that creates two visually distinct team
classes under randomized position, stripe, brightness, and sensor noise.

Synthetic data makes the default tests reproducible and avoids bundling match
footage. It does not replace evaluation on licensed real football footage.

## Generate Data

```bash
matchtrace generate-data
```

The configured seed produces the same arrays on repeated runs. The compressed
dataset contains separate train, validation, and test splits.

## Train

```bash
matchtrace train
```

The training loop:

1. seeds Python, NumPy, and PyTorch;
2. loads fixed dataset splits;
3. trains `TinyTeamCNN` from random initialization with AdamW;
4. measures validation loss and accuracy after each epoch;
5. saves the best validation checkpoint with preprocessing metadata.

Tune the `[model]` section in `configs/default.toml` to change channels, batch
size, epochs, learning rate, weight decay, worker count, or CPU thread count.

## Evaluate

```bash
matchtrace evaluate
```

Evaluation uses the untouched test split and writes:

- sample count
- cross-entropy loss
- accuracy
- class names
- confusion matrix

Do not tune against the test metrics.

## Export

```bash
matchtrace export
```

The exported TorchScript model contains the neural network graph for deployment.
The regular checkpoint additionally stores class and preprocessing metadata.

## Explainability

Create a saliency overlay for a player crop:

```bash
matchtrace explain --image player.jpg --output outputs/saliency.jpg
```

The overlay highlights input pixels with the strongest gradient influence on
the predicted team class. Saliency is a diagnostic, not a causal explanation.
