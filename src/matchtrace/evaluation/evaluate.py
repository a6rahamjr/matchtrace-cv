from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from matchtrace.models.team_classifier import TinyTeamCNN, load_checkpoint
from matchtrace.training.train import ArrayImageDataset, load_dataset
from matchtrace.utils.config import configured_path


def confusion_matrix(
    targets: np.ndarray,
    predictions: np.ndarray,
    classes: int,
) -> list[list[int]]:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions, strict=True):
        matrix[int(target), int(prediction)] += 1
    return matrix.tolist()


def evaluate_model(
    settings: dict[str, Any],
    dataset_path: str | Path | None = None,
    model_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    dataset_file = (
        Path(dataset_path)
        if dataset_path is not None
        else configured_path(settings, "dataset")
    )
    checkpoint_file = (
        Path(model_path)
        if model_path is not None
        else configured_path(settings, "model")
    )
    output = (
        Path(output_path)
        if output_path is not None
        else configured_path(settings, "evaluation")
    )

    arrays = load_dataset(dataset_file)
    checkpoint = load_checkpoint(checkpoint_file)
    model = TinyTeamCNN(
        channels=int(checkpoint["channels"]),
        classes=len(checkpoint["class_names"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    loader = DataLoader(
        ArrayImageDataset(arrays["test_images"], arrays["test_labels"]),
        batch_size=int(settings["model"]["batch_size"]),
        shuffle=False,
    )
    criterion = nn.CrossEntropyLoss(reduction="sum")
    predictions = []
    targets = []
    total_loss = 0.0
    with torch.inference_mode():
        for inputs, batch_targets in loader:
            logits = model(inputs)
            total_loss += float(criterion(logits, batch_targets).item())
            predictions.extend(logits.argmax(dim=1).tolist())
            targets.extend(batch_targets.tolist())

    target_array = np.asarray(targets)
    prediction_array = np.asarray(predictions)
    accuracy = float((target_array == prediction_array).mean())
    metrics = {
        "samples": len(targets),
        "accuracy": accuracy,
        "loss": total_loss / len(targets),
        "class_names": list(checkpoint["class_names"]),
        "confusion_matrix": confusion_matrix(
            target_array,
            prediction_array,
            len(checkpoint["class_names"]),
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics

