from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from matchtrace.models.team_classifier import TinyTeamCNN
from matchtrace.utils.config import configured_path
from matchtrace.utils.seed import seed_everything


class ArrayImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        if len(images) != len(labels):
            raise ValueError("Image and label counts do not match.")
        self.images = images
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.from_numpy(self.images[index]).permute(2, 0, 1).float()
        image = image / 255.0
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return image, label


def load_dataset(path: str | Path) -> dict[str, np.ndarray]:
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. Run generate-data first."
        )
    with np.load(dataset_path, allow_pickle=False) as data:
        required = {
            "train_images",
            "train_labels",
            "validation_images",
            "validation_labels",
            "test_images",
            "test_labels",
            "class_names",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"Dataset is missing arrays: {', '.join(missing)}")
        return {key: data[key] for key in data.files}


def _epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    examples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(inputs)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()

        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        examples += batch_size

    return total_loss / examples, correct / examples


def train_model(
    settings: dict[str, Any],
    dataset_path: str | Path | None = None,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    seed = int(settings["project"]["seed"])
    seed_everything(seed)
    model_config = settings["model"]
    torch.set_num_threads(int(model_config["torch_threads"]))

    dataset_file = (
        Path(dataset_path)
        if dataset_path is not None
        else configured_path(settings, "dataset")
    )
    output = (
        Path(model_path)
        if model_path is not None
        else configured_path(settings, "model")
    )
    arrays = load_dataset(dataset_file)
    train_dataset = ArrayImageDataset(
        arrays["train_images"],
        arrays["train_labels"],
    )
    validation_dataset = ArrayImageDataset(
        arrays["validation_images"],
        arrays["validation_labels"],
    )
    generator = torch.Generator().manual_seed(seed)
    batch_size = int(model_config["batch_size"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(model_config["num_workers"]),
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(model_config["num_workers"]),
    )

    channels = int(model_config["channels"])
    model = TinyTeamCNN(channels=channels, classes=len(arrays["class_names"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )

    history = []
    best_accuracy = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    epochs = int(model_config["epochs"])
    for epoch_index in range(epochs):
        train_loss, train_accuracy = _epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
        )
        validation_loss, validation_accuracy = _epoch(
            model,
            validation_loader,
            criterion,
            device,
        )
        history.append(
            {
                "epoch": epoch_index + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": best_state,
        "class_names": arrays["class_names"].tolist(),
        "image_size": int(arrays["train_images"].shape[1]),
        "channels": channels,
        "seed": seed,
        "best_validation_accuracy": best_accuracy,
        "history_json": json.dumps(history),
    }
    torch.save(checkpoint, output)
    return {
        "checkpoint": str(output),
        "device": str(device),
        "epochs": epochs,
        "best_validation_accuracy": best_accuracy,
        "history": history,
    }

