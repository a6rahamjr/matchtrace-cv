from __future__ import annotations

from pathlib import Path

import torch

from matchtrace.models.team_classifier import TinyTeamCNN, load_checkpoint


def export_torchscript(
    checkpoint_path: str | Path,
    output_path: str | Path,
) -> Path:
    checkpoint = load_checkpoint(checkpoint_path)
    model = TinyTeamCNN(
        channels=int(checkpoint["channels"]),
        classes=len(checkpoint["class_names"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    image_size = int(checkpoint["image_size"])
    example = torch.zeros(1, 3, image_size, image_size)
    traced = torch.jit.trace(model, example)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output))
    return output

