from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from matchtrace.models.team_classifier import TeamClassifier


def create_saliency_map(
    classifier: TeamClassifier,
    image_path: str | Path,
    output_path: str | Path,
) -> Path:
    source = cv2.imread(str(image_path))
    if source is None:
        raise FileNotFoundError(f"Image could not be read: {image_path}")

    tensor = classifier.prepare(source)
    tensor.requires_grad_(True)
    logits = classifier.model(tensor)
    predicted = logits.argmax(dim=1)
    logits[0, predicted].backward()
    saliency = tensor.grad.detach().abs().amax(dim=1)[0].cpu().numpy()
    saliency -= saliency.min()
    saliency /= max(float(saliency.max()), 1e-8)
    heatmap = cv2.applyColorMap(
        np.uint8(saliency * 255),
        cv2.COLORMAP_TURBO,
    )
    heatmap = cv2.resize(heatmap, (source.shape[1], source.shape[0]))
    overlay = cv2.addWeighted(source, 0.55, heatmap, 0.45, 0)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), overlay):
        raise ValueError(f"Could not write saliency image: {output}")
    return output

