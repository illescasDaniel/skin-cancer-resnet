from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


DEFAULT_MODEL_PATH = Path("models/resnet18_skin_cancer.safetensors")
CLASS_NAMES = ("benign", "malignant")


def save_checkpoint(state_dict: dict[str, torch.Tensor], path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	save_file(state_dict, str(path))


def load_checkpoint(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
	return load_file(str(path), device=str(device))
