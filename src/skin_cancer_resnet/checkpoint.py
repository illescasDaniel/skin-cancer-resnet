from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from skin_cancer_resnet.architecture import Architecture, default_model_path


DEFAULT_MODEL_PATH = default_model_path(Architecture.RESNET18)
CLASS_NAMES = ("benign", "malignant")


def save_checkpoint(state_dict: dict[str, torch.Tensor], path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	save_file(state_dict, str(path))


def load_checkpoint(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
	return load_file(str(path), device=str(device))


def remap_legacy_state_dict(
	state_dict: dict[str, torch.Tensor],
	architecture: Architecture | str,
) -> dict[str, torch.Tensor]:
	architecture = Architecture(architecture)
	remapped = dict(state_dict)

	if architecture == Architecture.RESNET18:
		if "classifier.weight" in remapped:
			classifier_weight = remapped["classifier.weight"]
			if classifier_weight.ndim == 2 and classifier_weight.shape[0] == 2:
				remapped["_classifier.weight"] = remapped.pop("classifier.weight")
				if "classifier.bias" in remapped:
					remapped["_classifier.bias"] = remapped.pop("classifier.bias")

		remapped = {key: value for key, value in remapped.items() if not key.startswith("_backbone.fc")}

	return remapped
