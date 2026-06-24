import random

import numpy as np
import torch


def get_best_device(preferred: str | None = None) -> torch.device:
	if preferred is not None:
		device = torch.device(preferred)
		if device.type == "cuda" and not torch.cuda.is_available():
			raise ValueError(f"Requested CUDA device but CUDA is not available: {preferred}")
		if device.type == "mps" and not _mps_is_available():
			raise ValueError(f"Requested MPS device but MPS is not available: {preferred}")
		return device

	if torch.cuda.is_available():
		return torch.device("cuda")
	if _mps_is_available():
		return torch.device("mps")
	return torch.device("cpu")


def configure_backend(device: torch.device) -> None:
	if device.type == "cuda":
		torch.backends.cudnn.benchmark = True


def seed_device(device: torch.device, seed: int) -> None:
	torch.manual_seed(seed)
	random.seed(seed)
	np.random.seed(seed)
	if device.type == "cuda":
		torch.cuda.manual_seed(seed)
	elif device.type == "mps" and hasattr(torch.mps, "manual_seed"):
		torch.mps.manual_seed(seed)
	torch.backends.cudnn.deterministic = True


def _mps_is_available() -> bool:
	return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
