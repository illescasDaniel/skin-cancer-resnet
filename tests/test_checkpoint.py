from pathlib import Path

import torch

from skin_cancer_resnet.checkpoint import CLASS_NAMES, load_checkpoint, save_checkpoint


def test_class_names() -> None:
	assert CLASS_NAMES == ("benign", "malignant")


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
	state = {"classifier.weight": torch.randn(2, 512), "classifier.bias": torch.randn(2)}
	path = tmp_path / "model.safetensors"
	device = torch.device("cpu")

	save_checkpoint(state, path)
	loaded = load_checkpoint(path, device)

	assert path.exists()
	assert set(loaded.keys()) == set(state.keys())
	for key in state:
		assert torch.equal(loaded[key], state[key])
