from pathlib import Path

import torch

from skin_cancer_resnet.architecture import Architecture, default_model_path, recommended_epochs, results_dir_for
from skin_cancer_resnet.checkpoint import CLASS_NAMES, load_checkpoint, save_checkpoint


def test_class_names() -> None:
	assert CLASS_NAMES == ("benign", "malignant")


def test_default_model_paths() -> None:
	assert default_model_path(Architecture.RESNET18).name == "resnet18_skin_cancer.safetensors"
	assert default_model_path(Architecture.MOBILENET_V3_SMALL).name == "mobilenet_v3_small_skin_cancer.safetensors"
	assert recommended_epochs(Architecture.RESNET18) == 15
	assert recommended_epochs(Architecture.MOBILENET_V3_SMALL) == 10
	assert results_dir_for(Architecture.RESNET18) == Path("results/resnet18")


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
