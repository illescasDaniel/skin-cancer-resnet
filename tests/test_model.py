from typing import cast

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from skin_cancer_resnet.architecture import Architecture
from skin_cancer_resnet.checkpoint import remap_legacy_state_dict
from skin_cancer_resnet.model import ImageBatch, Net, evaluate, get_transforms


@pytest.mark.parametrize("architecture", Architecture.choices())
def test_get_transforms_returns_train_and_eval_pipelines(architecture: str) -> None:
	_ = architecture
	transform_train, transform_eval = get_transforms()

	assert isinstance(transform_train, transforms.Compose)
	assert isinstance(transform_eval, transforms.Compose)
	assert len(transform_train.transforms) > len(transform_eval.transforms)


@pytest.mark.parametrize("architecture", Architecture.choices())
def test_net_forward_output_shape(architecture: str) -> None:
	model = Net(architecture=architecture, pretrained=False)
	batch = torch.randn(4, 3, 224, 224)

	output = model(batch)

	assert output.ndim == 2
	assert output.shape[0] == batch.shape[0]
	assert output.shape[1] == 2


@pytest.mark.parametrize("architecture", Architecture.choices())
def test_net_predict_returns_class_indices(architecture: str) -> None:
	model = Net(architecture=architecture, pretrained=False)
	batch = torch.randn(3, 3, 224, 224)

	predictions = model.predict(batch)

	assert predictions.shape == (3,)
	assert predictions.dtype == torch.int64


@pytest.mark.parametrize("architecture", Architecture.choices())
def test_evaluate_computes_accuracy(architecture: str) -> None:
	model = Net(architecture=architecture, pretrained=False)
	device = torch.device("cpu")
	data = torch.randn(8, 3, 224, 224)
	targets = torch.zeros(8, dtype=torch.long)
	loader = cast(
		DataLoader[ImageBatch],
		DataLoader(TensorDataset(data, targets), batch_size=4),
	)
	accuracy = evaluate(model, device, loader)

	assert 0.0 <= accuracy <= 1.0


def test_remap_legacy_state_dict_drops_old_resnet_head() -> None:
	state_dict = {
		"_backbone.fc.weight": torch.randn(1000, 512),
		"_backbone.fc.bias": torch.randn(1000),
		"classifier.weight": torch.randn(2, 512),
		"classifier.bias": torch.randn(2),
	}

	remapped = remap_legacy_state_dict(state_dict, Architecture.RESNET18)

	assert "_backbone.fc.weight" not in remapped
	assert remapped["_classifier.weight"].shape == (2, 512)
	assert remapped["_classifier.bias"].shape == (2,)
