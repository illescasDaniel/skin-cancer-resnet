from typing import cast

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from skin_cancer_resnet.model import ImageBatch, Net, evaluate, get_transforms


def test_get_transforms_returns_train_and_eval_pipelines() -> None:
	transform_train, transform_eval = get_transforms()

	assert isinstance(transform_train, transforms.Compose)
	assert isinstance(transform_eval, transforms.Compose)
	assert len(transform_train.transforms) > len(transform_eval.transforms)


def test_net_forward_output_shape() -> None:
	model = Net(weights=None)
	batch = torch.randn(4, 3, 224, 224)

	output = model(batch)

	assert output.ndim == 2
	assert output.shape[0] == batch.shape[0]
	assert output.shape[1] > 0


def test_net_predict_returns_class_indices() -> None:
	model = Net(weights=None)
	batch = torch.randn(3, 3, 224, 224)

	predictions = model.predict(batch)

	assert predictions.shape == (3,)
	assert predictions.dtype == torch.int64


def test_evaluate_computes_accuracy() -> None:
	model = Net(weights=None)
	device = torch.device("cpu")
	data = torch.randn(8, 3, 224, 224)
	targets = torch.zeros(8, dtype=torch.long)
	loader = cast(
		DataLoader[ImageBatch],
		DataLoader(TensorDataset(data, targets), batch_size=4),
	)
	accuracy = evaluate(model, device, loader)

	assert 0.0 <= accuracy <= 1.0
