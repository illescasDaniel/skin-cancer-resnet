from pathlib import Path
from typing import Callable, cast

import torch
import torchvision.models as models
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, ResNet18_Weights

from skin_cancer_resnet.architecture import Architecture
from skin_cancer_resnet.checkpoint import load_checkpoint, remap_legacy_state_dict
from skin_cancer_resnet.training_config import (
	DEFAULT_LR,
	TrainingConfig,
	build_criterion,
	build_optimizer,
	build_scheduler,
)


ImageBatch = tuple[torch.Tensor, torch.Tensor]

DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 15
DEFAULT_SEED = 28

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
	transform_train = transforms.Compose(
		[
			transforms.Resize(224),
			transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(3 / 4, 4 / 3)),
			transforms.RandomHorizontalFlip(),
			transforms.RandomVerticalFlip(),
			transforms.RandomChoice(
				[
					transforms.RandomRotation(degrees=(0, 0), expand=True, fill=0),
					transforms.RandomRotation(degrees=(90, 90), expand=True, fill=0),
					transforms.RandomRotation(degrees=(180, 180), expand=True, fill=0),
					transforms.RandomRotation(degrees=(270, 270), expand=True, fill=0),
				]
			),
			transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
			transforms.ToTensor(),
			transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
		]
	)

	transform_eval = transforms.Compose(
		[
			transforms.Resize(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
		]
	)

	return transform_train, transform_eval


def _create_backbone(architecture: Architecture, pretrained: bool) -> nn.Module:
	if architecture == Architecture.RESNET18:
		weights = ResNet18_Weights.DEFAULT if pretrained else None
		backbone = models.resnet18(weights=weights)
		backbone.fc = nn.Identity()
		return backbone

	weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
	backbone = models.mobilenet_v3_small(weights=weights)
	classifier = backbone.classifier
	last_layer = cast(nn.Linear, classifier[-1])
	classifier[-1] = nn.Linear(last_layer.in_features, 2)
	return backbone


def _classifier_in_features(architecture: Architecture, backbone: nn.Module) -> int:
	if architecture == Architecture.RESNET18:
		return 512

	classifier = cast(nn.Sequential, backbone.classifier)
	return cast(nn.Linear, classifier[-1]).in_features


class Net(nn.Module):
	def __init__(
		self,
		architecture: Architecture | str = Architecture.RESNET18,
		scheduler_iters: int = -1,
		initial_lr: float = DEFAULT_LR,
		pretrained: bool = True,
		training_config: TrainingConfig | None = None,
	):
		super().__init__()
		config = training_config or TrainingConfig(
			architecture=Architecture(architecture),
			scheduler_iters=scheduler_iters,
			initial_lr=initial_lr,
			pretrained=pretrained,
		)
		self.training_config = config
		self.architecture = config.architecture
		self._backbone = _create_backbone(self.architecture, config.pretrained)

		if self.architecture == Architecture.RESNET18:
			self._classifier = nn.Linear(_classifier_in_features(self.architecture, self._backbone), 2)

		self.optimizer = build_optimizer(self, config)
		self.scheduler = build_scheduler(self.optimizer, config)
		self.criterion = build_criterion(config)

	@property
	def classifier(self) -> nn.Linear:
		if self.architecture == Architecture.RESNET18:
			return self._classifier

		classifier = cast(nn.Sequential, self._backbone.classifier)
		return cast(nn.Linear, classifier[-1])

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if self.architecture == Architecture.RESNET18:
			features = self._backbone(x)
			return self.classifier(features)
		return self._backbone(x)

	@classmethod
	def from_checkpoint(
		cls,
		path: Path | str,
		device: torch.device,
		architecture: Architecture | str = Architecture.RESNET18,
		*,
		strict: bool = True,
	) -> "Net":
		model = cls(architecture=architecture, pretrained=False).to(device)
		state_dict = remap_legacy_state_dict(load_checkpoint(Path(path), device), architecture)
		model.load_state_dict(state_dict, strict=strict)
		model.eval()
		return model

	def fit(
		self,
		device: torch.device,
		train_loader: DataLoader[ImageBatch],
		epochs: int,
		epoch_callback: Callable[[int, float], None] | None = None,
	) -> float:
		if self.scheduler.T_max < 0:
			self.scheduler.T_max = epochs

		self.train()
		total_loss = 0.0

		for epoch in range(epochs):
			running_loss = 0.0
			batch_count = 0

			for data, target in train_loader:
				data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

				self.optimizer.zero_grad()
				output = self(data)
				loss = self.criterion(output, target)
				if torch.isnan(loss) or torch.isinf(loss):
					raise ValueError("Loss is invalid")
				loss.backward()
				self.optimizer.step()

				running_loss += loss.item()
				batch_count += 1

			if batch_count > 0:
				epoch_loss = running_loss / batch_count
				total_loss += epoch_loss
				if epoch_callback:
					epoch_callback(epoch, epoch_loss)

			self.scheduler.step()

		return total_loss / epochs

	def predict(self, inputs: torch.Tensor) -> torch.Tensor:
		self.eval()
		with torch.no_grad():
			outputs = self(inputs)
			_, predicted = torch.max(outputs, 1)
			return predicted


def evaluate(model: Net, device: torch.device, data_loader: DataLoader[ImageBatch]) -> float:
	model.eval()
	correct = 0
	total = 0
	with torch.no_grad():
		for data, target in data_loader:
			data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
			pred = model.predict(data)
			correct += pred.eq(target.view_as(pred)).sum().item()
			total += len(target)
	return correct / total


def create_data_loader(
	dataset: Dataset[ImageBatch],
	batch_size: int,
	shuffle: bool,
	device: torch.device | None = None,
) -> DataLoader[ImageBatch]:
	pin_memory = device is not None and device.type == "cuda"
	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=2,
		pin_memory=pin_memory,
		persistent_workers=True,
	)
