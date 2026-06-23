from typing import Callable

import torch
import torchvision.models as models
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import ResNet18_Weights

DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 15
DEFAULT_SEED = 28
DEFAULT_LR = 1e-3

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
	transform_train = transforms.Compose([
		transforms.Resize(224),
		transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(3/4, 4/3)),
		transforms.RandomHorizontalFlip(),
		transforms.RandomVerticalFlip(),
		transforms.RandomChoice([
			transforms.RandomRotation(degrees=(0, 0), expand=True, fill=0),
			transforms.RandomRotation(degrees=(90, 90), expand=True, fill=0),
			transforms.RandomRotation(degrees=(180, 180), expand=True, fill=0),
			transforms.RandomRotation(degrees=(270, 270), expand=True, fill=0),
		]),
		transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
		transforms.ToTensor(),
		transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
	])

	transform_eval = transforms.Compose([
		transforms.Resize(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
	])

	return transform_train, transform_eval


class Net(nn.Module):
	def __init__(self, scheduler_iters: int = -1, initial_lr: float = DEFAULT_LR):
		super().__init__()
		self._backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
		self.classifier = nn.Linear(in_features=self.classifier.in_features, out_features=2)

		self.optimizer = optim.Adam(self.classifier.parameters(), lr=initial_lr, weight_decay=1e-4)
		self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=scheduler_iters)
		self.criterion = nn.CrossEntropyLoss()

	@property
	def classifier(self) -> nn.Linear:
		return self._backbone.fc

	@classifier.setter
	def classifier(self, value: nn.Linear):
		self._backbone.fc = value

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self._backbone(x)

	def fit(
		self,
		device: torch.device,
		train_loader: DataLoader,
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


def evaluate(model: Net, device: torch.device, data_loader: DataLoader) -> float:
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
	dataset: torch.utils.data.Dataset,
	batch_size: int,
	shuffle: bool,
) -> DataLoader:
	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=2,
		pin_memory=True,
		persistent_workers=True,
	)
