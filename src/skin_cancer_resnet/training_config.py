from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from enum import Enum
from typing import TYPE_CHECKING, Any

from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

from skin_cancer_resnet.architecture import Architecture


if TYPE_CHECKING:
	from skin_cancer_resnet.model import Net


DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4


class OptimizerType(str, Enum):
	ADAM = "adam"

	@classmethod
	def choices(cls) -> list[str]:
		return [optimizer.value for optimizer in cls]


class SchedulerType(str, Enum):
	COSINE_ANNEALING = "cosine_annealing"

	@classmethod
	def choices(cls) -> list[str]:
		return [scheduler.value for scheduler in cls]


class CriterionType(str, Enum):
	CROSS_ENTROPY = "cross_entropy"

	@classmethod
	def choices(cls) -> list[str]:
		return [criterion.value for criterion in cls]


@dataclass
class TrainingConfig:
	architecture: Architecture = Architecture.RESNET18
	initial_lr: float = DEFAULT_LR
	weight_decay: float = DEFAULT_WEIGHT_DECAY
	optimizer: OptimizerType = OptimizerType.ADAM
	scheduler: SchedulerType = SchedulerType.COSINE_ANNEALING
	scheduler_iters: int = -1
	criterion: CriterionType = CriterionType.CROSS_ENTROPY
	pretrained: bool = True

	def to_dict(self) -> dict[str, Any]:
		data = asdict(self)
		data["architecture"] = self.architecture.value
		data["optimizer"] = self.optimizer.value
		data["scheduler"] = self.scheduler.value
		data["criterion"] = self.criterion.value
		return data

	@classmethod
	def from_dict(cls, data: dict[str, Any]) -> TrainingConfig:
		known_fields = {field.name for field in fields(cls)}
		filtered = {key: value for key, value in data.items() if key in known_fields}
		if "architecture" in filtered:
			filtered["architecture"] = Architecture(filtered["architecture"])
		if "optimizer" in filtered:
			filtered["optimizer"] = OptimizerType(filtered["optimizer"])
		if "scheduler" in filtered:
			filtered["scheduler"] = SchedulerType(filtered["scheduler"])
		if "criterion" in filtered:
			filtered["criterion"] = CriterionType(filtered["criterion"])
		return cls(**filtered)


def build_optimizer(model: Net, config: TrainingConfig) -> optim.Optimizer:
	if config.optimizer == OptimizerType.ADAM:
		return optim.Adam(
			model.classifier.parameters(),
			lr=config.initial_lr,
			weight_decay=config.weight_decay,
		)
	raise ValueError(f"Unsupported optimizer: {config.optimizer.value}")


def build_scheduler(optimizer: optim.Optimizer, config: TrainingConfig) -> LRScheduler:
	if config.scheduler == SchedulerType.COSINE_ANNEALING:
		return CosineAnnealingLR(optimizer, T_max=config.scheduler_iters)
	raise ValueError(f"Unsupported scheduler: {config.scheduler.value}")


def build_criterion(config: TrainingConfig) -> nn.Module:
	if config.criterion == CriterionType.CROSS_ENTROPY:
		return nn.CrossEntropyLoss()
	raise ValueError(f"Unsupported criterion: {config.criterion.value}")
