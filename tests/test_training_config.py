from skin_cancer_resnet.architecture import Architecture
from skin_cancer_resnet.model import Net
from skin_cancer_resnet.training_config import (
	CriterionType,
	OptimizerType,
	SchedulerType,
	TrainingConfig,
	build_criterion,
	build_optimizer,
	build_scheduler,
)


def test_training_config_defaults() -> None:
	config = TrainingConfig()

	assert config.architecture == Architecture.RESNET18
	assert config.initial_lr == 1e-3
	assert config.weight_decay == 1e-4
	assert config.optimizer == OptimizerType.ADAM
	assert config.scheduler == SchedulerType.COSINE_ANNEALING
	assert config.scheduler_iters == -1
	assert config.criterion == CriterionType.CROSS_ENTROPY
	assert config.pretrained is True


def test_training_config_from_dict_merges_partial_overrides() -> None:
	config = TrainingConfig.from_dict({"architecture": "mobilenet_v3_small", "initial_lr": 0.0005})

	assert config.architecture == Architecture.MOBILENET_V3_SMALL
	assert config.initial_lr == 0.0005
	assert config.optimizer == OptimizerType.ADAM


def test_training_config_round_trip() -> None:
	original = TrainingConfig(
		architecture=Architecture.MOBILENET_V3_SMALL,
		initial_lr=0.0002,
		weight_decay=0.001,
		pretrained=False,
	)
	restored = TrainingConfig.from_dict(original.to_dict())

	assert restored == original


def test_build_optimizer_scheduler_criterion() -> None:
	config = TrainingConfig(pretrained=False)
	model = Net(training_config=config)

	optimizer = build_optimizer(model, config)
	scheduler = build_scheduler(optimizer, config)
	criterion = build_criterion(config)

	assert optimizer.__class__.__name__ == "Adam"
	assert scheduler.__class__.__name__ == "CosineAnnealingLR"
	assert criterion.__class__.__name__ == "CrossEntropyLoss"
