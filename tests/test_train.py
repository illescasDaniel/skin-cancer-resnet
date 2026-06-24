from pathlib import Path

from skin_cancer_resnet.architecture import Architecture, recommended_epochs, results_dir_for
from skin_cancer_resnet.train import build_metrics


def test_results_dir_for() -> None:
	assert results_dir_for(Architecture.RESNET18) == Path("results/resnet18")
	assert results_dir_for(Architecture.MOBILENET_V3_SMALL) == Path("results/mobilenet_v3_small")
	assert results_dir_for(Architecture.RESNET18, Path("output")) == Path("output/resnet18")


def test_recommended_epochs() -> None:
	assert recommended_epochs(Architecture.RESNET18) == 15
	assert recommended_epochs(Architecture.MOBILENET_V3_SMALL) == 10


def test_build_metrics_records_best_epoch() -> None:
	metrics = build_metrics(
		Architecture.MOBILENET_V3_SMALL,
		epochs=3,
		loss_list=[0.5, 0.4, 0.3],
		test_accuracy_list=[0.7, 0.8, 0.75],
		train_accuracy_list=[0.75, 0.85, 0.9],
		best_epoch=1,
		best_test_accuracy=0.8,
	)

	assert metrics["architecture"] == "mobilenet_v3_small"
	assert metrics["best_epoch"] == 2
	assert metrics["test_accuracy"] == 0.8
	assert metrics["train_accuracy"] == 0.85
	assert metrics["loss"] == 0.4
	assert len(metrics["loss_per_epoch"]) == 3
