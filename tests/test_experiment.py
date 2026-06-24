import json
from pathlib import Path

from skin_cancer_resnet.architecture import Architecture
from skin_cancer_resnet.experiment import (
	default_max_workers,
	default_model_path_for_experiment,
	default_output_dir,
	load_experiment_file,
	parse_experiments,
	resolve_experiment,
)
from skin_cancer_resnet.training_config import TrainingConfig


def test_default_paths_for_named_experiment() -> None:
	assert default_output_dir("resnet18-baseline") == Path("results/experiments/resnet18-baseline")
	assert default_model_path_for_experiment("resnet18-baseline") == Path(
		"models/experiments/resnet18-baseline.safetensors"
	)


def test_resolve_experiment_merges_defaults() -> None:
	spec = resolve_experiment(
		{"name": "exp-a", "architecture": "resnet18", "epochs": 5},
		{"data_dir": "data", "batch_size": 8, "seed": 42},
	)

	assert spec.name == "exp-a"
	assert spec.data_dir == Path("data")
	assert spec.epochs == 5
	assert spec.batch_size == 8
	assert spec.seed == 42
	assert spec.training_config.architecture == Architecture.RESNET18
	assert spec.output_dir == default_output_dir("exp-a")
	assert spec.model_path == default_model_path_for_experiment("exp-a")


def test_parse_experiments_from_json_file(tmp_path: Path) -> None:
	config_path = tmp_path / "experiments.json"
	config_path.write_text(
		json.dumps(
			{
				"data_dir": "data",
				"experiments": [
					{"name": "one", "architecture": "resnet18"},
					{"name": "two", "architecture": "mobilenet_v3_small", "initial_lr": 0.0001},
				],
			}
		)
	)

	experiments = parse_experiments(load_experiment_file(config_path))

	assert len(experiments) == 2
	assert experiments[0].training_config == TrainingConfig(architecture=Architecture.RESNET18)
	assert experiments[1].training_config.initial_lr == 0.0001


def test_default_max_workers_is_one_without_cuda() -> None:
	assert default_max_workers(3) >= 1
