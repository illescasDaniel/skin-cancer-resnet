from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import torch

from skin_cancer_resnet.architecture import recommended_epochs
from skin_cancer_resnet.device import get_best_device
from skin_cancer_resnet.model import DEFAULT_BATCH_SIZE, DEFAULT_SEED
from skin_cancer_resnet.train import run_training
from skin_cancer_resnet.training_config import TrainingConfig


EXPERIMENTS_OUTPUT_DIR = Path("results/experiments")
EXPERIMENTS_MODEL_DIR = Path("models/experiments")


@dataclass
class ExperimentSpec:
	name: str
	training_config: TrainingConfig
	data_dir: Path
	epochs: int
	batch_size: int
	seed: int
	output_dir: Path
	model_path: Path
	device: torch.device | None = None


def default_output_dir(name: str) -> Path:
	return EXPERIMENTS_OUTPUT_DIR / name


def default_model_path_for_experiment(name: str) -> Path:
	return EXPERIMENTS_MODEL_DIR / f"{name}.safetensors"


def load_experiment_file(config_path: Path) -> dict[str, Any]:
	return json.loads(config_path.read_text())


def resolve_experiment(
	experiment_data: dict[str, Any],
	defaults: dict[str, Any],
) -> ExperimentSpec:
	name = experiment_data.get("name")
	if not name:
		raise ValueError("Each experiment must include a 'name' field.")

	merged = {**defaults, **experiment_data}
	run_fields = {
		"epochs",
		"batch_size",
		"seed",
		"data_dir",
		"output_dir",
		"model_path",
		"device",
		"name",
	}
	training_config = TrainingConfig.from_dict({key: value for key, value in merged.items() if key not in run_fields})

	architecture = training_config.architecture
	epochs = int(merged.get("epochs") or recommended_epochs(architecture))
	batch_size = int(merged.get("batch_size", DEFAULT_BATCH_SIZE))
	seed = int(merged.get("seed", DEFAULT_SEED))
	data_dir = Path(merged.get("data_dir", "data"))
	output_dir = Path(merged["output_dir"]) if merged.get("output_dir") else default_output_dir(name)
	model_path = Path(merged["model_path"]) if merged.get("model_path") else default_model_path_for_experiment(name)
	device = torch.device(merged["device"]) if merged.get("device") else None

	return ExperimentSpec(
		name=name,
		training_config=training_config,
		data_dir=data_dir,
		epochs=epochs,
		batch_size=batch_size,
		seed=seed,
		output_dir=output_dir,
		model_path=model_path,
		device=device,
	)


def parse_experiments(config_data: dict[str, Any]) -> list[ExperimentSpec]:
	experiments = config_data.get("experiments")
	if not experiments:
		raise ValueError("Experiment config must include an 'experiments' array.")

	defaults = {key: value for key, value in config_data.items() if key != "experiments"}
	return [resolve_experiment(experiment, defaults) for experiment in experiments]


def default_max_workers(experiment_count: int) -> int:
	if torch.cuda.is_available():
		return min(experiment_count, torch.cuda.device_count())
	return 1


def assign_devices(experiments: list[ExperimentSpec]) -> list[ExperimentSpec]:
	if not torch.cuda.is_available():
		return experiments

	assigned: list[ExperimentSpec] = []
	for index, experiment in enumerate(experiments):
		if experiment.device is not None:
			assigned.append(experiment)
			continue
		device_index = index % torch.cuda.device_count()
		assigned.append(
			ExperimentSpec(
				name=experiment.name,
				training_config=experiment.training_config,
				data_dir=experiment.data_dir,
				epochs=experiment.epochs,
				batch_size=experiment.batch_size,
				seed=experiment.seed,
				output_dir=experiment.output_dir,
				model_path=experiment.model_path,
				device=torch.device(f"cuda:{device_index}"),
			)
		)
	return assigned


def _run_experiment_worker(spec: ExperimentSpec) -> dict[str, Any]:
	device = spec.device or get_best_device()
	metrics = run_training(
		training_config=spec.training_config,
		data_dir=spec.data_dir,
		epochs=spec.epochs,
		batch_size=spec.batch_size,
		seed=spec.seed,
		output_dir=spec.output_dir,
		model_path=spec.model_path,
		device=device,
	)
	return {
		"name": spec.name,
		"output_dir": str(spec.output_dir),
		"model_path": str(spec.model_path),
		"device": str(device),
		"test_accuracy": metrics["test_accuracy"],
	}


def run_experiments(experiments: list[ExperimentSpec], max_workers: int) -> list[dict[str, Any]]:
	assigned = assign_devices(experiments)
	worker_count = min(max_workers, len(assigned))
	results: list[dict[str, Any]] = []

	if worker_count <= 1:
		for experiment in assigned:
			results.append(_run_experiment_worker(experiment))
		return results

	ctx = get_context("spawn")
	with ProcessPoolExecutor(max_workers=worker_count, mp_context=ctx) as executor:
		future_to_name = {
			executor.submit(_run_experiment_worker, experiment): experiment.name for experiment in assigned
		}
		for future in as_completed(future_to_name):
			name = future_to_name[future]
			try:
				results.append(future.result())
			except Exception as error:
				raise RuntimeError(f"Experiment '{name}' failed: {error}") from error

	return sorted(results, key=lambda result: result["name"])


def print_summary(results: list[dict[str, Any]]) -> None:
	print("\nExperiment summary:")
	print(f"{'Name':<30} {'Test Acc':>10} {'Device':<10} Output")
	print("-" * 90)
	for result in results:
		accuracy = result["test_accuracy"] * 100
		print(f"{result['name']:<30} {accuracy:>9.2f}% {result['device']:<10} {result['output_dir']}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run multiple training experiments in parallel.")
	parser.add_argument("--config", type=Path, required=True, help="Path to experiment JSON config file.")
	parser.add_argument(
		"--data-dir",
		type=Path,
		default=None,
		help="Override dataset root for all experiments.",
	)
	parser.add_argument(
		"--max-workers",
		type=int,
		default=None,
		help="Maximum number of parallel experiment workers.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	config_data = load_experiment_file(args.config)
	if args.data_dir is not None:
		config_data["data_dir"] = str(args.data_dir)

	experiments = parse_experiments(config_data)
	max_workers = args.max_workers or int(config_data.get("max_workers", default_max_workers(len(experiments))))
	results = run_experiments(experiments, max_workers=max_workers)
	print_summary(results)


if __name__ == "__main__":
	main()
