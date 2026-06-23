#!/usr/bin/env python3
"""Compare ResNet18 and MobileNetV3-small on model size, speed, and optional training accuracy."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torchvision import datasets

from skin_cancer_resnet.architecture import Architecture, default_model_path
from skin_cancer_resnet.checkpoint import load_checkpoint, remap_legacy_state_dict, save_checkpoint
from skin_cancer_resnet.model import Net, create_data_loader, evaluate, get_transforms


def count_parameters(model: Net) -> int:
	return sum(parameter.numel() for parameter in model.parameters())


def checkpoint_size_mb(path: Path) -> float:
	return path.stat().st_size / (1024 * 1024)


def benchmark_inference(model: Net, device: torch.device, batch_size: int, iterations: int) -> float:
	model.eval()
	inputs = torch.randn(batch_size, 3, 224, 224, device=device)
	with torch.no_grad():
		for _ in range(5):
			model(inputs)

	start = time.perf_counter()
	with torch.no_grad():
		for _ in range(iterations):
			model(inputs)
	elapsed = time.perf_counter() - start
	return elapsed / iterations


def train_and_evaluate(
	architecture: Architecture,
	data_dir: Path,
	epochs: int,
	batch_size: int,
	output_path: Path,
	device: torch.device,
) -> dict[str, float | str]:
	transform_train, transform_eval = get_transforms()
	train_dataset = datasets.ImageFolder(data_dir / "train", transform=transform_train)
	test_dataset = datasets.ImageFolder(data_dir / "test", transform=transform_eval)
	train_loader = create_data_loader(train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = create_data_loader(test_dataset, batch_size=batch_size, shuffle=False)

	model = Net(architecture=architecture).to(device)
	best_accuracy = 0.0

	def epoch_callback(epoch: int, loss: float) -> None:
		nonlocal best_accuracy
		test_accuracy = evaluate(model, device, test_loader)
		best_accuracy = max(best_accuracy, test_accuracy)
		print(f"[{architecture.value}] Epoch {epoch + 1}/{epochs} loss={loss:.4f} test_acc={test_accuracy:.4f}")

	avg_loss = model.fit(device, train_loader, epochs=epochs, epoch_callback=epoch_callback)
	final_accuracy = evaluate(model, device, test_loader)
	save_checkpoint(model.state_dict(), output_path)

	return {
		"architecture": architecture.value,
		"average_loss": avg_loss,
		"final_test_accuracy": final_accuracy,
		"best_test_accuracy": best_accuracy,
		"checkpoint_mb": checkpoint_size_mb(output_path),
		"parameters": count_parameters(model),
	}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Compare classifier architectures.")
	parser.add_argument("--data-dir", type=Path, default=Path("data"))
	parser.add_argument("--epochs", type=int, default=5)
	parser.add_argument("--batch-size", type=int, default=16)
	parser.add_argument("--benchmark-iterations", type=int, default=20)
	parser.add_argument("--output-json", type=Path, default=Path("results/architecture_comparison.json"))
	parser.add_argument("--skip-training", action="store_true")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	results: dict[str, object] = {"device": str(device), "architectures": {}}

	for architecture in Architecture:
		model = Net(architecture=architecture, pretrained=True).to(device)
		arch_results: dict[str, float | int | str] = {
			"parameters": count_parameters(model),
		}
		arch_results["inference_seconds_per_batch"] = benchmark_inference(
			model, device, args.batch_size, args.benchmark_iterations
		)

		checkpoint_path = default_model_path(architecture)
		if checkpoint_path.exists():
			loaded = Net(architecture=architecture, pretrained=False).to(device)
			state_dict = remap_legacy_state_dict(load_checkpoint(checkpoint_path, device), architecture)
			loaded.load_state_dict(state_dict, strict=False)
			arch_results["existing_checkpoint_mb"] = checkpoint_size_mb(checkpoint_path)
			arch_results["existing_checkpoint_inference_seconds_per_batch"] = benchmark_inference(
				loaded, device, args.batch_size, args.benchmark_iterations
			)

		if not args.skip_training:
			train_dir = args.data_dir / "train"
			test_dir = args.data_dir / "test"
			if not train_dir.is_dir() or not test_dir.is_dir():
				raise SystemExit(
					f"Training data not found under {args.data_dir}. "
					"Run scripts/download_dataset.py or pass --skip-training."
				)
			comparison_path = Path("models") / f"{architecture.value}_comparison.safetensors"
			arch_results.update(
				train_and_evaluate(
					architecture,
					args.data_dir,
					args.epochs,
					args.batch_size,
					comparison_path,
					device,
				)
			)

		results["architectures"][architecture.value] = arch_results

	args.output_json.parent.mkdir(parents=True, exist_ok=True)
	args.output_json.write_text(json.dumps(results, indent=2))
	print(json.dumps(results, indent=2))
	print(f"Wrote comparison to {args.output_json}")


if __name__ == "__main__":
	main()
