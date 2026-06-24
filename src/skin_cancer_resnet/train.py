import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets

from skin_cancer_resnet.architecture import Architecture, default_model_path, recommended_epochs, results_dir_for
from skin_cancer_resnet.checkpoint import save_checkpoint
from skin_cancer_resnet.model import (
	DEFAULT_BATCH_SIZE,
	DEFAULT_SEED,
	Net,
	create_data_loader,
	evaluate,
	get_transforms,
)


def set_seed(seed: int) -> None:
	torch.manual_seed(seed)
	random.seed(seed)
	np.random.seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True


def epoch_axis(epoch_count: int) -> range:
	return range(1, epoch_count + 1)


def save_plots(
	output_dir: Path,
	loss_list: list[float],
	test_accuracy_list: list[float],
	train_accuracy_list: list[float],
) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)
	epochs = epoch_axis(len(loss_list))

	plt.figure(figsize=(10, 5))
	plt.plot(epochs, loss_list, label="Loss")
	plt.title("Training loss")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.legend()
	plt.grid(True)
	plt.savefig(output_dir / "loss.png")
	plt.close()

	plt.figure(figsize=(10, 5))
	plt.plot(epochs, test_accuracy_list, label="Test Accuracy")
	plt.title("Test accuracy")
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.legend()
	plt.grid(True)
	plt.savefig(output_dir / "test_accuracy.png")
	plt.close()

	plt.figure(figsize=(10, 5))
	plt.plot(epochs, train_accuracy_list, label="Train Accuracy")
	plt.title("Train accuracy")
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.legend()
	plt.grid(True)
	plt.savefig(output_dir / "train_accuracy.png")
	plt.close()


def build_metrics(
	architecture: Architecture,
	epochs: int,
	loss_list: list[float],
	test_accuracy_list: list[float],
	train_accuracy_list: list[float],
	best_epoch: int,
	best_test_accuracy: float,
) -> dict[str, Any]:
	return {
		"architecture": architecture.value,
		"epochs": epochs,
		"loss_per_epoch": loss_list,
		"test_accuracy_per_epoch": test_accuracy_list,
		"train_accuracy_per_epoch": train_accuracy_list,
		"best_epoch": best_epoch + 1,
		"test_accuracy": best_test_accuracy,
		"train_accuracy": train_accuracy_list[best_epoch],
		"loss": loss_list[best_epoch],
	}


def save_metrics(output_dir: Path, metrics: dict[str, Any]) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)
	(output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train a skin lesion classifier with transfer learning.")
	parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Dataset root directory.")
	parser.add_argument(
		"--epochs",
		type=int,
		default=None,
		help="Number of training epochs (defaults to the recommended value for --architecture).",
	)
	parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size.")
	parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=None,
		help="Directory for plot output (defaults to results/{architecture}/).",
	)
	parser.add_argument(
		"--architecture",
		choices=Architecture.choices(),
		default=Architecture.RESNET18.value,
		help="Backbone architecture to train (default: resnet18).",
	)
	parser.add_argument(
		"--model-path",
		type=Path,
		default=None,
		help="Path to save trained model weights (defaults depend on --architecture).",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	architecture = Architecture(args.architecture)
	epochs = args.epochs or recommended_epochs(architecture)
	model_path = args.model_path or default_model_path(architecture)
	output_dir = args.output_dir or results_dir_for(architecture)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	torch.backends.cudnn.benchmark = True
	set_seed(args.seed)

	transform_train, transform_eval = get_transforms()
	train_dir = args.data_dir / "train"
	test_dir = args.data_dir / "test"

	train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
	train_eval_dataset = datasets.ImageFolder(train_dir, transform=transform_eval)
	test_dataset = datasets.ImageFolder(test_dir, transform=transform_eval)

	train_loader = create_data_loader(train_dataset, batch_size=args.batch_size, shuffle=True)
	train_eval_loader = create_data_loader(train_eval_dataset, batch_size=args.batch_size, shuffle=False)
	test_loader = create_data_loader(test_dataset, batch_size=args.batch_size, shuffle=False)

	model = Net(architecture=architecture, scheduler_iters=epochs).to(device)
	loss_list: list[float] = []
	test_accuracy_list: list[float] = []
	train_accuracy_list: list[float] = []
	best_test_accuracy = 0.0
	best_epoch = 0
	best_state_dict: dict[str, torch.Tensor] | None = None

	def epoch_callback(epoch: int, loss: float) -> None:
		nonlocal best_test_accuracy, best_epoch, best_state_dict
		loss_list.append(loss)
		test_accuracy = evaluate(model, device, test_loader)
		train_accuracy = evaluate(model, device, train_eval_loader)
		test_accuracy_list.append(test_accuracy)
		train_accuracy_list.append(train_accuracy)
		if test_accuracy > best_test_accuracy:
			best_test_accuracy = test_accuracy
			best_epoch = epoch
			best_state_dict = copy.deepcopy(model.state_dict())
		print(
			f"Epoch [{epoch + 1}/{epochs}] - "
			f"Loss: {loss:.4f}, Test Accuracy: {test_accuracy:.4f}, "
			f"Train Accuracy: {train_accuracy:.4f}"
		)

	print(f"Starting training with {architecture.value} for {epochs} epochs...")
	avg_loss = model.fit(device, train_loader, epochs=epochs, epoch_callback=epoch_callback)
	print(f"Average loss: {avg_loss:.4f}")
	print(f"Best test accuracy: {best_test_accuracy * 100:.2f}% (epoch {best_epoch + 1})")

	if best_state_dict is None:
		raise RuntimeError("Training did not produce a checkpoint.")

	model_path.parent.mkdir(parents=True, exist_ok=True)
	print(f"Saving best model to {model_path}...")
	save_checkpoint(best_state_dict, model_path)
	print("Model saved.")

	metrics = build_metrics(
		architecture,
		epochs,
		loss_list,
		test_accuracy_list,
		train_accuracy_list,
		best_epoch,
		best_test_accuracy,
	)
	save_metrics(output_dir, metrics)
	save_plots(output_dir, loss_list, test_accuracy_list, train_accuracy_list)
	print(f"Plots and metrics saved to {output_dir}.")


if __name__ == "__main__":
	main()
