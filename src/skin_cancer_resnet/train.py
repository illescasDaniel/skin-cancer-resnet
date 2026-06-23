import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets

from skin_cancer_resnet.architecture import Architecture, default_model_path
from skin_cancer_resnet.checkpoint import save_checkpoint
from skin_cancer_resnet.model import (
	DEFAULT_BATCH_SIZE,
	DEFAULT_EPOCHS,
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


def save_plots(
	output_dir: Path,
	loss_list: list[float],
	test_accuracy_list: list[float],
	train_accuracy_list: list[float],
) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)

	plt.figure(figsize=(10, 5))
	plt.plot(loss_list, label="Loss")
	plt.title("Training loss")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.legend()
	plt.grid(True)
	plt.savefig(output_dir / "loss.png")
	plt.close()

	plt.figure(figsize=(10, 5))
	plt.plot(test_accuracy_list, label="Test Accuracy")
	plt.title("Test accuracy")
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.legend()
	plt.grid(True)
	plt.savefig(output_dir / "test_accuracy.png")
	plt.close()

	plt.figure(figsize=(10, 5))
	plt.plot(train_accuracy_list, label="Train Accuracy")
	plt.title("Train accuracy")
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.legend()
	plt.grid(True)
	plt.savefig(output_dir / "train_accuracy.png")
	plt.close()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train a skin lesion classifier with transfer learning.")
	parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Dataset root directory.")
	parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
	parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size.")
	parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
	parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Directory for plot output.")
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
	model_path = args.model_path or default_model_path(architecture)
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

	model = Net(architecture=architecture).to(device)
	loss_list: list[float] = []
	test_accuracy_list: list[float] = []
	train_accuracy_list: list[float] = []

	def epoch_callback(epoch: int, loss: float) -> None:
		loss_list.append(loss)
		test_accuracy = evaluate(model, device, test_loader)
		train_accuracy = evaluate(model, device, train_eval_loader)
		test_accuracy_list.append(test_accuracy)
		train_accuracy_list.append(train_accuracy)
		print(
			f"Epoch [{epoch + 1}/{args.epochs}] - "
			f"Loss: {loss:.4f}, Test Accuracy: {test_accuracy:.4f}, "
			f"Train Accuracy: {train_accuracy:.4f}"
		)

	print(f"Starting training with {architecture.value}...")
	avg_loss = model.fit(device, train_loader, epochs=args.epochs, epoch_callback=epoch_callback)
	print(f"Average loss: {avg_loss:.4f}")

	model_path.parent.mkdir(parents=True, exist_ok=True)
	print(f"Saving model to {model_path}...")
	save_checkpoint(model.state_dict(), model_path)
	print("Model saved.")

	save_plots(args.output_dir, loss_list, test_accuracy_list, train_accuracy_list)
	print(f"Plots saved to {args.output_dir}.")


if __name__ == "__main__":
	main()
