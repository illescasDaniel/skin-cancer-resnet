import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from skin_cancer_resnet.architecture import Architecture, default_model_path
from skin_cancer_resnet.checkpoint import CLASS_NAMES
from skin_cancer_resnet.device import get_best_device
from skin_cancer_resnet.model import DEFAULT_BATCH_SIZE, Net, get_transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_image_paths(paths: list[Path]) -> list[Path]:
	image_paths: list[Path] = []
	for path in paths:
		if not path.exists():
			raise FileNotFoundError(f"Path not found: {path}")
		if path.is_dir():
			dir_images = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
			if not dir_images:
				raise ValueError(f"No images found in directory: {path}")
			image_paths.extend(dir_images)
		elif path.suffix.lower() in IMAGE_EXTENSIONS:
			image_paths.append(path)
		else:
			raise ValueError(f"Not a supported image file: {path}")
	return image_paths


def classify_images(
	model: Net,
	image_paths: list[Path],
	device: torch.device,
	batch_size: int,
) -> list[tuple[Path, str, float]]:
	_, transform_eval = get_transforms()
	results: list[tuple[Path, str, float]] = []

	for batch_start in range(0, len(image_paths), batch_size):
		batch_paths = image_paths[batch_start : batch_start + batch_size]
		tensors = []
		for image_path in batch_paths:
			with Image.open(image_path) as image:
				tensors.append(transform_eval(image.convert("RGB")))

		batch = torch.stack(tensors).to(device)
		with torch.no_grad():
			probs = F.softmax(model(batch), dim=1)
			confidences, predictions = torch.max(probs, dim=1)

		for image_path, prediction, confidence in zip(
			batch_paths, predictions.tolist(), confidences.tolist(), strict=True
		):
			results.append((image_path, CLASS_NAMES[prediction], confidence))

	return results


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Classify skin lesion images as benign or malignant.")
	parser.add_argument(
		"paths",
		nargs="+",
		type=Path,
		help="Image files and/or directories containing images.",
	)
	parser.add_argument(
		"--architecture",
		choices=Architecture.choices(),
		default=Architecture.RESNET18.value,
		help="Backbone architecture used for the checkpoint (default: resnet18).",
	)
	parser.add_argument(
		"--model-path",
		type=Path,
		default=None,
		help="Path to model weights (Safetensors). Defaults depend on --architecture.",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=DEFAULT_BATCH_SIZE,
		help="Batch size for inference.",
	)
	parser.add_argument(
		"--device",
		type=str,
		default=None,
		help="Device override (e.g. cuda, cuda:0, mps, cpu). Auto-selects when omitted.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	architecture = Architecture(args.architecture)
	model_path = args.model_path or default_model_path(architecture)
	device = get_best_device(args.device)

	if not model_path.exists():
		raise SystemExit(f"Model not found: {model_path}")

	image_paths = collect_image_paths(args.paths)
	model = Net.from_checkpoint(model_path, device, architecture=architecture)
	results = classify_images(model, image_paths, device, args.batch_size)

	for image_path, label, confidence in results:
		print(f"{image_path}: {label} ({confidence * 100:.1f}%)")


if __name__ == "__main__":
	main()
