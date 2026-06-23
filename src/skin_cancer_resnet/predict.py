import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from skin_cancer_resnet.checkpoint import CLASS_NAMES, DEFAULT_MODEL_PATH, load_checkpoint
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


def load_model(model_path: Path, device: torch.device) -> Net:
	model = Net(weights=None).to(device)
	model.load_state_dict(load_checkpoint(model_path, device))
	model.eval()
	return model


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
		"--model-path",
		type=Path,
		default=DEFAULT_MODEL_PATH,
		help="Path to model weights (Safetensors).",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=DEFAULT_BATCH_SIZE,
		help="Batch size for inference.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if not args.model_path.exists():
		raise SystemExit(f"Model not found: {args.model_path}")

	image_paths = collect_image_paths(args.paths)
	model = load_model(args.model_path, device)
	results = classify_images(model, image_paths, device, args.batch_size)

	for image_path, label, confidence in results:
		print(f"{image_path}: {label} ({confidence * 100:.1f}%)")


if __name__ == "__main__":
	main()
