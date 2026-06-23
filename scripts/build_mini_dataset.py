#!/usr/bin/env python3
"""Build a small augmented dataset from sample images for quick architecture comparisons."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def augment_image(image: Image.Image, seed: int) -> Image.Image:
	random_generator = random.Random(seed)
	augmented = image.copy()
	if random_generator.random() < 0.5:
		augmented = ImageOps.mirror(augmented)
	if random_generator.random() < 0.5:
		augmented = ImageOps.flip(augmented)
	angle = random_generator.choice([0, 90, 180, 270])
	augmented = augmented.rotate(angle, expand=True)
	augmented = ImageEnhance.Brightness(augmented).enhance(random_generator.uniform(0.85, 1.15))
	augmented = ImageEnhance.Contrast(augmented).enhance(random_generator.uniform(0.85, 1.15))
	return augmented.convert("RGB")


def build_split(source_dir: Path, target_dir: Path, count: int, seed: int) -> None:
	source_images = sorted(path for path in source_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
	if not source_images:
		raise SystemExit(f"No source images found in {source_dir}")

	target_dir.mkdir(parents=True, exist_ok=True)
	for index in range(count):
		source_path = source_images[index % len(source_images)]
		with Image.open(source_path) as image:
			augmented = augment_image(image, seed + index)
			output_path = target_dir / f"{source_path.stem}_{index:04d}.jpg"
			augmented.save(output_path, format="JPEG", quality=90)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Create a small augmented dataset for quick comparisons.")
	parser.add_argument("--source-dir", type=Path, required=True, help="Directory with benign/ and malignant/ folders.")
	parser.add_argument("--output-dir", type=Path, default=Path("data_mini"))
	parser.add_argument("--train-count", type=int, default=80, help="Augmented images per class in train/.")
	parser.add_argument("--test-count", type=int, default=20, help="Augmented images per class in test/.")
	parser.add_argument("--seed", type=int, default=28)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if args.output_dir.exists():
		shutil.rmtree(args.output_dir)

	for split, count in (("train", args.train_count), ("test", args.test_count)):
		for class_name in ("benign", "malignant"):
			build_split(
				args.source_dir / class_name,
				args.output_dir / split / class_name,
				count,
				seed=args.seed + hash((split, class_name)) % 10_000,
			)

	print(f"Created mini dataset at {args.output_dir.resolve()}")


if __name__ == "__main__":
	main()
