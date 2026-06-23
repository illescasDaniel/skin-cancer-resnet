#!/usr/bin/env python3
"""Download the melanoma skin cancer dataset from Kaggle into data/."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


DATASET_SLUG = "hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def check_kaggle_credentials() -> None:
	kaggle_config = Path.home() / ".kaggle" / "kaggle.json"
	if not kaggle_config.exists():
		raise SystemExit(
			"Kaggle credentials not found.\n"
			"Create an API token at https://www.kaggle.com/settings and save it to ~/.kaggle/kaggle.json\n"
			"Then run: chmod 600 ~/.kaggle/kaggle.json"
		)


def run_kaggle_download(target_dir: Path) -> Path:
	cmd = [
		"kaggle",
		"datasets",
		"download",
		"-d",
		DATASET_SLUG,
		"-p",
		str(target_dir),
		"--force",
	]
	print(f"Running: {' '.join(cmd)}")
	subprocess.run(cmd, check=True)
	archives = sorted(target_dir.glob("*.zip"))
	if not archives:
		raise SystemExit("Kaggle download completed but no zip archive was found.")
	return archives[0]


def find_dataset_root(extracted_dir: Path) -> Path:
	candidates = [
		extracted_dir / "melanoma_cancer_dataset",
		extracted_dir,
	]
	for candidate in candidates:
		if (candidate / "train").is_dir() and (candidate / "test").is_dir():
			return candidate

	for path in extracted_dir.rglob("train"):
		if path.is_dir() and (path.parent / "test").is_dir():
			return path.parent

	raise SystemExit(
		"Could not locate train/ and test/ folders in the downloaded archive. Check the dataset layout on Kaggle."
	)


def count_images(directory: Path) -> int:
	if not directory.is_dir():
		return 0
	return sum(1 for path in directory.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def copy_split(source_split: Path, target_split: Path) -> None:
	if target_split.exists():
		shutil.rmtree(target_split)
	shutil.copytree(source_split, target_split)


def install_dataset(source_root: Path, data_dir: Path) -> None:
	data_dir.mkdir(parents=True, exist_ok=True)
	for split in ("train", "test"):
		copy_split(source_root / split, data_dir / split)


def print_summary(data_dir: Path) -> None:
	print("\nDataset installed:")
	for split in ("train", "test"):
		split_dir = data_dir / split
		if not split_dir.is_dir():
			print(f"  {split}: missing")
			continue
		classes = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
		for class_name in classes:
			count = count_images(split_dir / class_name)
			print(f"  {split}/{class_name}: {count} images")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Download the skin cancer dataset from Kaggle.")
	parser.add_argument(
		"--data-dir",
		type=Path,
		default=Path("data"),
		help="Target directory for train/ and test/ splits.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	check_kaggle_credentials()

	with tempfile.TemporaryDirectory() as tmp:
		tmp_dir = Path(tmp)
		archive_path = run_kaggle_download(tmp_dir)

		extract_dir = tmp_dir / "extracted"
		extract_dir.mkdir()
		with zipfile.ZipFile(archive_path, "r") as archive:
			archive.extractall(extract_dir)

		source_root = find_dataset_root(extract_dir)
		install_dataset(source_root, args.data_dir)

	print_summary(args.data_dir)
	print(f"\nDone. Dataset is ready at {args.data_dir.resolve()}")


if __name__ == "__main__":
	try:
		main()
	except FileNotFoundError:
		raise SystemExit("The kaggle CLI was not found. Install it with: pip install kaggle") from None
	except subprocess.CalledProcessError as exc:
		raise SystemExit(f"Kaggle download failed with exit code {exc.returncode}.") from exc
