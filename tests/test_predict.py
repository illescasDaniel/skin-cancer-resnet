from pathlib import Path

import pytest
from PIL import Image

from skin_cancer_resnet.predict import IMAGE_EXTENSIONS, collect_image_paths


def _write_test_image(path: Path) -> None:
	Image.new("RGB", (32, 32), color="red").save(path)


def test_collect_image_paths_from_file(tmp_path: Path) -> None:
	image_path = tmp_path / "lesion.jpg"
	_write_test_image(image_path)

	paths = collect_image_paths([image_path])

	assert paths == [image_path]


def test_collect_image_paths_from_directory(tmp_path: Path) -> None:
	for name in ("a.png", "b.jpeg"):
		_write_test_image(tmp_path / name)
	(tmp_path / "notes.txt").write_text("not an image")

	paths = collect_image_paths([tmp_path])

	assert len(paths) == 2
	assert all(path.suffix.lower() in IMAGE_EXTENSIONS for path in paths)


def test_collect_image_paths_missing_path_raises(tmp_path: Path) -> None:
	with pytest.raises(FileNotFoundError, match="Path not found"):
		collect_image_paths([tmp_path / "missing.jpg"])


def test_collect_image_paths_empty_directory_raises(tmp_path: Path) -> None:
	with pytest.raises(ValueError, match="No images found"):
		collect_image_paths([tmp_path])


def test_collect_image_paths_unsupported_file_raises(tmp_path: Path) -> None:
	bad_file = tmp_path / "data.csv"
	bad_file.write_text("x,y\n1,2")

	with pytest.raises(ValueError, match="Not a supported image file"):
		collect_image_paths([bad_file])
