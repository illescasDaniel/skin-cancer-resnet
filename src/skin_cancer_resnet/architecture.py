from enum import Enum
from pathlib import Path


class Architecture(str, Enum):
	RESNET18 = "resnet18"
	MOBILENET_V3_SMALL = "mobilenet_v3_small"

	@classmethod
	def choices(cls) -> list[str]:
		return [architecture.value for architecture in cls]


DEFAULT_MODEL_PATHS: dict[Architecture, Path] = {
	Architecture.RESNET18: Path("models/resnet18_skin_cancer.safetensors"),
	Architecture.MOBILENET_V3_SMALL: Path("models/mobilenet_v3_small_skin_cancer.safetensors"),
}

RECOMMENDED_EPOCHS: dict[Architecture, int] = {
	Architecture.RESNET18: 15,
	Architecture.MOBILENET_V3_SMALL: 10,
}


def default_model_path(architecture: Architecture | str) -> Path:
	return DEFAULT_MODEL_PATHS[Architecture(architecture)]


def recommended_epochs(architecture: Architecture | str) -> int:
	return RECOMMENDED_EPOCHS[Architecture(architecture)]


def results_dir_for(architecture: Architecture | str, base_dir: Path = Path("results")) -> Path:
	return base_dir / Architecture(architecture).value
