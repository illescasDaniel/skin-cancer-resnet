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


def default_model_path(architecture: Architecture | str) -> Path:
	return DEFAULT_MODEL_PATHS[Architecture(architecture)]
