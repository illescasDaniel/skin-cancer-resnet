from unittest.mock import patch

import torch

from skin_cancer_resnet.device import configure_backend, get_best_device, seed_device


def test_get_best_device_defaults_to_cpu_without_accelerators() -> None:
	with (
		patch("skin_cancer_resnet.device.torch.cuda.is_available", return_value=False),
		patch("skin_cancer_resnet.device._mps_is_available", return_value=False),
	):
		device = get_best_device()

	assert device.type == "cpu"


def test_get_best_device_prefers_cuda() -> None:
	with (
		patch("skin_cancer_resnet.device.torch.cuda.is_available", return_value=True),
		patch("skin_cancer_resnet.device._mps_is_available", return_value=True),
	):
		device = get_best_device()

	assert device.type == "cuda"


def test_get_best_device_falls_back_to_mps() -> None:
	with (
		patch("skin_cancer_resnet.device.torch.cuda.is_available", return_value=False),
		patch("skin_cancer_resnet.device._mps_is_available", return_value=True),
	):
		device = get_best_device()

	assert device.type == "mps"


def test_get_best_device_honors_override() -> None:
	device = get_best_device("cpu")
	assert device.type == "cpu"


def test_configure_backend_only_enables_cudnn_for_cuda() -> None:
	torch.backends.cudnn.benchmark = False
	configure_backend(torch.device("cpu"))
	assert torch.backends.cudnn.benchmark is False

	configure_backend(torch.device("cuda"))
	assert torch.backends.cudnn.benchmark is True
	torch.backends.cudnn.benchmark = False


def test_seed_device_sets_cpu_seed() -> None:
	seed_device(torch.device("cpu"), 123)
	tensor_a = torch.randn(2)
	seed_device(torch.device("cpu"), 123)
	tensor_b = torch.randn(2)
	assert torch.equal(tensor_a, tensor_b)
