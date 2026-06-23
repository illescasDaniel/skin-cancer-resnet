# Skin Cancer ResNet Transfer Learning

Binary classification of skin lesions (benign vs malignant) using transfer learning with a pretrained backbone. Only the final classifier head is trained, keeping the rest of the network frozen.

Supported architectures:

| Architecture | CLI value | Default checkpoint |
|--------------|-----------|--------------------|
| ResNet18 (default) | `resnet18` | `models/resnet18_skin_cancer.safetensors` |
| MobileNetV3-Small (iOS / mobile) | `mobilenet_v3_small` | `models/mobilenet_v3_small_skin_cancer.safetensors` |

**Disclaimer:** This project is for educational and research purposes only. It is not intended for clinical or diagnostic use.

## Results

After 15 epochs on the [Melanoma Skin Cancer dataset](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images):

| Metric | ResNet18 (final) | ResNet18 (best) | ResNet152 (final) |
|--------|------------------|-----------------|-------------------|
| Test accuracy | 84.24% | **85.30%** (epoch 11) | 84.39% |
| Train accuracy | 86.27% | — | 86.92% |
| Final loss | 0.3155 | — | 0.2970 |

ResNet152 was evaluated as a comparison and achieved nearly identical test accuracy with a much larger model. **ResNet18** was chosen as the final architecture for its similar performance, faster training, and smaller footprint.

### Training curves

| Loss | Test accuracy | Train accuracy |
|------|---------------|----------------|
| ![Training loss](results/loss.png) | ![Test accuracy](results/test_accuracy.png) | ![Train accuracy](results/train_accuracy.png) |

A pretrained checkpoint is included at [`models/resnet18_skin_cancer.safetensors`](models/resnet18_skin_cancer.safetensors).

## Project structure

```
skin-cancer-resnet-transfer-learning/
├── src/skin_cancer_resnet/   # Model, training, and inference code
├── scripts/
│   ├── download_dataset.py
│   └── quality/              # checks.sh, ruff, pyright, etc.
├── data/                     # Dataset (not committed)
├── models/                   # Trained weights
└── results/                  # Training plots
```

## Setup

Requires Python 3.10+ and a CUDA-capable GPU (optional; CPU works but is slower).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Download the dataset

1. Create a [Kaggle API token](https://www.kaggle.com/settings) and save it to `~/.kaggle/kaggle.json`.
2. Restrict permissions: `chmod 600 ~/.kaggle/kaggle.json`
3. Install the Kaggle CLI and download the data:

```bash
pip install kaggle
python scripts/download_dataset.py
```

This populates `data/train/` and `data/test/` with `benign/` and `malignant/` subfolders.

## Training

```bash
python -m skin_cancer_resnet.train \
  --data-dir data \
  --epochs 15 \
  --output-dir results \
  --architecture resnet18 \
  --model-path models/resnet18_skin_cancer.safetensors
```

Train the lighter MobileNetV3-Small backbone (as used in the [MalignantMolesDetector](https://github.com/illescasDaniel/MalignantMolesDetector) iOS app):

```bash
python -m skin_cancer_resnet.train \
  --architecture mobilenet_v3_small \
  --model-path models/mobilenet_v3_small_skin_cancer.safetensors
```

Or use the installed CLI:

```bash
skin-cancer-train
```

Training applies random augmentations (flips, rotations, color jitter) and evaluates on both train and test sets each epoch. Plots are saved to `results/` and weights to `models/`.

## Inference

Classify one or more images (files or directories):

```bash
skin-cancer-predict data/validation/IMG_4226.JPG
skin-cancer-predict path/to/image1.jpg path/to/images/
skin-cancer-predict --architecture mobilenet_v3_small path/to/image.jpg
```

Or run as a module:

```bash
python -m skin_cancer_resnet.predict data/validation/IMG_4226.JPG
```

Output is one line per image with the predicted label and confidence, e.g. `benign (62.1%)` or `malignant (87.3%)`.

Programmatic usage:

```python
import torch
from skin_cancer_resnet import Net, load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net(weights=None).to(device)
model.load_state_dict(load_checkpoint("models/resnet18_skin_cancer.safetensors", device))
model.eval()

# inputs: batch of preprocessed images (N, 3, 224, 224)
predictions = model.predict(inputs)
# 0 = benign, 1 = malignant
```

## Approach

- **Backbones:** ResNet18 or MobileNetV3-Small with ImageNet weights (`torchvision.models`)
- **Classifier:** Single linear layer (512 or 1024 → 2), only trainable parameters
- **Optimizer:** Adam (lr=1e-3, weight_decay=1e-4)
- **Scheduler:** Cosine annealing over 15 epochs
- **Batch size:** 16
- **Input size:** 224×224, ImageNet normalization

## License

MIT — Copyright (c) 2026 Daniel Illescas Romero. See [LICENSE](LICENSE).

## Quality checks

```bash
pip install -e ".[dev]"
./scripts/quality/checks.sh          # check-only (matches CI)
./scripts/quality/checks.sh --fix    # Ruff autofix/format + shfmt on shell scripts
```

Steps: dependency audit → Ruff → ShellCheck + shfmt → codespell → pytest → basedpyright.

Individual scripts: `scripts/quality/{ruff,pyright,shellcheck,codespell,pytest}.sh`.

Compare architectures (size, speed, and optional training accuracy):

```bash
python scripts/compare_architectures.py --data-dir data --epochs 5
```
