# Conveyor Object Tracker 3D (Late-Fusion)

This repository contains an ML system designed to detect, track, and locate moving objects on an industrial conveyor belt using RGB-D camera inputs. The system is designed as an independent module, allowing it to run interchangeably in real-world deployments, ROS 2 / Gazebo simulations, or offline on synthetic datasets.

## System Logic (Late-Fusion Architecture)

Instead of processing dense 3D point clouds directly, this system employs a **late-fusion paradigm** to maintain high performance with minimal computational overhead:

1. **2D RGB Detection:** A real-time detector (RF-DETR) predicts bounding boxes for the class `conveyor_object` using the RGB stream.
2. **2D Association (ByteTrack):** A training-free 2D temporal tracker associates detections across frames to assign consistent tracking IDs.
3. **3D Depth Filtering:** The 2D bounding boxes are slightly expanded and projected onto the Depth channel. Pixels representing the flat conveyor belt are filtered out (using a pre-calibrated belt plane), leaving only the object. The remaining pixels are used to compute the exact 3D centroid coordinates.

---

## Directory Structure

```text
.
├── pyproject.toml
├── uv.lock
├── src/                      # Core tracking and projection logic
└── scripts/                  # Executable utility and training scripts
    └── examples/             # Examples for quick start
```

---

## Installation & Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast, robust package and environment management.

### 1. Prerequisites
Ensure you have `uv` installed. If not, install it via:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Environment Setup
Clone this repository, navigate to the root directory, and synchronize the virtual environment:

```bash
git clone https://github.com/okushnarev/robozon-vis-cls.git
cd robozon-vis-cls
uv sync
```
This automatically parses `pyproject.toml` and lock files to configure your local `.venv` with the exact dependencies.

---

## Running Scripts

>**Execution Note:** Every script in the `scripts/` directory relies on being executed from the **project root folder** to correctly append `src/` to the Python path. Ensure your terminal is at the root directory of the repository before executing any run commands.


### Train the RF-DETR Model
Fine-tune the RF-DETR detector on your custom BlenderProc-generated synthetic dataset:
```bash
uv run scripts/train_rf_detr.py \
  --dataset-dir /path/to/dataset \
  --model-size medium \
  --epochs 50
```

### Run Inference Predictions
Test your trained model's predictions on directory of images, save metrics and images:
```bash
uv run scripts/predict_rf_detr.py \
  --ckpt-dir runs/conveyor_rfdetr \
  --ckpt-type checkpoint_best_total \
  --ds-dir /path/to/test/ \
  --out-dir /path/to/out/
```

### Try Tracking & Full Pipeline of Classifier
To evaluate the 2D tracking performance or run baseline performance evaluations, use the example scripts:
```bash
uv run scripts/examples/full_classifier.py
uv run scripts/examples/tracking.py
```