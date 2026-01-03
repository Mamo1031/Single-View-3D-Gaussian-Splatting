# Single-View 3D Gaussian Splatting

A Python program to generate 3D Gaussian Splatting models from a single image using depth estimation via MiDaS.

## Overview

This project generates 3D Gaussian Splatting models from a single input image by:
1. Using MiDaS to estimate depth from the image
2. Converting the depth map to a 3D point cloud
3. Initializing and training a Gaussian Splatting model from the point cloud

The output is a 3D model file in `.ply` format that can be viewed and rendered from different viewpoints.

## Requirements

- Python 3.10 or higher
- uv (for package management)

## Installation

### Project Setup

```bash
# Navigate to the parent directory
cd Gaussian-Splatting   

# Install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

**Note**: The depth estimation uses MiDaS, which will automatically download the model from HuggingFace on first use. No additional setup is required.

## Project Structure

```
Gaussian-Splatting/
├── src/
│   └── gaussian_splatting/
│       ├── __init__.py
│       ├── gaussian_model.py    # Gaussian Splatting model definition
│       ├── renderer.py          # Rendering functionality
│       └── trainer.py           # Training logic
├── scripts/
│   └── single_view_gaussian.py  # Main execution script for single-view generation
├── pyproject.toml
├── README.md
└── .gitignore
```

## Usage

### Basic Usage

Generate a 3D Gaussian Splatting model from a single image:

```bash
uv run python scripts/single_view_gaussian.py \
    --input_image ./data/images/image.png \
    --output ./data/output \
    --depth_model DPT_Large \
    --iterations 30000 \
    --lr 0.01
```

### Arguments

- `--input_image`: Path to the input image (required)
- `--output`: Path to output directory (required)
- `--depth_model`: MiDaS model type (default: `DPT_Large`). Options: `DPT_Large`, `DPT_Hybrid`, `MiDaS_small`, etc.
- `--iterations`: Number of training iterations (default: 30000)
- `--lr`: Learning rate (default: 0.01)
- `--skip_depth_estimation`: Skip depth estimation and use existing depth map
- `--depth_map`: Path to existing depth map (required if `--skip_depth_estimation` is used)

### Example

```bash
# Generate 3D model from a single image using MiDaS depth estimation
uv run python scripts/single_view_gaussian.py \
    --input_image ./data/images/image.png \
    --output ./data/output \
    --depth_model DPT_Large \
    --iterations 30000 \
    --lr 0.01

# Use a smaller/faster model
uv run python scripts/single_view_gaussian.py \
    --input_image ./data/images/image.png \
    --output ./data/output \
    --depth_model MiDaS_small \
    --iterations 30000 \
    --lr 0.01

# Use existing depth map
uv run python scripts/single_view_gaussian.py \
    --input_image ./data/images/image.png \
    --output ./data/output \
    --skip_depth_estimation \
    --depth_map ./data/images/image_depth.png \
    --iterations 30000 \
    --lr 0.01
```

## How It Works

1. **Depth Estimation**: The script uses MiDaS (Intel's depth estimation model) to estimate a depth map from the input image. The model is automatically downloaded from HuggingFace on first use. MiDaS is a lightweight and efficient depth estimation model that works well for single-view 3D reconstruction.

2. **Point Cloud Generation**: The depth map is converted to a 3D point cloud using camera intrinsics. Each pixel's depth value is used to calculate its 3D position.

3. **Gaussian Initialization**: The point cloud is used to initialize a Gaussian Splatting model, where each point becomes a Gaussian with position, color, opacity, scale, and rotation.

4. **Training**: The Gaussian Splatting model is trained to render the input image from the single camera viewpoint, optimizing the Gaussian parameters.

## Output

The script generates:
- `depth_map.png`: Estimated depth map from MiDaS
- `model.ply`: Trained 3D Gaussian Splatting model (can be viewed in MeshLab, CloudCompare, etc.)
- `training_output/`: Training logs and intermediate results


## Limitations and Notes

- **Single Viewpoint**: This method generates 3D models from a single viewpoint, which may have limitations in quality compared to multi-view reconstruction methods.

- **Depth Quality**: The quality of the final 3D model depends heavily on the accuracy of the depth estimation. Results may vary depending on the input image.

- **MiDaS Models**: Different MiDaS models offer different trade-offs between accuracy and speed. `DPT_Large` provides the best accuracy but is slower, while `MiDaS_small` is faster but less accurate.
