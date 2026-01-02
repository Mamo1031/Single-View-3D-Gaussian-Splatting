# Gaussian Splatting

A Python program to generate 3D Gaussian Splatting models from multiple images.

## Overview

This project takes multiple images as input, estimates camera parameters using COLMAP, and generates 3D models using the Gaussian Splatting algorithm. The output is a 3D model file in .ply format.

## Requirements

- Python 3.10 or higher
- COLMAP (for camera parameter estimation)
- uv (for package management)

## Installation
### 1. Install COLMAP

#### Ubuntu/Debian:
```bash
sudo apt-get install colmap
```

#### macOS:
```bash
brew install colmap
```

#### Windows:
Download and install the installer from the [COLMAP official website](https://colmap.github.io/).

### 2. Project Setup

```bash
# Install dependencies
uv sync
```


## Project Structure

```
Gaussian-Splatting/
├── src/
│   └── gaussian_splatting/
│       ├── __init__.py
│       ├── colmap_utils.py      # COLMAP integration utilities
│       ├── gaussian_model.py    # Gaussian Splatting model definition
│       ├── renderer.py          # Rendering functionality
│       └── trainer.py           # Training logic
├── scripts/
│   └── main.py                  # Main execution script
├── pyproject.toml
├── README.md
└── .gitignore
```

## Usage

### Basic Usage

```bash
# Run COLMAP and then train
uv run python scripts/main.py \
    --input_images ./data/images \
    --output ./data/output \
    --run_colmap

# If COLMAP has already been executed
uv run python scripts/main.py \
    --input_images ./data/images \
    --output ./data/output \
    --colmap_dir ./data/colmap/output
```

### Example

```bash
# Train for 30000 iterations
uv run python scripts/main.py \
    --input_images ./images \
    --output ./output \
    --run_colmap \
    --iterations 30000 \
    --lr 0.01
```
