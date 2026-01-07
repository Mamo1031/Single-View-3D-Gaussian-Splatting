"""Single-view 3D Gaussian Splatting using depth estimation

This script generates 3D Gaussian Splatting from a single image by:
1. Using depth estimation (MiDaS or custom depth map) to estimate depth from the image
2. Converting depth map to 3D point cloud
3. Initializing Gaussian Splatting from the point cloud
4. Training the Gaussian Splatting model
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import cv2
from PIL import Image

from gaussian_splatting import (
    GaussianModel,
    Trainer,
    Camera,
)


def depth_to_pointcloud(
    image: np.ndarray, depth: np.ndarray, camera_intrinsics: np.ndarray
) -> np.ndarray:
    """Convert depth map to 3D point cloud.

    Args:
        image: RGB image [H, W, 3]
        depth: Depth map [H, W] (normalized 0-1, will be scaled)
        camera_intrinsics: Camera intrinsics matrix [3, 3]

    Returns:
        Point cloud [N, 6] (xyz + rgb)
    """
    H, W = depth.shape[:2]

    # Convert depth from normalized to actual depth values
    # Assuming depth is in range [0, 1], scale to reasonable range
    depth_scale = 10.0  # Adjust based on scene scale
    depth_values = depth[:, :, 0] if len(depth.shape) == 3 else depth
    depth_values = depth_values * depth_scale

    # Create pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Extract camera intrinsics
    fx = camera_intrinsics[0, 0]
    fy = camera_intrinsics[1, 1]
    cx = camera_intrinsics[0, 2]
    cy = camera_intrinsics[1, 2]

    # Convert to 3D coordinates
    x = (u - cx) * depth_values / fx
    y = (v - cy) * depth_values / fy
    z = depth_values

    # Stack coordinates
    xyz = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # Get RGB values
    if len(image.shape) == 3:
        rgb = image.reshape(-1, 3) / 255.0
    else:
        rgb = np.ones((xyz.shape[0], 3)) * 0.5

    # Filter out invalid points (zero depth or too far)
    valid_mask = (depth_values.flatten() > 0.1) & (
        depth_values.flatten() < depth_scale * 0.9
    )
    xyz = xyz[valid_mask]
    rgb = rgb[valid_mask]

    # Combine
    pcd = np.concatenate([xyz, rgb], axis=1)

    return pcd


def estimate_depth_with_midas(
    image_path: str,
    output_path: str,
    model_type: str = "DPT_Large",
) -> tuple:
    """Estimate depth using MiDaS.

    Args:
        image_path: Path to input image
        output_path: Path to save depth map
        model_type: MiDaS model type (DPT_Large, DPT_Hybrid, MiDaS_small, etc.)

    Returns:
        Tuple of (image, depth_map) as numpy arrays
    """
    # Map model type to torch.hub model name
    model_map = {
        "DPT_Large": "DPT_Large",
        "DPT_Hybrid": "DPT_Hybrid",
        "MiDaS_small": "MiDaS_small",
        "DPT_BEiT_Large_512": "DPT_BEiT_Large_512",
    }

    hub_model_name = model_map.get(model_type, "DPT_Large")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading MiDaS model ({hub_model_name})...")

    try:
        # Load MiDaS model from torch.hub
        model = torch.hub.load("intel-isl/MiDaS", hub_model_name, trust_repo=True)
        model.to(device)
        model.eval()

        # Load MiDaS transforms
        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms", trust_repo=True
        )
        if hub_model_name == "DPT_Large" or hub_model_name == "DPT_Hybrid":
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform

    except Exception as e:
        print(f"Error loading MiDaS model: {e}")
        if "timm" in str(e).lower():
            print("Error: 'timm' module not found.")
            print("Please install it: uv pip install timm")
            sys.exit(1)
        print("Trying alternative model: DPT_Hybrid")
        try:
            model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True)
            model.to(device)
            model.eval()
            midas_transforms = torch.hub.load(
                "intel-isl/MiDaS", "transforms", trust_repo=True
            )
            transform = midas_transforms.dpt_transform
        except Exception as e2:
            print(f"Error loading alternative model: {e2}")
            if "timm" in str(e2).lower():
                print("Error: 'timm' module not found.")
                print("Please install it: uv pip install timm")
            else:
                print("Please ensure you have internet connection for model download.")
            sys.exit(1)

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    # MiDaS transform expects numpy array (not PIL Image) and returns a tensor
    img = transform(image_np).to(device)

    # Predict depth
    with torch.no_grad():
        prediction = model(img)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.size[::-1],  # (height, width)
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    # Convert to numpy and normalize
    depth = prediction.cpu().numpy()
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())

    # Convert to 3-channel for compatibility (grayscale depth map)
    depth_3channel = np.stack([depth_normalized] * 3, axis=-1)
    depth_uint8 = (depth_3channel * 255).astype(np.uint8)

    # Save depth map
    depth_path = Path(output_path)
    depth_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(depth_path), cv2.cvtColor(depth_uint8, cv2.COLOR_RGB2BGR))

    return image_np, depth_3channel


def create_camera_from_image(image: np.ndarray, depth: np.ndarray) -> Camera:
    """Create a camera object from image and depth.

    Args:
        image: RGB image [H, W, 3]
        depth: Depth map [H, W, 3]

    Returns:
        Camera object
    """
    H, W = image.shape[:2]

    # Estimate camera intrinsics (assuming standard perspective camera)
    # Use image dimensions to estimate focal length
    focal_length = max(H, W) * 0.7  # Rough estimate
    fx = fy = focal_length
    cx = W / 2.0
    cy = H / 2.0

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # Identity rotation and zero translation (camera at origin)
    R = np.eye(3)
    T = np.zeros(3)

    # Calculate field of view
    FoVx = 2 * np.arctan(W / (2 * fx))
    FoVy = 2 * np.arctan(H / (2 * fy))

    camera = Camera(
        R=torch.tensor(R, dtype=torch.float32).cuda(),
        T=torch.tensor(T, dtype=torch.float32).cuda(),
        K=torch.tensor(K, dtype=torch.float32).cuda(),
        image_width=W,
        image_height=H,
        FoVx=FoVx,
        FoVy=FoVy,
    )

    return camera


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D Gaussian Splatting from single image using 3D-VLA"
    )
    parser.add_argument(
        "--input_image", type=str, required=True, help="Path to input image"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output directory"
    )
    parser.add_argument(
        "--depth_model",
        type=str,
        default="DPT_Large",
        help="MiDaS model type (DPT_Large, DPT_Hybrid, MiDaS_small, etc.)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=30000,
        help="Number of training iterations (default: 30000)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--skip_depth_estimation",
        action="store_true",
        help="Skip depth estimation and use existing depth map",
    )
    parser.add_argument(
        "--depth_map",
        type=str,
        default=None,
        help="Path to existing depth map (if skip_depth_estimation is True)",
    )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load or estimate depth
    if args.skip_depth_estimation and args.depth_map:
        print("Loading existing depth map...")
        image = cv2.imread(args.input_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(args.depth_map)
        if len(depth.shape) == 3:
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
        else:
            depth = np.stack([depth] * 3, axis=-1)
        depth = depth.astype(np.float32) / 255.0
    else:
        print("Estimating depth using MiDaS...")
        depth_output = output_path / "depth_map.png"
        image, depth = estimate_depth_with_midas(
            args.input_image, str(depth_output), args.depth_model
        )
        print(f"Depth map saved to {depth_output}")

    # Convert depth to point cloud
    print("Converting depth to point cloud...")
    # Estimate camera intrinsics
    H, W = image.shape[:2]
    focal_length = max(H, W) * 0.7
    camera_intrinsics = np.array(
        [[focal_length, 0, W / 2.0], [0, focal_length, H / 2.0], [0, 0, 1]]
    )

    pcd = depth_to_pointcloud(image, depth, camera_intrinsics)
    print(f"Generated {len(pcd)} points from depth map")

    # Create camera
    print("Creating camera...")
    camera = create_camera_from_image(image, depth)

    # Initialize Gaussian model
    print("Initializing Gaussian model...")
    gaussian_model = GaussianModel()
    scale = np.max(np.abs(pcd[:, :3])) if len(pcd) > 0 else 1.0
    gaussian_model.create_from_pcd(pcd, spatial_lr_scale=scale * 0.01)
    print(f"Initialized {gaussian_model._xyz.shape[0]} Gaussians")

    # Prepare image for training
    image_tensor = torch.tensor(image, dtype=torch.float32).cuda() / 255.0

    # Run training
    print("Starting training...")
    trainer = Trainer(
        gaussian_model=gaussian_model,
        cameras=[camera],
        images=[image_tensor],
        output_dir=str(output_path / "training_output"),
        iterations=args.iterations,
        lr=args.lr,
    )
    trainer.train()

    # Save model
    model_path = output_path / "model.ply"
    gaussian_model.save_ply(str(model_path))
    print(f"Model saved to {model_path}")
    print("Done!")


if __name__ == "__main__":
    main()
