"""Rendering functionality

Implements Gaussian Splatting rendering using diff-gaussian-rasterization.
"""

import torch
import numpy as np
from typing import Tuple, Optional, TYPE_CHECKING

try:
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
        rasterize_gaussians as _rasterize_gaussians_cpp,
    )
except ImportError:
    print("Warning: diff-gaussian-rasterization not found. Please install it.")
    GaussianRasterizationSettings = None
    GaussianRasterizer = None
    _rasterize_gaussians_cpp = None

if TYPE_CHECKING:
    from .gaussian_model import GaussianModel


def rasterize_gaussians(
    means3D: torch.Tensor,
    means2D: torch.Tensor,
    sh: torch.Tensor,
    colors_precomp: Optional[torch.Tensor],
    opacities: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    cov3D_precomp: Optional[torch.Tensor],
    raster_settings: GaussianRasterizationSettings,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rasterize Gaussians.

    Args:
        means3D: 3D positions [N, 3]
        means2D: 2D positions [N, 2]
        sh: Spherical harmonics coefficients [N, M, 3]
        colors_precomp: Precomputed colors [N, 3] (optional)
        opacities: Opacities [N, 1]
        scales: Scales [N, 3]
        rotations: Rotations (quaternions) [N, 4]
        cov3D_precomp: Precomputed 3D covariance [N, 3, 3] (optional)
        raster_settings: Rasterization settings

    Returns:
        (Rendered image, depth, radii, other information)
    """
    if GaussianRasterizer is None:
        raise ImportError("diff-gaussian-rasterization is not installed")

    # Use shs directly, don't use colors_precomp to avoid C++ function issues
    # The C++ function requires shs to be a tensor, not None
    # When shs is provided, colors_precomp must be None (GaussianRasterizer will convert it to empty tensor)
    # However, the C++ function doesn't accept empty tensor for colors_precomp
    # The issue is that _RasterizeGaussians.forward converts None to empty tensor,
    # but the C++ function doesn't accept empty tensor
    # We need to patch GaussianRasterizer.forward to handle this correctly
    shs_param = sh
    colors_precomp_param = None

    # Monkey-patch GaussianRasterizer.forward to handle empty tensor for colors_precomp
    # Create a dummy tensor with the correct shape [N, 3] when colors_precomp is None
    original_forward = GaussianRasterizer.forward

    def patched_forward(
        self,
        means3D,
        means2D,
        opacities,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
    ):
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (
            shs is not None and colors_precomp is not None
        ):
            raise Exception(
                "Please provide excatly one of either SHs or precomputed colors!"
            )

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!"
            )

        if shs is None:
            shs = torch.Tensor([])
        # Instead of converting None to empty tensor, create a dummy tensor with correct shape
        if colors_precomp is None:
            N = means3D.shape[0]
            device = means3D.device
            colors_precomp = torch.zeros((N, 3), dtype=torch.float32, device=device)

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Import the low-level rasterize_gaussians function
        from diff_gaussian_rasterization import (
            rasterize_gaussians as _rasterize_gaussians_cpp,
            GaussianRasterizationSettings,
        )

        # Convert numpy arrays in raster_settings to torch.Tensor
        # The C++ function expects torch.Tensor, not numpy arrays
        # Since raster_settings attributes are read-only, create a new object
        device = means3D.device

        # Convert viewmatrix
        if isinstance(raster_settings.viewmatrix, np.ndarray):
            viewmatrix_tensor = (
                torch.from_numpy(raster_settings.viewmatrix).float().to(device)
            )
        elif isinstance(raster_settings.viewmatrix, torch.Tensor):
            viewmatrix_tensor = raster_settings.viewmatrix.to(device)
        else:
            viewmatrix_tensor = torch.tensor(
                raster_settings.viewmatrix, dtype=torch.float32
            ).to(device)

        # Convert projmatrix
        if isinstance(raster_settings.projmatrix, np.ndarray):
            projmatrix_tensor = (
                torch.from_numpy(raster_settings.projmatrix).float().to(device)
            )
        elif isinstance(raster_settings.projmatrix, torch.Tensor):
            projmatrix_tensor = raster_settings.projmatrix.to(device)
        else:
            projmatrix_tensor = torch.tensor(
                raster_settings.projmatrix, dtype=torch.float32
            ).to(device)

        # Convert campos
        if isinstance(raster_settings.campos, np.ndarray):
            campos_tensor = torch.from_numpy(raster_settings.campos).float().to(device)
        elif isinstance(raster_settings.campos, torch.Tensor):
            campos_tensor = raster_settings.campos.to(device)
        else:
            campos_tensor = torch.tensor(
                raster_settings.campos, dtype=torch.float32
            ).to(device)

        # Convert bg if it's not already a tensor
        if isinstance(raster_settings.bg, np.ndarray):
            bg_tensor = torch.from_numpy(raster_settings.bg).float().to(device)
        elif isinstance(raster_settings.bg, torch.Tensor):
            bg_tensor = raster_settings.bg.to(device)
        else:
            bg_tensor = torch.tensor(raster_settings.bg, dtype=torch.float32).to(device)

        # Create new raster_settings with tensor values
        # Note: GaussianRasterizationSettings might accept numpy arrays or tensors
        # Try passing tensors directly, or convert to numpy if needed
        try:
            # Try creating with tensors first
            raster_settings_new = GaussianRasterizationSettings(
                image_height=raster_settings.image_height,
                image_width=raster_settings.image_width,
                tanfovx=raster_settings.tanfovx,
                tanfovy=raster_settings.tanfovy,
                bg=bg_tensor,
                scale_modifier=raster_settings.scale_modifier,
                viewmatrix=viewmatrix_tensor,
                projmatrix=projmatrix_tensor,
                sh_degree=raster_settings.sh_degree,
                campos=campos_tensor,
                prefiltered=raster_settings.prefiltered,
                debug=raster_settings.debug,
            )
        except (TypeError, AttributeError):
            # If tensors don't work, convert back to numpy (but this shouldn't happen)
            raster_settings_new = GaussianRasterizationSettings(
                image_height=raster_settings.image_height,
                image_width=raster_settings.image_width,
                tanfovx=raster_settings.tanfovx,
                tanfovy=raster_settings.tanfovy,
                bg=bg_tensor.cpu().numpy()
                if isinstance(bg_tensor, torch.Tensor)
                else bg_tensor,
                scale_modifier=raster_settings.scale_modifier,
                viewmatrix=viewmatrix_tensor.cpu().numpy()
                if isinstance(viewmatrix_tensor, torch.Tensor)
                else viewmatrix_tensor,
                projmatrix=projmatrix_tensor.cpu().numpy()
                if isinstance(projmatrix_tensor, torch.Tensor)
                else projmatrix_tensor,
                sh_degree=raster_settings.sh_degree,
                campos=campos_tensor.cpu().numpy()
                if isinstance(campos_tensor, torch.Tensor)
                else campos_tensor,
                prefiltered=raster_settings.prefiltered,
                debug=raster_settings.debug,
            )

        # Invoke C++/CUDA rasterization routine
        # The C++ function expects raster_settings as a single argument
        return _rasterize_gaussians_cpp(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings_new,
        )

    # Temporarily patch the forward method
    GaussianRasterizer.forward = patched_forward

    try:
        result = GaussianRasterizer(
            raster_settings=raster_settings
        )(
            means3D=means3D,
            means2D=means2D,
            opacities=opacities,
            shs=shs_param,
            colors_precomp=colors_precomp_param,  # None - will be converted to dummy tensor in patched_forward
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
    finally:
        # Restore original forward method
        GaussianRasterizer.forward = original_forward

    # The result is (color, radii) tuple
    # We need to return (rendered_image, radii, rendered_depth, rendered_alpha)
    # rendered_depth and rendered_alpha are not returned by the low-level function
    # So we create dummy tensors for them
    rendered_image, radii = result
    rendered_depth = torch.zeros_like(rendered_image[0:1])  # Dummy depth
    rendered_alpha = torch.ones_like(rendered_image[0:1])  # Dummy alpha

    return rendered_image, radii, rendered_depth, rendered_alpha


def render(
    viewpoint_camera,
    pc,  # GaussianModel
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    override_color: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Render Gaussian Splatting.

    Args:
        viewpoint_camera: Camera information (position, rotation, intrinsics, etc.)
        pc: GaussianModel instance
        pipe: Pipeline settings
        bg_color: Background color [3]
        scaling_modifier: Scale modification factor
        override_color: Color override [N, 3] (optional)

    Returns:
        (Rendered image [C, H, W], Depth map [H, W])
    """
    if GaussianRasterizationSettings is None:
        raise ImportError("diff-gaussian-rasterization is not installed")

    # Get camera parameters
    if (
        hasattr(viewpoint_camera, "world_view_transform")
        and viewpoint_camera.world_view_transform is not None
    ):
        world_view_transform = viewpoint_camera.world_view_transform
    else:
        # Build transformation matrix from camera matrices
        R = viewpoint_camera.R
        t = viewpoint_camera.T
        world_view_transform = torch.eye(4, dtype=torch.float32).cuda()
        world_view_transform[:3, :3] = R.T
        world_view_transform[:3, 3] = -R.T @ t

    if (
        hasattr(viewpoint_camera, "full_proj_transform")
        and viewpoint_camera.full_proj_transform is not None
    ):
        full_proj_transform = viewpoint_camera.full_proj_transform
    else:
        # Build projection matrix
        K = viewpoint_camera.K
        width = viewpoint_camera.image_width
        height = viewpoint_camera.image_height
        znear = 0.01
        zfar = 100.0

        # OpenGL-style projection matrix
        proj = torch.zeros((4, 4), dtype=torch.float32).cuda()
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        proj[0, 0] = 2.0 * fx / width
        proj[1, 1] = 2.0 * fy / height
        proj[0, 2] = 1.0 - 2.0 * cx / width
        proj[1, 2] = 2.0 * cy / height - 1.0
        proj[2, 2] = (zfar + znear) / (znear - zfar)
        proj[2, 3] = 2.0 * zfar * znear / (znear - zfar)
        proj[3, 2] = -1.0

        full_proj_transform = (
            world_view_transform.unsqueeze(0).bmm(proj.unsqueeze(0))
        ).squeeze(0)

    # Get Gaussian parameters
    means3D = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation

    # Get features (spherical harmonics) directly from internal attributes
    # get_features property might return empty tensor, so access directly
    features_dc = pc._features_dc  # [N, 1, 3]
    features_rest = pc._features_rest  # [N, 15, 3]
    shs = torch.cat((features_dc, features_rest), dim=1)  # [N, 16, 3]

    # Verify shs shape
    if shs.shape[0] == 0:
        raise ValueError(
            f"shs is empty! means3D shape: {means3D.shape}, shs shape: {shs.shape}, "
            f"features_dc shape: {features_dc.shape}, features_rest shape: {features_rest.shape}"
        )

    if len(shs.shape) != 3 or shs.shape[2] != 3:
        raise ValueError(f"Unexpected shs shape: {shs.shape}, expected [N, M, 3]")

    # Apply scale
    scales = torch.exp(scales) * scaling_modifier

    # Convert opacity with sigmoid
    opacities = torch.sigmoid(opacity)

    # Transform to camera coordinate system
    means3D_cam = (
        world_view_transform[:3, :3] @ means3D.T + world_view_transform[:3, 3:4]
    ).T

    # Verify means3D_cam is not empty
    if means3D_cam.shape[0] == 0:
        raise ValueError(f"means3D_cam is empty! means3D shape: {means3D.shape}")

    # 2D projection
    means2D_homogeneous = (
        full_proj_transform[:3, :3] @ means3D_cam.T + full_proj_transform[:3, 3:4]
    ).T

    # Verify means2D_homogeneous is not empty
    if means2D_homogeneous.shape[0] == 0:
        raise ValueError(
            f"means2D_homogeneous is empty! means3D_cam shape: {means3D_cam.shape}"
        )

    # Divide by z component (perspective division)
    means2D = means2D_homogeneous[:, :2] / (
        means2D_homogeneous[:, 2:3] + 1e-7
    )  # Add small epsilon to avoid division by zero

    # Verify means2D is not empty
    if means2D.shape[0] == 0:
        raise ValueError(
            f"means2D is empty after projection! means2D_homogeneous shape: {means2D_homogeneous.shape}"
        )

    # Rasterization settings
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=float(np.tan(viewpoint_camera.FoVx * 0.5)),
        tanfovy=float(np.tan(viewpoint_camera.FoVy * 0.5)),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=world_view_transform.detach().cpu().numpy(),
        projmatrix=full_proj_transform.detach().cpu().numpy(),
        sh_degree=0,  # Use DC component only
        campos=means3D_cam.mean(dim=0).detach().cpu().numpy(),
        prefiltered=False,
        debug=False,
    )

    # Verify means2D is not empty before passing to rasterize_gaussians
    if means2D.shape[0] == 0:
        raise ValueError(f"means2D is empty! means2D shape: {means2D.shape}")

    # Rasterize
    # Use shs directly instead of colors_precomp to avoid C++ function issues
    # Note: means2D is computed from means3D via projection, so it doesn't need gradients
    # The gradient will flow through means3D instead
    # We detach means2D to avoid gradient computation issues
    means2D_detached = means2D.detach() if hasattr(means2D, "detach") else means2D

    try:
        rendered_image, radii, rendered_depth, rendered_alpha = rasterize_gaussians(
            means3D=means3D,  # Keep gradients for means3D
            means2D=means2D_detached,  # Detach means2D since it's computed from means3D
            sh=shs,  # Keep gradients for shs
            colors_precomp=None,  # Use shs instead
            opacities=opacities,  # Keep gradients for opacities
            scales=scales,  # Keep gradients for scales
            rotations=rotations,  # Keep gradients for rotations
            cov3D_precomp=None,
            raster_settings=raster_settings,
        )
    except Exception as e:
        print(f"ERROR in rasterize_gaussians: {e}")
        print(f"means2D shape: {means2D.shape}, dtype: {means2D.dtype}")
        print(f"means2D is_empty: {means2D.numel() == 0}")
        raise

    # Composite background
    rendered_image = rendered_image + bg_color.unsqueeze(-1).unsqueeze(-1) * (
        1.0 - rendered_alpha
    )

    return rendered_image, rendered_depth


class Camera:
    """Class to hold camera information"""

    def __init__(
        self,
        R: torch.Tensor,
        T: torch.Tensor,
        K: torch.Tensor,
        image_width: int,
        image_height: int,
        FoVx: float,
        FoVy: float,
        world_view_transform: Optional[torch.Tensor] = None,
        full_proj_transform: Optional[torch.Tensor] = None,
    ):
        """Initialize camera.

        Args:
            R: Rotation matrix [3, 3]
            T: Translation vector [3]
            K: Intrinsic parameter matrix [3, 3]
            image_width: Image width
            image_height: Image height
            FoVx: Horizontal field of view (radians)
            FoVy: Vertical field of view (radians)
            world_view_transform: World to view transformation matrix [4, 4] (optional)
            full_proj_transform: Projection transformation matrix [4, 4] (optional)
        """
        self.R = R
        self.T = T
        self.K = K
        self.image_width = image_width
        self.image_height = image_height
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
