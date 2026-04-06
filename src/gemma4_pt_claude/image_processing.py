"""Image preprocessing for Gemma4 vision encoder.

Converts PIL images (or tensors) into patchified inputs with 2-D position IDs,
matching the HuggingFace Gemma4ImageProcessor pipeline.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None  # type: ignore[assignment, misc]

from .config import VisionConfig


# ---------------------------------------------------------------------------
# Resize target computation
# ---------------------------------------------------------------------------

def get_target_dimensions(
        height: int,
        width: int,
        patch_size: int,
        max_patches: int,
        pooling_kernel_size: int,
) -> tuple[int, int]:
    """Compute aspect-ratio-preserving resize dimensions.

    The target fits within ``max_patches`` patches and has both sides
    divisible by ``pooling_kernel_size * patch_size``.
    """
    total_px = height * width
    target_px = max_patches * (patch_size ** 2)
    factor = math.sqrt(target_px / total_px)
    ideal_h = factor * height
    ideal_w = factor * width
    side_mult = pooling_kernel_size * patch_size

    target_h = int(math.floor(ideal_h / side_mult)) * side_mult
    target_w = int(math.floor(ideal_w / side_mult)) * side_mult

    if target_h == 0 and target_w == 0:
        raise ValueError(
            f"Image too small: resized to 0x0. Dimensions must be divisible by "
            f"pooling_kernel_size * patch_size = {side_mult}."
        )
    if target_h == 0:
        target_h = side_mult
    if target_w == 0:
        target_w = side_mult
    return target_h, target_w


# ---------------------------------------------------------------------------
# Patchify
# ---------------------------------------------------------------------------

def patchify(
        image: torch.Tensor,
        patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a ``[C, H, W]`` image tensor into patches with position IDs.

    Args:
        image: ``[C, H, W]`` in [0, 1] range.
        patch_size: side length of each square patch.

    Returns:
        ``(patches, position_ids)`` where patches is
        ``[num_patches, patch_size**2 * C]`` and position_ids is
        ``[num_patches, 2]`` with (x, y) grid coordinates.
    """
    C, H, W = image.shape
    ph = H // patch_size
    pw = W // patch_size

    # Reshape: [C, ph, patch_size, pw, patch_size] → [ph, pw, patch_size, patch_size, C]
    patches = image.reshape(C, ph, patch_size, pw, patch_size)
    patches = patches.permute(1, 3, 2, 4, 0)  # [ph, pw, ps, ps, C]
    patches = patches.reshape(ph * pw, -1)  # [num_patches, ps*ps*C]

    # Position IDs: meshgrid of (x, y)
    grid = torch.meshgrid(
        torch.arange(pw, device=image.device),
        torch.arange(ph, device=image.device),
        indexing="xy",
    )
    position_ids = torch.stack(grid, dim=-1).reshape(-1, 2)  # [num_patches, 2]

    return patches, position_ids


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------

def pad_to_max_patches(
        patches: torch.Tensor,
        position_ids: torch.Tensor,
        max_patches: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad patches and position_ids to ``max_patches`` length.

    Patches are zero-padded; position_ids are padded with -1.
    """
    n = patches.shape[0]
    pad_len = max_patches - n
    if pad_len <= 0:
        return patches, position_ids

    patches = F.pad(patches, (0, 0, 0, pad_len), value=0)
    position_ids = F.pad(position_ids, (0, 0, 0, pad_len), value=-1)
    return patches, position_ids


# ---------------------------------------------------------------------------
# Single image preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(
        image: "PILImage.Image | torch.Tensor",
        patch_size: int,
        max_patches: int,
        pooling_kernel_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Preprocess a single image for the Gemma4 vision encoder.

    Args:
        image: PIL Image or ``[C, H, W]`` tensor.
        patch_size: patch side length.
        max_patches: maximum number of patches (from ``VisionConfig.max_patches``).
        pooling_kernel_size: spatial pooling kernel side.

    Returns:
        ``(patches, position_ids, num_soft_tokens)`` where patches is
        ``[num_patches, patch_dim]``, position_ids is ``[num_patches, 2]``,
        and num_soft_tokens is the number of output tokens after pooling.
    """
    # Convert PIL to tensor
    if PILImage is not None and isinstance(image, PILImage.Image):
        image = image.convert("RGB")
        image = torch.from_numpy(
            __import__("numpy").array(image)
        ).permute(2, 0, 1).float() / 255.0
    elif isinstance(image, torch.Tensor):
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        elif image.dtype != torch.float32:
            image = image.float()
    else:
        raise TypeError(f"Expected PIL Image or torch.Tensor, got {type(image)}")

    C, H, W = image.shape

    # Aspect-ratio-preserving resize
    target_h, target_w = get_target_dimensions(H, W, patch_size, max_patches, pooling_kernel_size)
    if target_h != H or target_w != W:
        image = F.interpolate(
            image.unsqueeze(0),
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).squeeze(0).clamp(0.0, 1.0)

    patches, position_ids = patchify(image, patch_size)
    num_soft_tokens = patches.shape[0] // (pooling_kernel_size ** 2)
    return patches, position_ids, num_soft_tokens


# ---------------------------------------------------------------------------
# Batch preprocessing
# ---------------------------------------------------------------------------

def preprocess_images(
        images: "Sequence[PILImage.Image | torch.Tensor]",
        config: VisionConfig,
) -> dict[str, torch.Tensor | list[int]]:
    """Preprocess a batch of images for the Gemma4 vision encoder.

    Args:
        images: list of PIL Images or ``[C, H, W]`` tensors.
        config: ``VisionConfig`` instance.

    Returns:
        Dict with keys:
        - ``pixel_values``: ``[B, max_patches, patch_dim]``
        - ``image_position_ids``: ``[B, max_patches, 2]``
        - ``num_soft_tokens_per_image``: ``list[int]``
    """
    max_patches = config.max_patches
    all_patches = []
    all_position_ids = []
    num_soft_tokens_per_image = []

    for img in images:
        patches, pos_ids, n_soft = preprocess_image(
            img, config.patch_size, max_patches, config.pooling_kernel_size,
        )
        patches, pos_ids = pad_to_max_patches(patches, pos_ids, max_patches)
        all_patches.append(patches)
        all_position_ids.append(pos_ids)
        num_soft_tokens_per_image.append(n_soft)

    return {
        "pixel_values": torch.stack(all_patches, dim=0),
        "image_position_ids": torch.stack(all_position_ids, dim=0),
        "num_soft_tokens_per_image": num_soft_tokens_per_image,
    }
