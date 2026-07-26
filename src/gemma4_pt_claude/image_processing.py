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
except Exception:
    # Any failure (missing, or a broken native build) means "no PIL support".
    PILImage = None  # type: ignore[assignment, misc]

from .config import EncoderFreeVisionConfig, VisionConfig


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
# Patch merging (encoder-free variants)
# ---------------------------------------------------------------------------

def merge_patches(
        patches: torch.Tensor,
        position_ids: torch.Tensor,
        output_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge ``k x k`` spatially adjacent patches into single larger patches.

    Encoder-free variants have no vision tower, so the spatial pooling the tower
    models do *after* their encoder happens here instead: ``k x k`` groups of
    ``patch_size`` patches become one ``k * patch_size`` patch.

    Args:
        patches: ``[L, patch_size**2 * 3]``.
        position_ids: ``[L, 2]`` — (x, y) grid coords.
        output_length: number of merged patches; ``L`` must equal
            ``output_length * k**2``.

    Returns:
        ``(merged [output_length, (k*patch_size)**2 * 3], positions [output_length, 2])``
    """
    patch_dim = patches.shape[-1]
    patch_size = math.isqrt(patch_dim // 3)
    if patch_size * patch_size * 3 != patch_dim:
        raise ValueError(f"Patch dim {patch_dim} is not patch_size**2 * 3")

    k = math.isqrt(patches.shape[0] // output_length)
    if k * k * output_length != patches.shape[0]:
        raise ValueError(f"Cannot merge {patches.shape[0]} patches into {output_length}")

    # Order patches so each k x k kernel is contiguous, in raster order within
    # the kernel and raster order across kernels.
    max_x = position_ids[..., 0].max() + 1
    kernel_idx = torch.div(position_ids, k, rounding_mode="floor")
    kernel_start = k * k * kernel_idx[..., 0] + k * max_x * kernel_idx[..., 1]
    within = torch.remainder(position_ids, k)
    order = (within[..., 0] + within[..., 1] * k + kernel_start).long().argsort()

    ordered = patches[order]
    # [L, p*p*3] -> [out, k, k, p, p, 3] -> [out, k, p, k, p, 3] -> [out, (k*p)**2*3]
    ordered = ordered.reshape(output_length, k, k, patch_size, patch_size, 3)
    ordered = ordered.permute(0, 1, 3, 2, 4, 5)
    merged = ordered.reshape(output_length, (k * patch_size) ** 2 * 3)

    # Merged position is the kernel index, i.e. the min coord in the group.
    merged_positions = torch.div(position_ids[order], k, rounding_mode="floor")
    merged_positions = merged_positions.reshape(output_length, k * k, 2).amin(dim=1)
    return merged, merged_positions.to(torch.long)


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
        merge: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Preprocess a single image for the Gemma4 vision path.

    Args:
        image: PIL Image or ``[C, H, W]`` tensor.
        patch_size: patch side length.
        max_patches: maximum number of patches (from ``VisionConfig.max_patches``).
        pooling_kernel_size: spatial pooling kernel side.
        merge: merge ``pooling_kernel_size`` square groups of patches into single
            larger patches, for encoder-free variants that have no vision tower.

    Returns:
        ``(patches, position_ids, num_soft_tokens)`` where patches is
        ``[num_patches, patch_dim]``, position_ids is ``[num_patches, 2]``,
        and num_soft_tokens is the number of output tokens after pooling.
        With ``merge=True`` there is exactly one patch per soft token.
    """
    # Convert PIL to tensor (via the raw buffer, so numpy is not required)
    if PILImage is not None and isinstance(image, PILImage.Image):
        image = image.convert("RGB")
        width, height = image.size
        buffer = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        image = buffer.view(height, width, 3).permute(2, 0, 1).float() / 255.0
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
    if merge:
        # Encoder-free variants pool before the projection, not after a tower.
        patches, position_ids = merge_patches(patches, position_ids, num_soft_tokens)
    return patches, position_ids, num_soft_tokens


# ---------------------------------------------------------------------------
# Batch preprocessing
# ---------------------------------------------------------------------------

def preprocess_images(
        images: "Sequence[PILImage.Image | torch.Tensor]",
        config: "VisionConfig | EncoderFreeVisionConfig",
) -> dict[str, torch.Tensor | list[int]]:
    """Preprocess a batch of images for the Gemma4 vision path.

    Handles both the tower configs and the encoder-free configs; for the latter
    patches are merged and padded to ``output_length`` rather than ``max_patches``.

    Args:
        images: list of PIL Images or ``[C, H, W]`` tensors.
        config: ``VisionConfig`` or ``EncoderFreeVisionConfig``.

    Returns:
        Dict with keys:
        - ``pixel_values``: ``[B, num_patches, patch_dim]``
        - ``image_position_ids``: ``[B, num_patches, 2]``
        - ``num_soft_tokens_per_image``: ``list[int]``
    """
    merge = isinstance(config, EncoderFreeVisionConfig)
    max_patches = config.max_patches
    pad_length = config.output_length if merge else max_patches
    all_patches = []
    all_position_ids = []
    num_soft_tokens_per_image = []

    for img in images:
        patches, pos_ids, n_soft = preprocess_image(
            img, config.patch_size, max_patches, config.pooling_kernel_size, merge=merge,
        )
        patches, pos_ids = pad_to_max_patches(patches, pos_ids, pad_length)
        all_patches.append(patches)
        all_position_ids.append(pos_ids)
        num_soft_tokens_per_image.append(n_soft)

    return {
        "pixel_values": torch.stack(all_patches, dim=0),
        "image_position_ids": torch.stack(all_position_ids, dim=0),
        "num_soft_tokens_per_image": num_soft_tokens_per_image,
    }
