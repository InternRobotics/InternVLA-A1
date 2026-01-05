import torch
import torch.nn.functional as F
import math


def make_bool_mask(*dims: int) -> torch.BoolTensor:
    """Make a boolean mask for the given dimensions.

    Example:
        _make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        _make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tensor of booleans.
    """
    lens = [abs(d) for d in dims]
    flags = [d > 0 for d in dims]
    parts = [torch.full((l,), f, dtype=torch.bool) for l, f in zip(lens, flags)]
    return torch.cat(parts) if parts else torch.empty(0, dtype=torch.bool)



def resize_with_pad(
    image: torch.Tensor,
    target_h: int,
    target_w: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """
    Resize a tensor to target size while keeping aspect ratio.
    Supports both [3, H, W] and [B, 3, H, W] inputs.
    Vectorized over the batch dimension (no Python loop).

    Args:
        image: torch.Tensor, shape [3, H, W] or [B, 3, H, W], values in [0, 1].
        target_h: target height.
        target_w: target width.
        mode: interpolation mode ('bilinear', 'nearest', etc.).
    Returns:
        torch.Tensor of shape [3, target_h, target_w] or [B, 3, target_h, target_w].
    """
    if image.ndim == 3:
        # [3, H, W] -> [1, 3, H, W]
        image = image.unsqueeze(0)
        squeeze_back = True
    elif image.ndim == 4:
        squeeze_back = False
    else:
        raise ValueError(f"Expected tensor of shape [3,H,W] or [B,3,H,W], got {image.shape}")

    B, C, H, W = image.shape

    if H == target_h and W == target_w:
        return image.squeeze(0) if squeeze_back else image

    scale = min(target_h / H, target_w / W)
    new_h, new_w = int(round(H * scale)), int(round(W * scale))

    image_resized = F.interpolate(
        image,
        size=(new_h, new_w),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )  # [B, 3, new_h, new_w]

    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left

    image_padded = F.pad(image_resized, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
    image_padded = image_padded.clamp(0.0, 1.0)

    # sanity check
    assert image_padded.shape[-2:] == (target_h, target_w)

    return image_padded.squeeze(0) if squeeze_back else image_padded


def resize_center_crop(
    image: torch.Tensor,
    target_h: int,
    target_w: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """
    Resize an image or a batch of images so that the shortest side is scaled
    to at least the target size, then apply a center crop to exactly
    [C, target_h, target_w] or [B, C, target_h, target_w].

    Supports:
        - [C, H, W]
        - [B, C, H, W]
    and performs vectorized operations over the batch dimension.

    Args:
        image: Input tensor of shape [C,H,W] or [B,C,H,W].
        target_h: Target height after cropping.
        target_w: Target width after cropping.
        mode: Interpolation mode for resizing.

    Returns:
        Tensor of shape [C, target_h, target_w] or [B, C, target_h, target_w].
    """

    # ------------------------------------------------------
    # Normalize input shape to batch mode
    # ------------------------------------------------------
    if image.ndim == 3:
        # Convert [C, H, W] → [1, C, H, W]
        assert image.shape[0] in (1, 3), f"Expected [C,H,W], got {image.shape}"
        image = image.unsqueeze(0)
        squeeze_back = True
    elif image.ndim == 4:
        squeeze_back = False
    else:
        raise ValueError(f"Expected [C,H,W] or [B,C,H,W], got {image.shape}")

    B, C, H, W = image.shape

    # If the image is already the correct size, return early
    if (H, W) == (target_h, target_w):
        return image.squeeze(0) if squeeze_back else image

    # ------------------------------------------------------
    # Compute scale factor so the shortest side >= target size
    # ------------------------------------------------------
    # Use max() to ensure both resized dimensions are >= target dimensions.
    scale = max(target_h / H, target_w / W)

    # Ceil prevents rounding down to a size smaller than required
    new_h = int(math.ceil(H * scale))
    new_w = int(math.ceil(W * scale))

    # ------------------------------------------------------
    # Resize the batch in a single vectorized operation
    # ------------------------------------------------------
    align_corners = False if mode in ("bilinear", "bicubic") else None
    x = F.interpolate(
        image,
        size=(new_h, new_w),
        mode=mode,
        align_corners=align_corners,
    )  # [B, C, new_h, new_w]

    # ------------------------------------------------------
    # Center crop to the target size
    # ------------------------------------------------------
    top = max((new_h - target_h) // 2, 0)
    left = max((new_w - target_w) // 2, 0)

    x = x[:, :, top : top + target_h, left : left + target_w]

    # ------------------------------------------------------
    # (Rare) If rounding left us with slightly smaller output,
    # add minimal padding to reach the exact target size
    # ------------------------------------------------------
    cur_h, cur_w = x.shape[-2:]
    pad_h = target_h - cur_h
    pad_w = target_w - cur_w

    if pad_h > 0 or pad_w > 0:
        # F.pad padding order: (left, right, top, bottom)
        x = F.pad(
            x,
            (0, max(pad_w, 0), 0, max(pad_h, 0)),  # Only pad right/bottom
        )

    # Clamp values to valid range
    x = x.clamp(0.0, 1.0)

    # Sanity check
    assert x.shape[-2:] == (target_h, target_w)

    # ------------------------------------------------------
    # Return a squeezed tensor if input was a single image
    # ------------------------------------------------------
    return x.squeeze(0) if squeeze_back else x
