import cv2
import numpy as np

def feather_mask(mask, radius=5):
    """
    Feather mask edges using Gaussian blur.
    """
    if radius <= 0:
        return mask.astype(np.float32) / 255.0

    if len(mask.shape) == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask.astype(np.uint8)

    ksize = 2 * radius + 1
    blurred = cv2.GaussianBlur(mask_gray, (ksize, ksize), 0)
    return blurred.astype(np.float32) / 255.0

def composite_images(base_image, warped_design, mask, feather_radius=5):
    """
    Composite warped_design onto base_image using the mask, with edge feathering.
    """
    h, w = base_image.shape[:2]
    if warped_design.shape[:2] != (h, w):
        warped_design = cv2.resize(warped_design, (w, h), interpolation=cv2.INTER_CUBIC)

    if mask.shape[:2] != (h, w):
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        mask_resized = mask

    alpha = feather_mask(mask_resized, radius=feather_radius)
    if len(alpha.shape) == 2:
        alpha = np.stack([alpha] * 3, axis=2)

    base_f = base_image.astype(np.float32)
    design_f = warped_design.astype(np.float32)

    composited = design_f * alpha + base_f * (1.0 - alpha)
    return np.clip(composited, 0, 255).astype(np.uint8)
