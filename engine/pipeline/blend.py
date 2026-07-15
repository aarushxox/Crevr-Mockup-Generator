import cv2
import numpy as np

def match_histogram_lab(design, base_roi):
    """
    Match the color temperature/lighting tone of the design to the base product image in LAB space.
    """
    design_lab = cv2.cvtColor(design, cv2.COLOR_BGR2LAB).astype(np.float32)
    base_lab = cv2.cvtColor(base_roi, cv2.COLOR_BGR2LAB).astype(np.float32)

    for c in [1, 2]:
        d_mean = np.mean(design_lab[:, :, c])
        d_std = np.std(design_lab[:, :, c])
        b_mean = np.mean(base_lab[:, :, c])
        b_std = np.std(base_lab[:, :, c])

        if d_std > 0:
            design_lab[:, :, c] = ((design_lab[:, :, c] - d_mean) / d_std) * b_std + b_mean
        else:
            design_lab[:, :, c] = design_lab[:, :, c] - d_mean + b_mean

    design_lab = np.clip(design_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(design_lab, cv2.COLOR_LAB2BGR)

def blend_multiply(foreground, background):
    """
    Multiply blend mode. Out = (Foreground * Background) / 255.
    """
    fg_f = foreground.astype(np.float32)
    bg_f = background.astype(np.float32)
    result = (fg_f * bg_f) / 255.0
    return np.clip(result, 0, 255).astype(np.uint8)

def blend_screen(foreground, background):
    """
    Screen blend mode. Out = 255 - ((255 - Foreground) * (255 - Background)) / 255.
    """
    fg_f = foreground.astype(np.float32)
    bg_f = background.astype(np.float32)
    result = 255.0 - ((255.0 - fg_f) * (255.0 - bg_f)) / 255.0
    return np.clip(result, 0, 255).astype(np.uint8)

def blend_lighting(warped_design, lighting_map, blend_mode="multiply"):
    """
    Blend lighting/shadow map over the warped design.
    """
    if lighting_map is None:
        return warped_design

    if len(lighting_map.shape) == 2 or lighting_map.shape[2] == 1:
        if len(lighting_map.shape) == 2:
            lighting_map = np.stack([lighting_map] * 3, axis=2)
        else:
            lighting_map = np.repeat(lighting_map, 3, axis=2)

    h, w = warped_design.shape[:2]
    if lighting_map.shape[:2] != (h, w):
        lighting_map = cv2.resize(lighting_map, (w, h), interpolation=cv2.INTER_LINEAR)

    if blend_mode.lower() == "multiply":
        return blend_multiply(warped_design, lighting_map)
    elif blend_mode.lower() == "screen":
        return blend_screen(warped_design, lighting_map)
    else:
        return blend_multiply(warped_design, lighting_map)
