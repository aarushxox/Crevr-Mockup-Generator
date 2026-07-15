import cv2
import numpy as np

def apply_displacement(warped_design, displacement_map, intensity=10):
    """
    Apply displacement to locally shift warped_design pixels based on the displacement map.
    The displacement map is neutral gray (128). Lighter pushes one way, darker the other.
    """
    if displacement_map is None or intensity == 0:
        return warped_design

    h, w = warped_design.shape[:2]
    # Resize displacement map to match warped design if they don't match
    if displacement_map.shape[:2] != (h, w):
        disp_map = cv2.resize(displacement_map, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        disp_map = displacement_map

    # Ensure grayscale
    if len(disp_map.shape) == 3:
        disp_map = cv2.cvtColor(disp_map, cv2.COLOR_BGR2GRAY)

    # Normalize map to [-1, 1] relative to neutral gray (128)
    disp_norm = (disp_map.astype(np.float32) - 128.0) / 128.0

    # Compute gradients of the displacement map to get directional offset fields dx, dy
    grad_x = cv2.Sobel(disp_norm, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(disp_norm, cv2.CV_32F, 0, 1, ksize=3)

    # Create coordinate grid
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

    # Compute offset coordinates
    map_x = x_coords + grad_x * intensity
    map_y = y_coords + grad_y * intensity

    # Apply remap
    displaced = cv2.remap(
        warped_design,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )
    return displaced
