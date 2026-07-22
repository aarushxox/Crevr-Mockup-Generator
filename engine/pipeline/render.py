import cv2
import numpy as np
import os
import json
from engine.pipeline.warp import get_warp_matrix, warp_design
from engine.pipeline.displacement import apply_displacement
from engine.pipeline.blend import blend_lighting, match_histogram_lab
from engine.pipeline.mask import composite_images

def render_mockup(template_folder, design_image, transform_options, export_options):
    """
    Core Runtime Render Engine.
    Executes the entire deterministic OpenCV/NumPy rendering pipeline.
    Returns: (composited_result, warnings_list)
    """
    warnings = []

    # Load metadata
    metadata_path = os.path.join(template_folder, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Load template assets
    base_image = cv2.imread(os.path.join(template_folder, metadata["base_image"]))
    mask_image = cv2.imread(os.path.join(template_folder, metadata["mask_image"]), cv2.IMREAD_GRAYSCALE)
    displacement_image = cv2.imread(os.path.join(template_folder, metadata["displacement_image"]), cv2.IMREAD_GRAYSCALE)
    lighting_image = cv2.imread(os.path.join(template_folder, metadata["lighting_image"]))

    h_base, w_base = base_image.shape[:2]

    # Preprocess design: Strip alpha if any or handle transparency
    has_alpha = design_image.shape[2] == 4
    if has_alpha:
        design_bgr = design_image[:, :, :3]
        design_alpha = design_image[:, :, 3]
    else:
        design_bgr = design_image
        design_alpha = np.ones_like(design_bgr[:, :, 0]) * 255

    # 1. Physical size and DPI dynamic scaling
    dpi = export_options.get("dpi") or metadata.get("target_dpi") or 300
    physical_size_mm = metadata.get("physical_size_mm")

    scale_x = 1.0
    scale_y = 1.0

    if physical_size_mm:
        # Expected physical_size_mm is a list [width_mm, height_mm]
        target_w = int(round(physical_size_mm[0] / 25.4 * dpi))
        target_h = int(round(physical_size_mm[1] / 25.4 * dpi))

        scale_x = target_w / w_base
        scale_y = target_h / h_base

        base_image = cv2.resize(base_image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        mask_image = cv2.resize(mask_image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        displacement_image = cv2.resize(displacement_image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        lighting_image = cv2.resize(lighting_image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        h_base, w_base = target_h, target_w

    # 2. Export Resolution Clamping / Safeguard
    max_res = metadata.get("export_max_resolution_px", [3000, 3000])
    if w_base > max_res[0] or h_base > max_res[1]:
        scale_f = min(max_res[0] / w_base, max_res[1] / h_base)
        clamped_w = int(round(w_base * scale_f))
        clamped_h = int(round(h_base * scale_f))

        warnings.append(f"Resolution clamped from {w_base}x{h_base} to maximum limit of {clamped_w}x{clamped_h}px.")

        scale_x *= scale_f
        scale_y *= scale_f

        base_image = cv2.resize(base_image, (clamped_w, clamped_h), interpolation=cv2.INTER_CUBIC)
        mask_image = cv2.resize(mask_image, (clamped_w, clamped_h), interpolation=cv2.INTER_LINEAR)
        displacement_image = cv2.resize(displacement_image, (clamped_w, clamped_h), interpolation=cv2.INTER_LINEAR)
        lighting_image = cv2.resize(lighting_image, (clamped_w, clamped_h), interpolation=cv2.INTER_LINEAR)

        h_base, w_base = clamped_h, clamped_w

    # 3. Option: Tone matching in LAB space
    if export_options.get("color_correct", True):
        sample_roi = cv2.bitwise_and(base_image, base_image, mask=mask_image)
        design_bgr = match_histogram_lab(design_bgr, sample_roi)

    # Reconstruct design image with matched BGR and alpha
    processed_design = np.zeros((design_bgr.shape[0], design_bgr.shape[1], 4), dtype=np.uint8)
    processed_design[:, :, :3] = design_bgr
    processed_design[:, :, 3] = design_alpha

    # 4. Fit and Transform into the design zone corners
    dst_corners = np.array(metadata["design_zone_corners"], dtype=np.float32)
    # Apply dynamic scaling factors to corners
    dst_corners[:, 0] *= scale_x
    dst_corners[:, 1] *= scale_y

    hd, wd = processed_design.shape[:2]
    src_corners = np.array([[0, 0], [wd, 0], [wd, hd], [0, hd]], dtype=np.float32)

    x_min, y_min = dst_corners.min(axis=0)
    x_max, y_max = dst_corners.max(axis=0)
    box_w = x_max - x_min
    box_h = y_max - y_min

    inter_corners = np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max]
    ], dtype=np.float32)

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    tx = transform_options.get("x", 0.0) * scale_x
    ty = transform_options.get("y", 0.0) * scale_y
    scale = transform_options.get("scale", 1.0)
    rotation = transform_options.get("rotation", 0.0)

    R = cv2.getRotationMatrix2D((cx, cy), rotation, scale)
    R[0, 2] += tx
    R[1, 2] += ty

    adjusted_dst_corners = []
    for pt in inter_corners:
        px = R[0, 0] * pt[0] + R[0, 1] * pt[1] + R[0, 2]
        py = R[1, 0] * pt[0] + R[1, 1] * pt[1] + R[1, 2]
        adjusted_dst_corners.append([px, py])
    adjusted_dst_corners = np.array(adjusted_dst_corners, dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(src_corners, adjusted_dst_corners)

    # 5. Warp the design
    warped_design_rgba = warp_design(processed_design, matrix, (w_base, h_base))
    warped_design_bgr = warped_design_rgba[:, :, :3]
    warped_design_alpha = warped_design_rgba[:, :, 3]

    # 6. Apply displacement (folds)
    fold_intensity = metadata.get("fold_intensity", 15)
    displaced_bgr = apply_displacement(warped_design_bgr, displacement_image, fold_intensity)
    displaced_alpha = apply_displacement(warped_design_alpha, displacement_image, fold_intensity)

    # 7. Apply lighting blend map
    lit_bgr = blend_lighting(displaced_bgr, lighting_image, blend_mode="multiply")

    # Reconstruct final design to overlay
    final_design = np.zeros_like(warped_design_rgba)
    final_design[:, :, :3] = lit_bgr
    final_design[:, :, 3] = cv2.bitwise_and(displaced_alpha, mask_image)

    # 8. Edge integration (feathering) & Compositing
    final_mask = final_design[:, :, 3]
    feather_radius = export_options.get("feather_radius", 3)
    composited_result = composite_images(base_image, final_design[:, :, :3], final_mask, feather_radius)

    return composited_result, warnings
