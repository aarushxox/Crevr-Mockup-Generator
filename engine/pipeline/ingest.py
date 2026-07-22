import cv2
import numpy as np
import json
import os
import re

def clean_mask(mask):
    """
    Clean mask using morphological operations (Closing then Opening with a 5x5 kernel).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened

def ingest_raw_mockup(base_path, category, subtype, label, fold_intensity=15, physical_size_mm=None, target_dpi=300):
    """
    Ingestion Pipeline to pre-process raw mockup images into fully formed templates.
    Produces base.png, mask.png, displacement.png, lighting.png, and metadata.json.
    """
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Source file {base_path} not found.")

    img = cv2.imread(base_path)
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Generate mask & corners
    mask = np.zeros((h, w), dtype=np.uint8)
    corners = []

    if "laptop" in base_path:
        # Detect white screen area on laptop mockup
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        mask = clean_mask(thresh)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            peri = cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
            if len(approx) == 4:
                corners = approx.reshape(4, 2).tolist()
            else:
                x, y, bw, bh = cv2.boundingRect(largest)
                corners = [[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]]
        else:
            corners = [[int(w*0.1), int(h*0.15)], [int(w*0.9), int(h*0.15)], [int(w*0.9), int(h*0.65)], [int(w*0.1), int(h*0.65)]]
            mask[int(h*0.15):int(h*0.65), int(w*0.1):int(w*0.9)] = 255

    elif "t_shirt.png" in base_path:
        # mockup_t_shirt.png: white t-shirt flat lay.
        cy1, cy2 = int(h * 0.25), int(h * 0.65)
        cx1, cx2 = int(w * 0.28), int(w * 0.72)

        roi = gray[cy1:cy2, cx1:cx2]
        roi_mask = (roi > 150).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel)

        mask[cy1:cy2, cx1:cx2] = roi_mask
        corners = [[cx1, cy1], [cx2, cy1], [cx2, cy2], [cx1, cy2]]

    elif "t-shirt-2" in base_path:
        # mockup_t-shirt-2.png: White t-shirt model portrait.
        cy1, cy2 = int(h * 0.22), int(h * 0.60)
        cx1, cx2 = int(w * 0.25), int(w * 0.75)

        # Segment white fabric inside chest box (thresholding at 160 to exclude neck, background, hair)
        roi = gray[cy1:cy2, cx1:cx2]
        roi_mask = (roi > 160).astype(np.uint8) * 255

        # Structural closing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel)

        mask[cy1:cy2, cx1:cx2] = roi_mask
        corners = [[cx1, cy1], [cx2, cy1], [cx2, cy2], [cx1, cy2]]

    else:
        # Generic fallback
        cy1, cy2 = int(h * 0.25), int(h * 0.75)
        cx1, cx2 = int(w * 0.25), int(w * 0.75)
        mask[cy1:cy2, cx1:cx2] = 255
        corners = [[cx1, cy1], [cx2, cy1], [cx2, cy2], [cx1, cy2]]

    # Strict check: fail loudly if corners is not exactly 4 points
    if len(corners) != 4:
        raise ValueError("Corner detection did not yield exactly 4 points.")

    # 2. Generate displacement map from base image's high frequency details (grayscale folds)
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    displacement = cv2.normalize(bilateral, None, 50, 205, cv2.NORM_MINMAX)
    if "laptop" in base_path:
        displacement = np.ones_like(gray) * 128

    # 3. Generate lighting/shadow layer
    lighting = cv2.bilateralFilter(gray, 9, 50, 50)
    lighting = cv2.normalize(lighting, None, 60, 255, cv2.NORM_MINMAX)
    if "laptop" in base_path:
        lighting = np.ones_like(gray) * 255

    # 4. Calculate print margins
    if isinstance(physical_size_mm, str):
        parts = re.findall(r"\d+\.?\d*", physical_size_mm)
        if len(parts) == 2:
            physical_size_mm = [float(parts[0]), float(parts[1])]
        else:
            physical_size_mm = [400.0, 500.0]  # default fallback
    elif not physical_size_mm:
        # default fallback physical size
        physical_size_mm = [400.0, 500.0]

    corners_np = np.array(corners)
    min_x, min_y = corners_np.min(axis=0)
    max_x, max_y = corners_np.max(axis=0)

    margin_left_px = min_x
    margin_top_px = min_y
    margin_right_px = w - max_x
    margin_bottom_px = h - max_y

    scale_x = physical_size_mm[0] / w
    scale_y = physical_size_mm[1] / h

    print_margins = {
        "left": round(margin_left_px * scale_x, 2),
        "right": round(margin_right_px * scale_x, 2),
        "top": round(margin_top_px * scale_y, 2),
        "bottom": round(margin_bottom_px * scale_y, 2)
    }

    return {
        "base": img,
        "mask": mask,
        "displacement": displacement,
        "lighting": lighting,
        "corners": corners,
        "print_margins": print_margins,
        "physical_size_mm": physical_size_mm,
        "target_dpi": int(target_dpi)
    }
