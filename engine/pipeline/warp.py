import cv2
import numpy as np

def get_warp_matrix(src_pts, dst_pts):
    """
    Compute the perspective transform matrix from src_pts to dst_pts.
    """
    return cv2.getPerspectiveTransform(
        np.array(src_pts, dtype=np.float32),
        np.array(dst_pts, dtype=np.float32)
    )

def warp_design(design, matrix, output_size):
    """
    Warp the design image using the perspective matrix to the specified output size.
    Uses cubic interpolation for high quality.
    """
    return cv2.warpPerspective(
        design,
        matrix,
        output_size,
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )
