import unittest
import numpy as np
import cv2
import os
import json
import tempfile
from engine.pipeline.warp import get_warp_matrix, warp_design
from engine.pipeline.displacement import apply_displacement
from engine.pipeline.blend import blend_multiply, blend_screen, match_histogram_lab
from engine.pipeline.mask import composite_images

class TestPipeline(unittest.TestCase):
    def test_warp_matrix(self):
        src_pts = [[0, 0], [10, 0], [10, 10], [0, 10]]
        dst_pts = [[2, 2], [12, 2], [12, 12], [2, 12]]
        matrix = get_warp_matrix(src_pts, dst_pts)
        self.assertEqual(matrix.shape, (3, 3))

    def test_warp_design(self):
        design = np.zeros((100, 100, 3), dtype=np.uint8)
        matrix = np.eye(3, dtype=np.float32)
        warped = warp_design(design, matrix, (100, 100))
        self.assertEqual(warped.shape, (100, 100, 3))

    def test_displacement(self):
        design = np.ones((100, 100, 3), dtype=np.uint8) * 128
        displacement_map = np.ones((100, 100), dtype=np.uint8) * 128
        displaced = apply_displacement(design, displacement_map, intensity=10)
        self.assertEqual(displaced.shape, (100, 100, 3))

    def test_blend_modes(self):
        fg = np.ones((10, 10, 3), dtype=np.uint8) * 100
        bg = np.ones((10, 10, 3), dtype=np.uint8) * 200
        mul = blend_multiply(fg, bg)
        self.assertTrue(np.all(mul <= 100))

        scr = blend_screen(fg, bg)
        self.assertTrue(np.all(scr >= 200))

    def test_composite(self):
        base = np.zeros((100, 100, 3), dtype=np.uint8)
        design = np.ones((100, 100, 3), dtype=np.uint8) * 255
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255

        comp = composite_images(base, design, mask, feather_radius=3)
        self.assertEqual(comp.shape, (100, 100, 3))
        self.assertEqual(comp[50, 50, 0], 255)
        self.assertEqual(comp[5, 5, 0], 0)

    def test_ingest_fails_on_invalid_corners(self):
        # Using a suffix containing "laptop" triggers laptop corner detection path.
        # Since the image is blank, contour detection fails and should raise ValueError.
        with tempfile.NamedTemporaryFile(suffix="_laptop.png", delete=False) as f:
            temp_path = f.name
        try:
            blank = np.zeros((200, 200, 3), dtype=np.uint8)
            cv2.imwrite(temp_path, blank)
            with self.assertRaises(ValueError):
                from engine.pipeline.ingest import ingest_raw_mockup
                ingest_raw_mockup(
                    base_path=temp_path,
                    category="tech",
                    subtype="laptop",
                    label="Test Invalid"
                )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_render_mockup_physical_scaling_clamping(self):
        from engine.pipeline.render import render_mockup
        design = np.zeros((100, 100, 4), dtype=np.uint8)
        design[:, :] = [255, 0, 0, 255]

        template_folder = "templates/tshirt_white_front_01"
        transform_options = {"x": 0.0, "y": 0.0, "scale": 1.0, "rotation": 0.0}
        export_options = {"dpi": 600, "color_correct": False, "feather_radius": 1}

        rendered, warnings = render_mockup(template_folder, design, transform_options, export_options)
        self.assertEqual(rendered.shape[:2], (2000, 1428))
        self.assertTrue(any("clamped" in w.lower() for w in warnings))

if __name__ == "__main__":
    unittest.main()
