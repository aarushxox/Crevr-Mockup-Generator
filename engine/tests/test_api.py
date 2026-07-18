import unittest
import os
import json
import io
import cv2
import numpy as np
from fastapi.testclient import TestClient
from engine.api.main import app

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "service": "Crevr Compositing Engine"})

    def test_templates_list(self):
        response = self.client.post("/api/templates")
        self.assertEqual(response.status_code, 200)
        self.assertIn("templates", response.json())
        templates = response.json()["templates"]
        self.assertTrue(len(templates) > 0)

    def test_upload_design_validation(self):
        file_data = b"Hello, this is a plain text file, not an image!"
        response = self.client.post(
            "/api/designs/upload",
            files={"file": ("test.txt", file_data, "text/plain")}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported or corrupt image signature", response.json()["detail"])

    def test_upload_design_and_render(self):
        # Create an image with a green background and a red circle, so remove-bg can actually make it transparent
        img = np.zeros((100, 100, 4), dtype=np.uint8)
        img[:, :] = [0, 255, 0, 255]
        cv2.circle(img, (50, 50), 40, (0, 0, 255, 255), -1)
        _, buf = cv2.imencode(".png", img)
        file_data = buf.tobytes()

        response = self.client.post(
            "/api/designs/upload",
            files={"file": ("design.png", file_data, "image/png")}
        )
        self.assertEqual(response.status_code, 200)
        design_id = response.json()["design_id"]
        self.assertIsNotNone(design_id)

        bg_res = self.client.post(f"/api/designs/{design_id}/remove-bg")
        self.assertEqual(bg_res.status_code, 200)

        render_payload = {
            "template_id": "tshirt_white_front_01",
            "design_id": design_id,
            "transform": {
                "x": 0.0,
                "y": 0.0,
                "scale": 1.0,
                "rotation": 0.0
            },
            "export": {
                "format": "png",
                "resolution": 300,
                "dpi": 300,
                "color_correct": True,
                "feather_radius": 3
            }
        }
        render_res = self.client.post("/api/render", json=render_payload)
        self.assertEqual(render_res.status_code, 200)
        job_id = render_res.json()["job_id"]
        self.assertIsNotNone(job_id)

        dl_res = self.client.get(f"/api/render/{job_id}/download")
        self.assertEqual(dl_res.status_code, 200)
        self.assertTrue(len(dl_res.content) > 0)

    def test_missing_alpha_for_apparel_trigger(self):
        # Create an entirely opaque design (no transparency, has_alpha is True but opaque or has_alpha is False)
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        _, buf = cv2.imencode(".jpg", img)
        file_data = buf.tobytes()

        response = self.client.post(
            "/api/designs/upload",
            files={"file": ("opaque.jpg", file_data, "image/jpeg")}
        )
        self.assertEqual(response.status_code, 200)
        design_id = response.json()["design_id"]

        render_payload = {
            "template_id": "tshirt_white_front_01",
            "design_id": design_id,
            "transform": {
                "x": 0.0,
                "y": 0.0,
                "scale": 1.0,
                "rotation": 0.0
            },
            "export": {
                "format": "png",
                "resolution": 300,
                "dpi": 300,
                "color_correct": True,
                "feather_radius": 3
            }
        }
        render_res = self.client.post("/api/render", json=render_payload)
        self.assertEqual(render_res.status_code, 400)
        self.assertEqual(render_res.json()["detail"]["code"], "E1005")

    def test_invalid_transform_rotation_limit(self):
        # Create a valid transparent design
        img = np.zeros((100, 100, 4), dtype=np.uint8)
        cv2.circle(img, (50, 50), 40, (0, 0, 255, 255), -1)
        _, buf = cv2.imencode(".png", img)
        file_data = buf.tobytes()

        upload_res = self.client.post(
            "/api/designs/upload",
            files={"file": ("design.png", file_data, "image/png")}
        )
        design_id = upload_res.json()["design_id"]

        # Try to render on laptop with a non-zero rotation (rotation_limits_deg is [0, 0] for tech)
        render_payload = {
            "template_id": "laptop_macbook_front_01",
            "design_id": design_id,
            "transform": {
                "x": 0.0,
                "y": 0.0,
                "scale": 1.0,
                "rotation": 10.0
            },
            "export": {
                "format": "png",
                "resolution": 300,
                "dpi": 300,
                "color_correct": True,
                "feather_radius": 3
            }
        }
        render_res = self.client.post("/api/render", json=render_payload)
        self.assertEqual(render_res.status_code, 400)
        self.assertEqual(render_res.json()["detail"]["code"], "E3003")

    def test_negative_scale_limit(self):
        # Create a valid transparent design
        img = np.zeros((100, 100, 4), dtype=np.uint8)
        cv2.circle(img, (50, 50), 40, (0, 0, 255, 255), -1)
        _, buf = cv2.imencode(".png", img)
        file_data = buf.tobytes()

        upload_res = self.client.post(
            "/api/designs/upload",
            files={"file": ("design.png", file_data, "image/png")}
        )
        design_id = upload_res.json()["design_id"]

        render_payload = {
            "template_id": "tshirt_white_front_01",
            "design_id": design_id,
            "transform": {
                "x": 0.0,
                "y": 0.0,
                "scale": -0.5,
                "rotation": 0.0
            },
            "export": {
                "format": "png",
                "resolution": 300,
                "dpi": 300,
                "color_correct": True,
                "feather_radius": 3
            }
        }
        render_res = self.client.post("/api/render", json=render_payload)
        self.assertEqual(render_res.status_code, 400)
        self.assertEqual(render_res.json()["detail"]["code"], "E3003")

    def test_decompression_bomb_prevention(self):
        # A virtual/empty upload but stating larger sizes or corrupt size
        # We can construct content with huge size if we could, but a small file with a size that
        # indicates huge pixels will fail the verify or the size check on pil size header.
        # Construct a corrupt or huge PNG structure
        file_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x27\x10\x00\x00\x27\x10\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
        response = self.client.post(
            "/api/designs/upload",
            files={"file": ("bomb.png", file_data, "image/png")}
        )
        self.assertEqual(response.status_code, 400)

if __name__ == "__main__":
    unittest.main()
