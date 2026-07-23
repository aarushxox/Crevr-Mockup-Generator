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

    def test_upload_design_rejection_over_64mp(self):
        # Create an 8001x8001 monochrome image (tiny in byte size when compressed)
        img = np.zeros((8001, 8001), dtype=np.uint8)
        _, buf = cv2.imencode(".png", img)
        file_data = buf.tobytes()

        response = self.client.post(
            "/api/designs/upload",
            files={"file": ("large_design.png", file_data, "image/png")}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("dimensions exceed", response.json()["detail"])

    def test_upload_design_and_render(self):
        img = np.zeros((100, 100, 4), dtype=np.uint8)
        img[:, :] = [0, 0, 255, 255]
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

    def test_render_mockup_clamping_warnings(self):
        # Upload a valid small design
        img = np.zeros((100, 100, 4), dtype=np.uint8)
        img[:, :] = [0, 255, 0, 255]
        _, buf = cv2.imencode(".png", img)
        file_data = buf.tobytes()

        response = self.client.post(
            "/api/designs/upload",
            files={"file": ("design.png", file_data, "image/png")}
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
                "dpi": 600,  # High DPI to trigger resolution clamping
                "color_correct": True,
                "feather_radius": 3
            }
        }
        render_res = self.client.post("/api/render", json=render_payload)
        self.assertEqual(render_res.status_code, 200)

        # Check warnings in the JSON response
        warnings = render_res.json().get("warnings")
        self.assertIsNotNone(warnings)
        self.assertTrue(any("clamped" in w.lower() for w in warnings))

if __name__ == "__main__":
    unittest.main()
