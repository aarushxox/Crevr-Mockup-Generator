import os
import cv2
import json
from engine.pipeline.ingest import ingest_raw_mockup

def run_pre_ingest():
    raw_assets = [
        {
            "path": "assets/mockup_laptop.png",
            "id": "laptop_macbook_front_01",
            "category": "tech",
            "subtype": "laptop",
            "label": "MacBook Pro — Front View, Clean Desk",
            "fold_intensity": 0,
            "physical_size_mm": [357.8, 229.0],
            "target_dpi": 300
        },
        {
            "path": "assets/mockup_t_shirt.png",
            "id": "tshirt_white_front_01",
            "category": "apparel",
            "subtype": "t-shirt",
            "label": "Classic White T-Shirt — Flat Lay",
            "fold_intensity": 12,
            "physical_size_mm": [500.0, 700.0],
            "target_dpi": 300
        },
        {
            "path": "assets/mockup_t-shirt-2.png",
            "id": "tshirt_white_front_02",
            "category": "apparel",
            "subtype": "t-shirt",
            "label": "White T-Shirt — Studio Portrait",
            "fold_intensity": 18,
            "physical_size_mm": [500.0, 700.0],
            "target_dpi": 300
        }
    ]

    os.makedirs("templates", exist_ok=True)

    for asset in raw_assets:
        print(f"Ingesting {asset['path']} -> templates/{asset['id']}...")
        result = ingest_raw_mockup(
            base_path=asset["path"],
            category=asset["category"],
            subtype=asset["subtype"],
            label=asset["label"],
            fold_intensity=asset["fold_intensity"],
            physical_size_mm=asset["physical_size_mm"],
            target_dpi=asset["target_dpi"]
        )

        template_dir = f"templates/{asset['id']}"
        os.makedirs(template_dir, exist_ok=True)

        # Save files
        cv2.imwrite(os.path.join(template_dir, "base.png"), result["base"])
        cv2.imwrite(os.path.join(template_dir, "mask.png"), result["mask"])
        cv2.imwrite(os.path.join(template_dir, "displacement.png"), result["displacement"])
        cv2.imwrite(os.path.join(template_dir, "lighting.png"), result["lighting"])

        # Save metadata.json
        metadata = {
            "id": asset["id"],
            "category": asset["category"],
            "subtype": asset["subtype"],
            "label": asset["label"],
            "base_image": "base.png",
            "mask_image": "mask.png",
            "displacement_image": "displacement.png",
            "lighting_image": "lighting.png",
            "design_zone_corners": result["corners"],
            "fold_intensity": asset["fold_intensity"],
            "physical_size_mm": asset["physical_size_mm"],
            "target_dpi": asset["target_dpi"],
            "print_margin_px": result["print_margin_px"],
            "allow_rotation": True,
            "rotation_limits_deg": [-15, 15] if asset["category"] == "apparel" else [0, 0],
            "allow_perspective_adjust": False,
            "recommended_design_resolution_px": [1500, 1500],
            "min_upload_resolution_px": [300, 300],
            "max_upload_resolution_px": [6000, 6000],
            "supported_formats": ["png", "jpg", "webp"],
            "export_default_format": "png",
            "export_max_resolution_px": [4096, 4096] if asset["id"] == "laptop_macbook_front_01" else [2000, 2000],
            "created_at": "2026-07-15",
            "engine_version": "1.0"
        }

        with open(os.path.join(template_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Successfully created template: {asset['id']}")

if __name__ == "__main__":
    run_pre_ingest()
