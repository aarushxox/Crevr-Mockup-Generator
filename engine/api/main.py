import os
import json
import uuid
import sqlite3
import base64
import re
import io
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import cv2
import numpy as np
from PIL import Image, ImageOps

from engine.pipeline.render import render_mockup
from engine.pipeline.ingest import ingest_raw_mockup, clean_mask

app = FastAPI(title="Crevr Mockup Generator — API Engine", version="1.0.0")

# Setup CORS for development and gateways
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"],
)

DB_PATH = "data/crevr.db"
UPLOAD_DIR = "data/designs"
EXPORT_DIR = "data/exports"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

# ----------------- DB Helpers -----------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.on_event("startup")
def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "CREATE TABLE IF NOT EXISTS history ("
        "id TEXT PRIMARY KEY, "
        "template_id TEXT, "
        "design_id TEXT, "
        "created_at TEXT, "
        "output_path TEXT"
        ")"
    )
    conn.commit()
    conn.close()

# Run initialization immediately to guarantee table exists for unit tests/scripts
init_db()

# ----------------- Schemas -----------------
class RenderTransform(BaseModel):
    x: float = 0.0
    y: float = 0.0
    scale: float = 1.0
    rotation: float = 0.0

class RenderExportOptions(BaseModel):
    format: str = "png"
    resolution: int = 300
    dpi: int = 300
    color_correct: bool = True
    feather_radius: int = 3

class RenderRequest(BaseModel):
    template_id: str
    design_id: str
    transform: RenderTransform = RenderTransform()
    export: RenderExportOptions = RenderExportOptions()

# ----------------- API Endpoints -----------------

@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "Crevr Compositing Engine"}

@app.post("/api/templates")
def list_templates(category: Optional[str] = Query(None)):
    """
    List ingested and ready-to-use mockup templates.
    """
    templates_dir = "templates"
    if not os.path.exists(templates_dir):
        return {"templates": []}

    template_list = []
    for tid in os.listdir(templates_dir):
        meta_path = os.path.join(templates_dir, tid, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            # Treat empty category string as None (no filter)
            if not category or category == "" or meta.get("category") == category:
                template_list.append(meta)
    return {"templates": template_list}

@app.get("/api/templates/{template_id}")
def get_template(template_id: str):
    """
    Get full metadata for one template.
    """
    meta_path = os.path.join("templates", template_id, "metadata.json")
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="Template not found")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return meta

@app.get("/api/templates/{template_id}/asset/{file_name}")
def get_template_asset(template_id: str, file_name: str):
    """
    Serve raw template asset files (base.png, mask.png, lighting.png, displacement.png).
    """
    file_path = os.path.join("templates", template_id, file_name)
    # Security: path traversal prevention
    resolved_path = os.path.abspath(file_path)
    if not resolved_path.startswith(os.path.abspath("templates")):
        raise HTTPException(status_code=403, detail="Path traversal forbidden")

    if not os.path.exists(resolved_path):
        raise HTTPException(status_code=404, detail="Asset file not found")
    return FileResponse(resolved_path)

@app.post("/api/designs/upload")
async def upload_design(file: UploadFile = File(...)):
    """
    Upload a user design (png, jpg, webp), validate size/type, strip metadata,
    and return unique design_id plus metadata.
    """
    content = await file.read()
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Decompression/file size exceeds 25MB limit")

    try:
        # Load with PIL for signature and integrity verification
        pil_img = Image.open(io.BytesIO(content))
        pil_img.verify()  # Verifies the file is not corrupt

        # Now reopen since verify() closes the file/stream
        pil_img = Image.open(io.BytesIO(content))

        # Enforce 64 Megapixel limit (8000x8000 = 64,000,000 pixels) to prevent decompression bombs
        w, h = pil_img.size
        if w * h > 64 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image resolution exceeds 64 Megapixels limit to protect against decompression bombs.")

        # Strip EXIF metadata and apply orientation
        pil_img = ImageOps.exif_transpose(pil_img)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unsupported or corrupt image signature. Please upload PNG, JPG, or WebP. Details: {str(e)}")

    # Convert PIL Image to numpy array (OpenCV uses BGR/BGRA)
    if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
        pil_img = pil_img.convert("RGBA")
        img = np.array(pil_img)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    else:
        pil_img = pil_img.convert("RGB")
        img = np.array(pil_img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    design_id = str(uuid.uuid4())
    ext = "png" if img.shape[2] == 4 else "jpg"
    design_filename = f"{design_id}.{ext}"
    design_path = os.path.join(UPLOAD_DIR, design_filename)

    if ext == "png":
        cv2.imwrite(design_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    else:
        cv2.imwrite(design_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return {
        "design_id": design_id,
        "width": w,
        "height": h,
        "has_alpha": img.shape[2] == 4,
        "preview_url": f"/api/designs/{design_id}/file"
    }

@app.get("/api/designs/{design_id}/file")
def get_design_file(design_id: str):
    """
    Retrieve uploaded design file.
    """
    for ext in ["png", "jpg", "jpeg", "webp"]:
        path = os.path.join(UPLOAD_DIR, f"{design_id}.{ext}")
        if os.path.exists(path):
            return FileResponse(path)
    raise HTTPException(status_code=404, detail="Design file not found")

@app.post("/api/designs/{design_id}/remove-bg")
def remove_background(design_id: str):
    """
    Classical CV green/chroma key background removal for design uploads.
    """
    file_path = None
    for ext in ["png", "jpg", "jpeg", "webp"]:
        path = os.path.join(UPLOAD_DIR, f"{design_id}.{ext}")
        if os.path.exists(path):
            file_path = path
            break

    if not file_path:
        raise HTTPException(status_code=404, detail="Design not found")

    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot decode design")

    if img.shape[2] == 4:
        bgr = img[:, :, :3]
    else:
        bgr = img

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    white_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)[1]

    bg_mask = cv2.bitwise_or(green_mask, white_mask)
    fg_mask = cv2.bitwise_not(bg_mask)

    rgba = np.zeros((bgr.shape[0], bgr.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = bgr
    rgba[:, :, 3] = fg_mask

    new_path = os.path.join(UPLOAD_DIR, f"{design_id}.png")
    cv2.imwrite(new_path, rgba, [cv2.IMWRITE_PNG_COMPRESSION, 6])

    if not file_path.endswith(".png"):
        os.remove(file_path)

    return {
        "status": "success",
        "design_id": design_id,
        "message": "Background removed successfully",
        "preview_url": f"/api/designs/{design_id}/file"
    }

@app.post("/api/render")
def render_template_mockup(req: RenderRequest):
    """
    Triggers the high-fidelity rendering pipeline using the precomputed templates.
    """
    template_folder = os.path.join("templates", req.template_id)
    if not os.path.exists(template_folder):
        raise HTTPException(status_code=404, detail="Mockup template not found")

    # Load metadata to get category and limits
    meta_path = os.path.join(template_folder, "metadata.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    # 1. Enforce scale validation
    if req.transform.scale <= 0:
        raise HTTPException(status_code=400, detail="Scale must be a positive non-zero value.")

    # 2. Enforce rotation permissions and limits
    rotation = req.transform.rotation
    rotation_limits = meta.get("rotation_limits_deg")
    if rotation_limits and len(rotation_limits) == 2 and (rotation_limits[0] != 0 or rotation_limits[1] != 0):
        clamped_rotation = max(rotation_limits[0], min(rotation_limits[1], rotation))
        rotation = clamped_rotation

    design_img = None
    for ext in ["png", "jpg", "jpeg", "webp"]:
        path = os.path.join(UPLOAD_DIR, f"{req.design_id}.{ext}")
        if os.path.exists(path):
            design_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            break

    if design_img is None:
        raise HTTPException(status_code=404, detail="Design file not found")

    warnings_list = []

    # 3. Detect missing alpha channel for apparel designs
    category = meta.get("category", "")
    has_alpha = design_img.shape[2] == 4
    if category == "apparel":
        is_fully_opaque = True
        if has_alpha:
            alpha_channel = design_img[:, :, 3]
            if not np.all(alpha_channel == 255):
                is_fully_opaque = False
        if is_fully_opaque:
            warnings_list.append("Transparency was expected for apparel mockups. The design upload is fully opaque; consider background removal to avoid a solid background block.")

    # 4. Detect low resolution designs
    hd, wd = design_img.shape[:2]
    min_res = meta.get("min_upload_resolution_px", [300, 300])
    rec_res = meta.get("recommended_design_resolution_px", [1500, 1500])

    if wd < min_res[0] or hd < min_res[1]:
        warnings_list.append(f"Design resolution ({wd}x{hd}px) is extremely low. Minimum required is {min_res[0]}x{min_res[1]}px. Visual quality may be degraded.")
    elif wd < rec_res[0] or hd < rec_res[1]:
        warnings_list.append(f"Design resolution ({wd}x{hd}px) is lower than recommended ({rec_res[0]}x{rec_res[1]}px). Visual quality may be reduced.")

    transform_options = {
        "x": req.transform.x,
        "y": req.transform.y,
        "scale": req.transform.scale,
        "rotation": rotation
    }

    export_options = {
        "format": req.export.format,
        "dpi": req.export.dpi,
        "color_correct": req.export.color_correct,
        "feather_radius": req.export.feather_radius
    }

    try:
        rendered, render_warnings = render_mockup(template_folder, design_img, transform_options, export_options)
        warnings_list.extend(render_warnings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline rendering failed: {str(e)}")

    job_id = str(uuid.uuid4())
    output_ext = req.export.format.lower()
    if output_ext not in ["png", "jpg", "webp"]:
        output_ext = "png"

    output_filename = f"{job_id}.{output_ext}"
    output_path = os.path.join(EXPORT_DIR, output_filename)

    if output_ext == "png":
        cv2.imwrite(output_path, rendered, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    elif output_ext == "webp":
        cv2.imwrite(output_path, rendered, [cv2.IMWRITE_WEBP_QUALITY, 90])
    else:
        cv2.imwrite(output_path, rendered, [cv2.IMWRITE_JPEG_QUALITY, 92])

    conn = get_db()
    c = conn.cursor()
    c.execute(
        "INSERT INTO history (id, template_id, design_id, created_at, output_path) VALUES (?, ?, ?, ?, ?)",
        (job_id, req.template_id, req.design_id, datetime.utcnow().isoformat(), output_path)
    )
    conn.commit()
    conn.close()

    return {
        "job_id": job_id,
        "status": "completed",
        "output_url": f"/api/render/{job_id}/download",
        "created_at": datetime.utcnow().isoformat(),
        "warnings": warnings_list
    }

@app.get("/api/render/{job_id}/download")
def download_rendered_mockup(job_id: str):
    """
    Download/retrieve the rendered high-res mockup image.
    """
    conn = get_db()
    c = conn.cursor()
    row = c.execute("SELECT output_path FROM history WHERE id=?", (job_id,)).fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Render job not found")

    out_path = row["output_path"]
    if not os.path.exists(out_path):
        raise HTTPException(status_code=404, detail="File has been cleaned or removed")

    return FileResponse(out_path)

@app.get("/api/history")
def get_render_history():
    """
    Get full list of previously rendered jobs from local SQLite database.
    """
    conn = get_db()
    c = conn.cursor()
    rows = c.execute("SELECT id, template_id, design_id, created_at FROM history ORDER BY created_at DESC").fetchall()
    conn.close()

    history = []
    for r in rows:
        history.append({
            "id": r["id"],
            "template_id": r["template_id"],
            "design_id": r["design_id"],
            "created_at": r["created_at"],
            "output_url": f"/api/render/{r['id']}/download"
        })
    return {"history": history}

@app.delete("/api/history/{job_id}")
def delete_history_item(job_id: str):
    """
    Delete render job from history list and clean up its file on disk.
    """
    conn = get_db()
    c = conn.cursor()
    row = c.execute("SELECT output_path FROM history WHERE id=?", (job_id,)).fetchone()
    if row:
        out_path = row["output_path"]
        if os.path.exists(out_path):
            os.remove(out_path)
        c.execute("DELETE FROM history WHERE id=?", (job_id,))
        conn.commit()
    conn.close()
    return {"status": "success", "message": f"Render job {job_id} deleted"}

@app.post("/api/templates/ingest")
async def ingest_template(
    file: UploadFile = File(...),
    id: str = Form(...),
    category: str = Form(...),
    subtype: str = Form(...),
    label: str = Form(...),
    fold_intensity: int = Form(15),
    physical_size_mm: Optional[str] = Form(None),
    target_dpi: int = Form(300)
):
    """
    (Admin API) Upload and ingest a brand new blank product mockup photo.
    Runs the automated CV segmentation and analytical pipeline.
    """
    content = await file.read()
    temp_path = f"data/temp_{uuid.uuid4()}.png"
    with open(temp_path, "wb") as f:
        f.write(content)

    parsed_physical_size = None
    if physical_size_mm:
        try:
            parsed_physical_size = json.loads(physical_size_mm)
        except Exception:
            parts = re.findall(r"\d+\.?\d*", physical_size_mm)
            if len(parts) == 2:
                parsed_physical_size = [float(parts[0]), float(parts[1])]

    try:
        result = ingest_raw_mockup(
            base_path=temp_path,
            category=category,
            subtype=subtype,
            label=label,
            fold_intensity=fold_intensity,
            physical_size_mm=parsed_physical_size,
            target_dpi=target_dpi
        )

        template_dir = os.path.join("templates", id)
        os.makedirs(template_dir, exist_ok=True)

        cv2.imwrite(os.path.join(template_dir, "base.png"), result["base"])
        cv2.imwrite(os.path.join(template_dir, "mask.png"), result["mask"])
        cv2.imwrite(os.path.join(template_dir, "displacement.png"), result["displacement"])
        cv2.imwrite(os.path.join(template_dir, "lighting.png"), result["lighting"])

        metadata = {
            "id": id,
            "category": category,
            "subtype": subtype,
            "label": label,
            "base_image": "base.png",
            "mask_image": "mask.png",
            "displacement_image": "displacement.png",
            "lighting_image": "lighting.png",
            "design_zone_corners": result["corners"],
            "fold_intensity": fold_intensity,
            "allow_rotation": True,
            "rotation_limits_deg": [-15, 15] if category == "apparel" else [0, 0],
            "allow_perspective_adjust": False,
            "recommended_design_resolution_px": [1500, 1500],
            "min_upload_resolution_px": [300, 300],
            "max_upload_resolution_px": [6000, 6000],
            "supported_formats": ["png", "jpg", "webp"],
            "export_default_format": "png",
            "export_max_resolution_px": [2000, 2000],
            "created_at": "2026-07-15",
            "engine_version": "1.0",
            "physical_size_mm": result["physical_size_mm"],
            "target_dpi": result["target_dpi"],
            "print_margins": result["print_margins"]
        }

        with open(os.path.join(template_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {"status": "success", "template_id": id, "metadata": metadata}
