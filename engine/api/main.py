import os
import json
import uuid
import sqlite3
import base64
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import cv2
import numpy as np

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

    header = content[:12]
    is_png = header.startswith(b"\x89PNG\r\n\x1a\n")
    is_jpg = header.startswith(b"\xff\xd8\xff")
    is_webp = b"WEBP" in header

    if not (is_png or is_jpg or is_webp):
        raise HTTPException(status_code=400, detail="Unsupported or corrupt image signature. Please upload PNG, JPG, or WebP.")

    # Validate image using PIL without full decompression first to protect against zip bombs
    from PIL import Image
    import io
    try:
        img_pil = Image.open(io.BytesIO(content))
        img_pil.verify()

        # Verify dimensions on the verified image header
        w, h = img_pil.size
        if w > 8000 or h > 8000 or w * h > 64000000:
            raise HTTPException(status_code=400, detail="Image dimensions exceed 8000x8000 pixels limit")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=400, detail="Image corrupt or undecodable.")

    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(status_code=400, detail="Image corrupt or undecodable.")

    h, w = img.shape[:2]
    if w > 8000 or h > 8000:
        raise HTTPException(status_code=400, detail="Image dimensions exceed 8000x8000 pixels limit")

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

    meta_path = os.path.join(template_folder, "metadata.json")
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="Template metadata not found")
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    design_img = None
    for ext in ["png", "jpg", "jpeg", "webp"]:
        path = os.path.join(UPLOAD_DIR, f"{req.design_id}.{ext}")
        if os.path.exists(path):
            design_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            break

    if design_img is None:
        raise HTTPException(status_code=404, detail="Design file not found")

    # Missing alpha channel detection for apparel templates
    if metadata.get("category") == "apparel":
        has_alpha = design_img.shape[2] == 4 if len(design_img.shape) == 3 else False
        is_fully_opaque = True
        if has_alpha:
            alpha_channel = design_img[:, :, 3]
            if not np.all(alpha_channel == 255):
                is_fully_opaque = False
        if not has_alpha or is_fully_opaque:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "E1005",
                    "message": "Missing alpha channel on a design meant to be transparent. Please run background removal or upload a transparent image."
                }
            )

    # Enforce transform limits against metadata parameters
    allow_rotation = metadata.get("allow_rotation", True)
    rotation_limits = metadata.get("rotation_limits_deg")

    # If allow_rotation is false, then rotation must be 0
    if not allow_rotation and req.transform.rotation != 0.0:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "E3003",
                "message": f"Rotation is not allowed for template: {req.template_id}"
            }
        )

    # Check rotation limits if they are defined
    if allow_rotation and rotation_limits and len(rotation_limits) == 2:
        min_rot, max_rot = rotation_limits
        # Allow a slight epsilon tolerance for floating-point inaccuracies
        if not (min_rot - 0.01 <= req.transform.rotation <= max_rot + 0.01):
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "E3003",
                    "message": f"Rotation {req.transform.rotation} is out of bounds [{min_rot}, {max_rot}] for template: {req.template_id}"
                }
            )

    # Enforce strictly positive scale to prevent homography projection failure
    if req.transform.scale <= 0.0:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "E3003",
                "message": f"Scale must be strictly positive (greater than 0.0)"
            }
        )

    transform_options = {
        "x": req.transform.x,
        "y": req.transform.y,
        "scale": req.transform.scale,
        "rotation": req.transform.rotation
    }

    export_options = {
        "format": req.export.format,
        "dpi": req.export.dpi,
        "color_correct": req.export.color_correct,
        "feather_radius": req.export.feather_radius
    }

    # Track warnings for the client
    warnings = []

    # Enforce design image resolution limits with warnings instead of failing
    min_res = metadata.get("min_upload_resolution_px", [300, 300])
    rec_res = metadata.get("recommended_design_resolution_px", [1500, 1500])
    dh, dw = design_img.shape[:2]

    if dw < min_res[0] or dh < min_res[1]:
        warnings.append(
            f"The uploaded design's resolution ({dw}x{dh}px) is below the minimum recommended resolution ({min_res[0]}x{min_res[1]}px) for this template, which may result in visual pixelation."
        )
    elif dw < rec_res[0] or dh < rec_res[1]:
        warnings.append(
            f"For optimal visual quality, a higher resolution design of at least {rec_res[0]}x{rec_res[1]}px is recommended. Current: {dw}x{dh}px."
        )

    try:
        rendered = render_mockup(template_folder, design_img, transform_options, export_options)
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
        "warnings": warnings
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
    fold_intensity: int = Form(15)
):
    """
    (Admin API) Upload and ingest a brand new blank product mockup photo.
    Runs the automated CV segmentation and analytical pipeline.
    """
    content = await file.read()
    temp_path = f"data/temp_{uuid.uuid4()}.png"
    with open(temp_path, "wb") as f:
        f.write(content)

    try:
        result = ingest_raw_mockup(
            base_path=temp_path,
            category=category,
            subtype=subtype,
            label=label,
            fold_intensity=fold_intensity
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
            "engine_version": "1.0"
        }

        with open(os.path.join(template_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {"status": "success", "template_id": id, "metadata": metadata}
