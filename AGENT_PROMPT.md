# Crevr Mockup Generator — Product & Engineering PRD
### Version 1.0 — Single Source of Truth for AI Coding Agents (Jules, Claude Code, Antigravity, Codex, OpenCode, Cursor, Windsurf)

Repository: https://github.com/aarushxox/Crevr-Mockup-Generator

---

## 0. Read This First (For the AI Agent)

You are building **Crevr**, a local-first, deterministic **image compositing engine** for product mockups. This is explicitly **NOT**:
- An AI image generator (no Stable Diffusion / Midjourney / DALL-E pipeline as the core mechanism)
- A "prompt in, mockup out" tool

It **IS**:
- A pixel-math pipeline: take a blank product photo (t-shirt, laptop, etc.), find the flat surface where a design goes, warp the user's design to match that surface's perspective and fabric/screen texture, blend it in with correct lighting/shadow, and export a photorealistic image.

AI/ML models are used **only** for the perception step (finding the surface, its corners, its depth/wrinkle map) — never for generating the final pixels of the product itself. The final render is always produced by deterministic OpenCV/NumPy math, so the same input always produces the same output (reproducibility is a hard requirement).

Current assets (already in repo):
```
assets/
  mockup_laptop.png
  mockup_t-shirt.png
  mockup_t-shirt-2.png
```
These are blank product photos with no pre-existing metadata (no mask, no corners, no displacement map). Step one of the system is an **ingestion pipeline** that takes a raw photo like this and auto-generates all the metadata a template needs.

---

## 1. Why This Works — The Core Theory (Teach This to the Agent)

### 1.1 What a mockup actually is
A commercial mockup (Placeit, Smartmockups, Renderforest) is **not** a new photo per design. It's one photo, reused thousands of times, with a different flat image "glued" onto it each time, using math that bends the flat image to match the photo's surface.

This is exactly what Photoshop **Smart Objects** do:
1. A base PSD has a photo of, say, a t-shirt.
2. Inside that PSD is a Smart Object layer — literally just a placeholder rectangle.
3. When you double-click it, you get a flat canvas. You paste your logo there and save.
4. Photoshop takes your flat logo and re-projects it back onto the t-shirt photo using a **pre-baked displacement map** + a **blend mode** (usually Multiply or Linear Burn) so the fabric's shadows/highlights show through your logo.

Crevr replicates this exact mechanism in code, without Photoshop, without paying for PSD files, using OpenCV/NumPy — and automates the "double click and create the Smart Object" step using CV models instead of a human designer manually painting a displacement map.

### 1.2 The four pillars of realism
Every realistic mockup composite needs these four things, in this order:

1. **Geometric alignment (Perspective/Homography)** — the design must be warped into the exact quadrilateral (or curved surface) where it will sit, matching the camera's viewing angle.
2. **Surface displacement (Wrinkles/Folds/Bezels)** — the design must locally bend along the surface's own topology (a t-shirt has folds; a laptop screen does not — it's flat, so this step's intensity is template-dependent).
3. **Photometric blending (Lighting/Shadow/Highlight transfer)** — the base photo's own lighting information must be re-applied on top of the pasted design, or it will look like a flat sticker.
4. **Edge integration (Masking/Feathering/Anti-aliasing)** — the design's boundary must blend seamlessly with the product edge, with no visible seams, jagged pixels, or color fringing.

Every architectural decision below exists to serve one of these four pillars.

---

## 2. Foundations: Images, NumPy, and the Math Underneath

### 2.1 An image is just numbers
A color image is a 3D NumPy array: `shape = (height, width, channels)`. A 1920×1080 RGB image is `numpy.ndarray` of shape `(1080, 1920, 3)`, dtype `uint8` (values 0–255 per channel). Adding a 4th channel (alpha, transparency) gives shape `(1080, 1920, 4)` — this is how PNGs store "cut-out" designs with transparent backgrounds.

Why this matters: every operation in this system (warp, blend, mask) is just **matrix math** on these arrays. There is no hidden magic — `image * mask` zeroes out unwanted pixels, `image_a * alpha + image_b * (1-alpha)` blends two images, and so on.

### 2.2 Key NumPy concepts you will use constantly
- **Broadcasting**: applying a `(H,W,1)` mask array against a `(H,W,3)` color array without manually looping pixels — NumPy expands the mask across the 3 channels automatically. This is why NumPy is 100–1000x faster than Python for-loops on images.
- **Vectorization**: never loop over pixels in Python. Always express operations as whole-array math (`np.where`, `np.clip`, `cv2.*` functions), or performance collapses (a 4K image has 8+ million pixels).
- **dtype discipline**: images are `uint8` (0-255) but math like blending needs `float32` intermediate precision, or you get banding/clipping artifacts. Always cast up before math, clip, then cast back down: `img.astype(np.float32)` → do math → `np.clip(...,0,255).astype(np.uint8)`.
- **Coordinate systems**: NumPy indexes `[row, col]` = `[y, x]` — the opposite of how you'd naturally think "x,y". OpenCV functions expect `(x,y)` in their point arguments but arrays are still `[y,x]` internally. This mismatch is the #1 source of bugs in this kind of codebase — the agent must be careful and consistent.
- **Interpolation/sampling**: when warping an image (stretching a flat design into a trapezoid), new pixel values must be *sampled* from the source at non-integer coordinates. `cv2.warpPerspective` handles this via interpolation flags (`INTER_LINEAR`, `INTER_CUBIC`) — cubic gives smoother results for upscaling, linear is faster for downscaling.

### 2.3 Color spaces (why RGB alone isn't enough)
- **RGB**: good for display, bad for "how bright is this pixel independent of its hue" — needed for shadow/highlight extraction.
- **HSV** (Hue, Saturation, Value): separates color from brightness. Used to detect the "green screen" or flat design-area color in some template creation workflows, and to adjust design brightness/saturation to match ambient product lighting.
- **LAB** (Lightness, A, B): perceptually uniform color space, the industry-standard space for **color matching / histogram matching**, because Euclidean distance in LAB roughly matches human-perceived color difference. Crevr uses LAB, not RGB, for any "match design color temperature to product photo" logic.
- **Grayscale**: single channel, used for masks, displacement maps, and edge detection — throwing away hue/saturation entirely because those steps only care about intensity/geometry, not color.

### 2.4 Gamma, DPI, CMYK, and export correctness
- **Gamma correction**: monitors display light non-linearly; blending two images in "gamma-encoded" (regular 0-255 sRGB) space instead of "linear light" space causes mid-tones to look muddy/dark at hard edges. For most mockup purposes, standard alpha blending in sRGB space is visually acceptable, but Crevr's blend engine exposes a `linear_blend: bool` flag for the (rare) cases where users need photographic-grade accuracy (packaging renders, print previews).
- **DPI**: irrelevant for on-screen preview, critical when the export target is print (business cards, packaging, posters). Crevr must let a template declare a `target_dpi` and `physical_size_mm`, and compute the required pixel export resolution: `pixels = mm / 25.4 * dpi`.
- **CMYK vs RGB**: browsers and most cameras work in RGB; commercial printers expect CMYK. Crevr will NOT attempt in-house CMYK conversion in v1 (color-managed CMYK conversion needs proper ICC profiles and is a deep rabbit hole) — v1 exports high-res RGB PNG/TIFF and documents that professional print users should run their own ICC-based conversion in their print software. This is called out explicitly as an out-of-scope decision with rationale, not an oversight.
- **File formats**: PNG (lossless, supports alpha — use for all internal working files and any output needing transparency), JPEG (lossy, no alpha, smaller — fine for final web-preview exports), WebP (modern, smaller than PNG/JPEG at similar quality, alpha-capable — good default for web delivery), TIFF (lossless, supports CMYK and layers — used only for the print-export path), AVIF (best compression, but slower to encode and not yet universally supported — optional future export target).

---

## 3. Computer Vision Concepts (Beginner → Advanced)

This section is the "textbook" the agent should internalize before writing any perception code.

### 3.1 Masks and thresholding
A **mask** is a grayscale (or binary) image where white = "this region matters" and black = "ignore this region." Masks are how Crevr isolates "the flat design-placement area of the t-shirt" from "everything else in the photo" (background, arms, collar). Simple **thresholding** (`cv2.threshold`) turns a grayscale image into pure black/white based on an intensity cutoff — useful for quick tests but too fragile for real photos with varied lighting. **Adaptive thresholding** adjusts the cutoff per local region, more robust to uneven lighting.

### 3.2 Morphology (cleaning up masks)
Raw masks from thresholding or ML models are noisy — small holes, jagged edges, stray blobs. **Morphological operations** fix this:
- **Erosion**: shrinks white regions (removes small noise specks).
- **Dilation**: grows white regions (fills small holes).
- **Opening** (erode→dilate): removes small noise while preserving overall shape.
- **Closing** (dilate→erode): fills small holes while preserving overall shape.
Crevr's mask post-processing pipeline always runs Closing then Opening with a small kernel (e.g. 5×5) before using any auto-generated mask.

### 3.3 Contours and polygon detection
A **contour** is the boundary curve of a connected white region in a binary mask. `cv2.findContours` extracts these. Once Crevr has the design-area mask, it finds the largest contour, then approximates it to a polygon (`cv2.approxPolyDP`) to get the **4 corner points** of the design zone (or more points if the surface is curved, like a mug). This corner data is exactly what the perspective-warp step needs.

### 3.4 Edge detection (Canny) and line detection (Hough)
**Canny edge detection** finds sharp intensity transitions (product boundaries, screen bezels, seams). **Hough Transform** finds straight lines from those edges — useful for detecting laptop screen bezels or picture-frame borders, which are naturally rectangular and lend themselves to line-based corner-finding rather than blob-based contour finding.

### 3.5 Feature matching (SIFT/ORB) and homography
When a user uploads their *own* blank mockup photo (not from Crevr's curated set), the system may need to detect a *known object class* (e.g. "this is a laptop") to apply reasonable defaults. **ORB** (fast, free, patent-unencumbered — unlike SIFT which had patent issues historically, though SIFT is patent-free again as of 2020) extracts distinctive keypoints and descriptors that can be matched between images. Feature matching + `cv2.findHomography` (using RANSAC to reject outlier matches) computes the **homography matrix** — the 3×3 transformation matrix that maps one flat plane to another perspective view. This is the mathematical heart of every perspective warp in Crevr.

### 3.6 Depth, normal maps, and displacement maps
- **Depth map**: grayscale image where pixel intensity = distance from camera. Needed to understand "this part of the t-shirt is closer to camera (chest) vs farther (side)."
- **Normal map**: encodes surface *direction* (which way is this bit of fabric facing) using RGB channels to represent 3D vector direction (common in 3D graphics, less critical for 2D mockup compositing but useful for advanced relighting).
- **Displacement map**: grayscale image where pixel intensity = "how much to locally shift this pixel." This is what Photoshop Smart Objects bake per-template. Lighter areas push the design "up/out," darker areas push it "in," simulating folds and wrinkles. Crevr generates this automatically for fabric templates using depth-estimation models (see §5), then converts depth → displacement via a simple formula: `displacement = (depth - depth.mean()) * intensity_factor`.

### 3.7 Segmentation and object detection
- **Semantic segmentation**: classifies every pixel into a category (shirt vs skin vs background) — good for isolating "the shirt" as a whole.
- **Instance segmentation**: separates individual object instances (useful if there are 2 mugs in a photo).
- **SAM2 (Segment Anything Model 2)**: a promptable segmentation model — give it a point or box, it returns a precise mask. This is the single most useful model for Crevr's auto-ingestion pipeline (see §5), because it needs no training on "t-shirt" specifically — you just click roughly where the design should go, and SAM2 returns the exact flat region mask.

---

## 4. OpenCV Reference for This Project

Practical, mockup-specific usage of the OpenCV functions the agent will actually call:

| Function | Purpose in Crevr |
|---|---|
| `cv2.findContours` | Extract design-zone boundary from a mask |
| `cv2.approxPolyDP` | Reduce a noisy contour to a clean 4-point (or N-point) polygon |
| `cv2.getPerspectiveTransform(src_pts, dst_pts)` | Compute the 3×3 matrix mapping the flat design's 4 corners to the target zone's 4 corners |
| `cv2.warpPerspective(design, matrix, size)` | Apply that matrix — this is the actual "paste design onto surface" operation |
| `cv2.remap` | Apply the displacement map — shifts pixels according to a per-pixel offset field, used for fold/wrinkle simulation after the perspective warp |
| `cv2.GaussianBlur` | Soften mask edges before compositing (avoids jagged/aliased boundaries); also used to create soft shadow blurs |
| `cv2.bilateralFilter` | Edge-preserving blur — smooths noise in a lighting map without losing the sharp edges of fabric folds |
| `cv2.Canny` | Edge detection for automatic corner/boundary discovery |
| `cv2.HoughLinesP` | Detect straight bezel/frame lines (laptops, picture frames, business cards) |
| `cv2.morphologyEx` (MORPH_OPEN / MORPH_CLOSE) | Clean up noisy auto-generated masks |
| `cv2.addWeighted` | Simple linear blend of two images — used for basic overlay preview before full pipeline runs |
| `cv2.seamlessClone` (Poisson blending) | High-quality edge-blending alternative to manual alpha feathering — very useful for texture/highlight integration at design boundaries |
| `cv2.cvtColor` | Convert between RGB/BGR/HSV/LAB/Gray — used constantly since OpenCV loads images as BGR by default (a classic bug source — the agent must convert to RGB immediately after any `cv2.imread`) |
| Bitwise ops (`cv2.bitwise_and/or/not`) | Combine/invert masks |

---

## 5. Displacement Maps — Deep Dive

### 5.1 What Photoshop actually stores
A PSD Smart Object mockup contains, per template, roughly:
- A **Smart Object placeholder layer** (defines the flat zone + its perspective corners)
- A **displacement map layer** (grayscale, same dimensions as the placeholder, encoding fold intensity)
- One or more **light/shadow overlay layers**, set to Multiply or Screen blend mode, painted or extracted from the original photo

### 5.2 How Crevr generates this automatically instead of a human artist
For each template ingestion:
1. Run a monocular depth-estimation model (Depth Anything v2 or MiDaS) on the base photo → raw depth map.
2. Crop the depth map to just the design-zone mask.
3. Normalize it to 0–255, subtract the mean (so "average surface height" = neutral gray = no displacement), scale by a per-template `fold_intensity` constant (t-shirts need strong displacement ~15-25 px equivalent shift; laptops need near-zero, since a screen is flat).
4. Store this as `displacement.png` inside the template folder.
5. At render-time: after `warpPerspective` places the design into the zone, run `cv2.remap` using the displacement map as a per-pixel (dx, dy) offset field, so the design's own edges/prints bend to follow the fabric folds visible in the original photo.

### 5.3 Why this beats naive alpha-overlay
A naive "just paste your logo flat on top" result always looks like a sticker — flat, no fold interaction, obviously fake. The displacement step is what separates an amateur mockup from a professional one, and it is 100% deterministic math, no generative AI required once the map exists.

---

## 6. Where AI Belongs (and Where It Explicitly Does Not)

### 6.1 Explicit boundary
**AI/ML is used only in the "understand this new blank photo" (ingestion) step — never in the "render this specific user's design" (runtime) step.** Runtime rendering is 100% OpenCV/NumPy determinism: same design + same template = pixel-identical output, every time, instantly, with zero GPU cost. This is what makes the free/unlimited business model possible — you're not paying per-image generation compute, you're paying (once) to pre-process each template.

### 6.2 Recommended AI usage (ingestion-time only)
| Task | Model | Why |
|---|---|---|
| Auto-detect the flat design zone in a new user-uploaded blank photo | **SAM2** (Segment Anything 2, Meta) | Promptable (click/box) segmentation, extremely accurate on arbitrary objects, runs locally, Apache-2.0-adjacent license (check current Meta license terms), no training needed |
| Detect object class (t-shirt/laptop/mug/etc.) for applying sane defaults | **Grounding DINO** (open-vocabulary detector) | Text-prompted detection ("find the shirt") without a fixed label set |
| Generate depth map for displacement generation | **Depth Anything v2** (small/base checkpoint) | Best open-source monocular depth model as of 2025-2026, runs fine on CPU for base-size, MIT-licensed |
| Remove background from user-uploaded design PNGs (if they upload a JPG with a busy background instead of a clean transparent PNG) | **RMBG-1.4 / BiRefNet** | Purpose-built background removal, better than generic segmentation for this narrow task |
| Upscale a low-res user design before compositing | **Real-ESRGAN** | Best free upscaler, avoids visible pixelation when a small logo is stretched onto a large product photo |
| Inpaint/remove an unwanted watermark or blemish on a *user's own* base photo before templating it | **LaMa** (Large Mask Inpainting) | Best free inpainting model for object/text removal, MIT-licensed |
| Auto color-correct design to match ambient lighting tone | Classical CV (LAB histogram matching) — **not a model** | This is solved well by deterministic math; don't reach for a generative model where classical CV is more predictable and instant |

### 6.3 Where AI should NOT be used
- Generating the mockup photo itself (defeats the entire "reusable template" model, breaks reproducibility, costs GPU money per render, and is exactly the failed approach the user already tried for a year).
- Any part of the actual render/export pipeline that runs per-user-per-design (must stay instant and free).
- Text/logo generation — Crevr composites what the user uploads; it does not create designs for them (that's a different product).

### 6.4 Local deployment reality check
- **SAM2** (base/small checkpoints), **Depth Anything v2** (small), **ORB/SIFT/Canny/Hough** (native OpenCV, no GPU needed at all) — all run comfortably on a normal dev laptop CPU, or a low-end GPU, in a few seconds per image. This matters because ingestion happens rarely (once per template), not per-render.
- **ONNX Runtime** should be the deployment format for any of these models in production — smaller, faster, no PyTorch dependency bloat at inference time. Export each model to `.onnx` once, load via `onnxruntime` in the FastAPI service.
- None of this requires cloud GPUs for the MVP. A CPU-only local machine (exactly what "local-first" demands) is sufficient because ingestion is infrequent and can be slow (~5-30 sec/template is fine); rendering (the frequent, per-user operation) never touches these ML models at all.

---

## 7. System Architecture

### 7.1 High-level diagram (described)
```
┌─────────────┐      ┌──────────────────┐      ┌─────────────────────┐
│   React     │◄────►│  Node.js Gateway │◄────►│  Python FastAPI      │
│  Frontend   │      │  (auth, routing, │      │  Compositing Engine  │
│ (Fabric.js  │      │   file uploads)  │      │  (OpenCV/NumPy/ONNX) │
│  canvas)    │      └──────────────────┘      └─────────────────────┘
└─────────────┘                                          │
                                                          ▼
                                              ┌───────────────────────┐
                                              │  Template Store        │
                                              │  (local disk / SQLite  │
                                              │   metadata + JSON)     │
                                              └───────────────────────┘
```

### 7.2 Why this split
- **React + Fabric.js/Konva.js**: handles the *interactive* part — drag/resize/rotate the design within the zone boundaries, live low-res preview (client-side canvas transform, not the final compositing math — just a visual approximation so it feels instant).
- **Node.js gateway**: thin layer — receives uploads, validates file types/sizes, forwards jobs to the Python engine, handles auth/session/history if added later. Node is good here purely because the frontend team (or agent) stays in one language for anything non-CV; it is not doing image math.
- **Python FastAPI compositing engine**: does 100% of the actual pixel work — warping, displacement, blending, export. This is intentionally isolated as its own service so it can be scaled independently, run in its own Docker container with OpenCV/NumPy/onnxruntime dependencies, and (if ever needed) be swapped for a queue-based worker pool without touching the frontend/gateway code.
- **Local-first storage**: SQLite for metadata (template list, job history), flat JSON files per template for its geometry/displacement config, local disk (not S3) for images in v1 — matching the explicit requirement that this run entirely on a personal machine with no cloud dependency for the core loop.

### 7.3 Processing pipeline stages (runtime render path)
1. **Validate upload** (file type, size, dimensions, corrupt-file check)
2. **Preprocess design** (background removal if requested, optional upscale via Real-ESRGAN if resolution too low, EXIF strip)
3. **Fit into zone** (client sends final position/scale/rotation chosen interactively; server re-validates it's within template's allowed transform limits)
4. **Perspective warp** (`getPerspectiveTransform` + `warpPerspective`)
5. **Displacement remap** (`cv2.remap` using template's precomputed displacement map)
6. **Lighting/shadow blend** (multiply the template's precomputed light/shadow layer over the warped design; apply LAB-space color correction if enabled)
7. **Edge feathering** (Gaussian-blurred mask edge or `seamlessClone` for the boundary)
8. **Composite onto base photo** (alpha blend using the final processed mask)
9. **Export** (encode to requested format/resolution/DPI)

### 7.4 Ingestion pipeline stages (one-time, per new template)
1. Load raw base photo
2. Run SAM2 (with a rough user-provided click/box, or Grounding DINO auto-detection) → raw design-zone mask
3. Clean mask (morphology open/close)
4. Extract 4 (or N) corner points (`findContours` + `approxPolyDP`, or Hough lines for rectangular objects like laptops)
5. Run Depth Anything v2 on the full photo → depth map
6. Crop/normalize depth map into template's displacement map
7. Extract a lighting/shadow overlay layer: convert base photo region to grayscale, normalize, this becomes the multiply-blend lighting layer
8. Save everything into `templates/<template_id>/` as metadata.json + supporting PNGs
9. Human review step (approve/reject/adjust corners manually via the same Fabric.js canvas used for end-users, just in an admin mode) — because auto-ingestion will not be perfect 100% of the time, and this keeps quality high without requiring the human to hand-paint a displacement map from scratch

---

## 8. Template System Design

Each template lives in its own folder:
```
templates/
  tshirt_white_front_01/
    metadata.json
    preview.jpg
    base.png
    mask.png
    displacement.png
    lighting.png
```

`metadata.json` schema:
```json
{
  "id": "tshirt_white_front_01",
  "category": "apparel",
  "subtype": "t-shirt",
  "label": "White T-Shirt — Front, Studio Light",
  "base_image": "base.png",
  "mask_image": "mask.png",
  "displacement_image": "displacement.png",
  "lighting_image": "lighting.png",
  "design_zone_corners": [[412, 210], [880, 205], [895, 640], [400, 650]],
  "fold_intensity": 18,
  "allow_rotation": true,
  "rotation_limits_deg": [-15, 15],
  "allow_perspective_adjust": false,
  "recommended_design_resolution_px": [1500, 1500],
  "min_upload_resolution_px": [500, 500],
  "max_upload_resolution_px": [6000, 6000],
  "supported_formats": ["png", "jpg", "webp"],
  "export_default_format": "png",
  "export_max_resolution_px": [3000, 3000],
  "created_at": "2026-07-15",
  "engine_version": "1.0"
}
```
Every field here maps directly to a decision made in §5–7. `fold_intensity` is the scale factor from §5.2. `design_zone_corners` feeds `getPerspectiveTransform`. `rotation_limits_deg` prevents users from creating physically nonsensical results (e.g., a logo flipped upside-down on a laptop screen).

---

## 9. User Flow

1. **Home** — pick a category (Apparel / Tech / Print / Packaging / Custom Upload)
2. **Choose Template** — grid of preview thumbnails
3. **Upload Design** — drag-drop PNG/JPG/WebP; auto background-removal offered if no alpha channel detected
4. **Edit Canvas** (Fabric.js) — drag to reposition within zone bounds, scale handle, rotation handle (clamped to template's `rotation_limits_deg`), live low-res preview
5. **Preview (Full Render)** — sends final transform to backend, gets back the true composited high-quality render (this is the only point real OpenCV pipeline runs, not on every drag-frame, to keep the UI snappy)
6. **Export** — choose resolution/format/DPI, download
7. **History** — locally stored list of past renders (SQLite) for quick re-export
8. **Settings** — default export format, default resolution, local storage path
9. **(Future) Marketplace** — user-submitted templates, sharing

---

## 10. Frontend Design Direction

Explicit constraints (per product owner):
- **No dark mode.** White background, clean, high-contrast text.
- **No glassmorphism, no over-designed dashboards.** No frosted-glass panels, no excessive gradients, no unnecessary card shadows.
- **Productivity-first layout**: canvas dominates the screen; controls are a slim sidebar, not a cluttered toolbar.
- **Typography**: one clean sans-serif (system font stack is fine — Inter or similar), large enough to read without strain, minimal font-weight variation (2 weights max: regular + semibold).
- **Motion**: minimal — a fast fade/scale on template selection, no scroll-jacking, no heavy parallax. This is a tool, not a portfolio site.

---

## 11. Backend API Surface (FastAPI)

```
POST   /api/templates                 -> list templates (filterable by category)
GET    /api/templates/{id}            -> full metadata for one template
POST   /api/templates/ingest          -> (admin) run ingestion pipeline on a new base photo
POST   /api/designs/upload            -> upload a user design, returns design_id + preprocessed preview
POST   /api/designs/{id}/remove-bg    -> run background removal on an uploaded design
POST   /api/render                    -> body: {template_id, design_id, transform: {x,y,scale,rotation}, export: {format,resolution,dpi}} -> returns rendered image
GET    /api/render/{job_id}           -> poll render status (if async queue used)
GET    /api/history                   -> list past renders (local SQLite)
DELETE /api/history/{id}              -> delete a past render
```

Validation on every upload: MIME-type sniffing (not just extension check), max file size (e.g. 25MB), max pixel dimensions (e.g. 8000×8000, to prevent decompression-bomb style attacks), and immediate EXIF/metadata stripping on ingest.

---

## 12. Folder Structure

```
Crevr-Mockup-Generator/
  assets/                     # raw source photos (pre-ingestion)
  templates/                  # ingested, ready-to-use templates (see §8)
  frontend/
    src/
      components/
      canvas/                 # Fabric.js/Konva.js wrapper logic
      pages/
  gateway/                    # Node.js — upload handling, routing to Python engine
  engine/                     # Python FastAPI compositing engine
    api/
    pipeline/
      ingest.py
      render.py
      warp.py
      displacement.py
      blend.py
      mask.py
    models/                   # local ONNX model weights (SAM2, Depth Anything, etc.)
    tests/
  data/
    crevr.db                  # SQLite — metadata, history
  docs/
    PRD.md                    # this file
```

---

## 13. Error Handling

- **Corrupt/invalid image**: verify via `PIL.Image.verify()` before any processing; reject with clear error, never crash the pipeline.
- **Missing alpha on a design meant to be transparent**: detect (fully opaque image where transparency was expected) and prompt background removal instead of silently compositing a white box.
- **Unsupported DPI/resolution requests**: clamp to template's declared `export_max_resolution_px`, warn rather than silently upscale beyond source quality.
- **Template corner data missing/corrupted**: ingestion pipeline must fail loudly and flag for human review rather than silently producing a broken template.
- **Zip bombs / decompression attacks**: enforce max decompressed pixel count before allocation, not just file size on disk.

---

## 14. Performance

- Runtime render pipeline (warp+displacement+blend) on a single core, for a ~2000×2000 image, should target well under 1 second — this is pure OpenCV, no ML inference, so this is realistic.
- Use `cv2.setNumThreads()` tuning and NumPy's built-in multi-threaded BLAS backend; avoid Python-level loops entirely.
- Cache decoded template base images/masks/displacement maps in memory (they're reused across every render of that template) rather than re-reading from disk per request.
- Ingestion (ML-model-based) is the only slow path (~seconds, CPU-bound) and should run as a background task, not block the request/response cycle.

---

## 15. Security

- Strip all EXIF/metadata from uploaded and exported images (privacy + prevents embedded scripts in malformed metadata fields).
- Path traversal protection on all file operations involving user-supplied filenames (never trust a client-provided path; always generate server-side UUIDs for stored files).
- Reject any upload whose declared MIME type doesn't match its actual file signature (magic-byte check).

---

## 16. Testing Strategy

- **Unit tests**: each pipeline stage (warp, displacement, blend, mask) tested independently with small synthetic fixture images.
- **Golden master / regression tests**: for each template, keep a known-good reference render for a fixed test design; on every pipeline change, re-render and compare pixel-diff (allow small tolerance for floating-point rounding) — catches unintended visual regressions immediately.
- **Integration tests**: full API round-trip (upload → render → export) against the FastAPI test client.
- **Performance benchmarks**: track render time per template size to catch performance regressions over time.

---

## 17. Roadmap

- **MVP**: 3 existing templates ingested + manually verified, full render pipeline, minimal React canvas UI, local-only, PNG export.
- **Phase 2**: Auto-ingestion pipeline (SAM2 + Depth Anything) so new templates can be added by drag-dropping a raw photo, no manual mask-painting.
- **Phase 3**: More categories (packaging, print, devices), history, batch export (many designs onto one template in one click).
- **Phase 4**: Optional cloud sync (still local-first by default), shareable template links.
- **Enterprise/Marketplace**: user-submitted templates, revenue share, API access for other developers/agencies (Crevr-as-a-service).
- **Future R&D**: video mockups (same pipeline, applied per-frame to a product video with tracked corners instead of a static photo — this is a materially harder problem needing optical flow/tracking, explicitly out of scope until core product is solid).

---

## 18. Explicit Non-Goals for v1

- No generative AI in the render path.
- No CMYK/ICC print-color-management (documented workaround only).
- No video/3D mockups.
- No cloud-hosted multi-tenant SaaS infrastructure — this is local-first by design, matching the explicit requirement to run on the user's own machine via Jules/Antigravity/Claude Code without external hosting cost.
# Crevr Mockup Generator
## Product Requirements Document (PRD)

**Version:** 1.0.0
**Date:** 2026-07-15
**Status:** Draft
**Repository:** https://github.com/aarushxox/Crevr-Mockup-Generator

---

# Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Core Philosophy & Educational Foundation](#2-core-philosophy--educational-foundation)
3. [Computer Vision Fundamentals](#3-computer-vision-fundamentals)
4. [NumPy & Image Processing Fundamentals](#4-numpy--image-processing-fundamentals)
5. [OpenCV Mastery for Mockup Generation](#5-opencv-mastery-for-mockup-generation)
6. [Displacement Mapping & Smart Object Mechanics](#6-displacement-mapping--smart-object-mechanics)
7. [AI Integration Strategy](#7-ai-integration-strategy)
8. [Model Research & Selection](#8-model-research--selection)
9. [System Architecture](#9-system-architecture)
10. [Template System Design](#10-template-system-design)
11. [User Experience & Interface Design](#11-user-experience--interface-design)
12. [API Design & Backend Architecture](#12-api-design--backend-architecture)
13. [Folder Structure & Naming Conventions](#13-folder-structure--naming-conventions)
14. [Error Handling Architecture](#14-error-handling-architecture)
15. [Performance Optimization](#15-performance-optimization)
16. [Security Architecture](#16-security-architecture)
17. [Testing Strategy](#17-testing-strategy)
18. [GitHub Research & Architectural Patterns](#18-github-research--architectural-patterns)
19. [Roadmap & Phased Development](#19-roadmap--phased-development)
20. [Deployment & Operations](#20-deployment--operations)
21. [Monetization Strategy](#21-monetization-strategy)

---

## 1. Executive Summary

### 1.1 Project Vision

Crevr Mockup Generator is designed to become the world's premier free, local-first mockup generation platform. Unlike existing solutions that rely on cloud-based AI image generation or subscription models, Crevr is fundamentally an **image compositing engine** that leverages deterministic computer vision algorithms with optional AI enhancements only where they genuinely improve realism.

**Core Philosophy:** Every mockup must be reproducible, deterministic, and mathematically verifiable. Users should never wonder if their mockup will look different tomorrow because a model was updated. The system should "just work" with consistent, predictable results.

### 1.2 Why This Matters

Professional mockup tools like Placeit, Renderforest, and Smartmockups charge $15-35/month for unlimited access. They lock users into subscription models and proprietary ecosystems. Free alternatives often lack quality, limit resolution, or add watermarks.

Crevr solves this by:
- Being **free and open-source** (MIT licensed)
- Running **entirely locally** (no cloud dependency for core features)
- Providing **deterministic results** (same input = same output, always)
- Supporting **unlimited resolution** (limited only by hardware)
- Supporting **any object** (users can upload their own templates)

### 1.3 Target Audience

**Primary Users:**
- E-commerce sellers (Shopify, Amazon, Meesho sellers)
- Fashion brands and boutiques
- Graphic designers and agencies
- Print-on-demand entrepreneurs
- Marketing professionals
- Product designers
- Students and educators

**Secondary Users:**
- App developers needing app store screenshots
- Content creators and social media managers
- Packaging designers
- Signage and display designers

### 1.4 Core Use Cases

**Use Case 1: Simple T-Shirt Mockup**
- User uploads a design (PNG with transparency)
- Selects a T-shirt template
- System automatically detects the printable area
- Design is warped and displaced to match fabric folds
- Realistic shadows and lighting are applied
- High-resolution export is generated

**Use Case 2: Laptop Screen Mockup**
- User uploads a screenshot or design
- Selects a laptop template
- System detects the screen area via edge detection
- Design is perspective-transformed to match screen angle
- Screen glare and reflections are simulated
- Export is generated

**Use Case 3: Custom Template Upload**
- User uploads a photograph of an object
- System analyzes the image to detect flat areas
- User manually adjusts the perspective zone
- Template is saved for future use
- Automatic detection improves with each use

**Use Case 4: Batch Processing**
- User uploads 50 designs
- Applies them to a single template
- System generates 50 mockups in parallel
- Results are downloaded as a ZIP file

---

## 2. Core Philosophy & Educational Foundation

### 2.1 Traditional Mockup Systems

Before understanding how to build a modern mockup generator, we must understand how traditional systems work. This knowledge provides the foundation for everything we build.

#### 2.1.1 The Photoshop Smart Object Workflow

Photoshop's Smart Object system has been the industry standard for mockup generation for over a decade. Understanding it is crucial.

**What is a Smart Object?**

A Smart Object is essentially a container that holds image data from a raster or vector source. It preserves the source content's intrinsic characteristics, enabling non-destructive editing.

**How Smart Objects Work:**

1. **Embedding:** When you convert a layer to a Smart Object, Photoshop stores the source data separately from the composite image.

2. **Transformation Matrix:** Each Smart Object has an associated transformation matrix that defines how it should be rendered. This matrix can include:
   - Translation (position)
   - Scaling
   - Rotation
   - Skew
   - Perspective

3. **Linked vs Embedded:** Linked Smart Objects reference external files, while embedded Smart Objects store data internally.

4. **Rasterization:** When rendering, Photoshop applies the transformation matrix to the source data, interpolates pixels, and composites the result.

**The Mockup Workflow:**

```
Original Mockup PSD
    ↓
Empty Smart Object Layer
    ↓
User Double-Clicks
    ↓
New Window Opens with Embedded Image
    ↓
User Places Design
    ↓
Save and Close
    ↓
Smart Object Updates with Transform
```

**Why This Works:**

1. **Non-Destructive:** The original design data is never modified
2. **Deterministic:** The same design always produces the same result
3. **Reusable:** The same mockup can be reused with different designs
4. **Parameterized:** The transformation defines the relationship, not the result

#### 2.1.2 Displacement Maps

A displacement map is a grayscale image where pixel intensity represents displacement amount. In mockup generation, displacement maps simulate how fabric folds, wrinkles, and surface irregularities affect a design printed on them.

**The Mathematics of Displacement:**

Given a base image I(x,y) and displacement map D(x,y) where D(x,y) ∈ [0, 255]:

```
Normalized displacement: d(x,y) = D(x,y) / 255.0
Displacement vector: Δ(x,y) = (d(x,y) * scale_x, d(x,y) * scale_y)
Output pixel: O(x + Δx, y + Δy) = I(x,y)
```

**Why Photoshop Uses Displacement:**

1. **Realistic Folds:** Designs appear to follow fabric contours
2. **Wrinkle Preservation:** Wrinkles and folds are maintained
3. **Surface Irregularities:** Bumps, seams, and textures are preserved
4. **Depth Simulation:** Creates illusion of three-dimensionality

**How Smart Objects Use Displacement:**

Photoshop's Smart Objects don't directly use displacement maps. Instead, they use:

1. **Warp Transform:** A grid-based deformation system
2. **Mesh Warp:** A more flexible deformation system
3. **Perspective Transform:** A projective transformation

However, when converting a design to a Smart Object and applying a mockup, you're essentially doing:

```
Design → Smart Object → Transform → Composite
```

Where the transform can be:
- Perspective (for flat surfaces)
- Warp (for curved surfaces)
- A combination of both

### 2.2 Image Compositing Fundamentals

Image compositing is the process of combining multiple images to create a single output image. In our context, we're combining a user design with a template mockup.

#### 2.2.1 The Compositing Equation

The fundamental equation for image compositing is:

```
Output = (Design * Alpha) over (Background * (1 - Alpha))
```

More precisely, for each pixel:

```
R_out = R_design * α_design + R_background * (1 - α_design)
G_out = G_design * α_design + G_background * (1 - α_design)
B_out = B_design * α_design + B_background * (1 - α_design)
A_out = α_design + α_background * (1 - α_design)
```

**Why This Matters:**

1. **Alpha Blending:** Preserves transparency naturally
2. **Premultiplied vs Straight Alpha:** The choice affects edge quality
3. **Order Matters:** Front-to-back compositing is non-commutative

#### 2.2.2 Perspective Transform

Perspective transform maps points from one plane to another using a projective transformation. This is essential for placing designs on surfaces that aren't front-facing.

**The Mathematics:**

```
[x']   [a b c]   [x]
[y'] = [d e f] * [y]
[1]    [g h 1]   [1]
```

Where (x', y') are the transformed coordinates, and the matrix represents the perspective transformation.

**Implementation:**

```
x' = (a*x + b*y + c) / (g*x + h*y + 1)
y' = (d*x + e*y + f) / (g*x + h*y + 1)
```

**Why This Matters:**

1. **Realistic Placement:** Designs appear to be on the surface
2. **Three-Dimensionality:** Creates depth illusion
3. **Accuracy:** Matches real-world projection

#### 2.2.3 Warping

While perspective transforms handle planar surfaces, warping handles non-planar surfaces. A warp is a more general transformation.

**Types of Warps:**

1. **Affine Warp:** Preserves parallelism and proportions
2. **Projective Warp:** Handles perspective (our main focus)
3. **Thin-Plate Spline:** Smooth, flexible warping
4. **Mesh Warp:** Grid-based deformation

**When to Use Warp:**

| Surface Type | Transform | Complexity |
|-------------|-----------|------------|
| Flat screen | Perspective | Low |
| T-shirt front | Perspective + Warp | Medium |
| T-shirt with folds | Displacement + Warp | High |
| Curved packaging | Warp | Medium |
| Fabric draping | Displacement | High |

#### 2.2.4 Lighting and Shadows

Lighting and shadows are what make mockups look realistic. A design that's perfectly placed but lacks proper lighting looks fake.

**Components of Lighting:**

1. **Ambient Lighting:** Uniform base illumination
2. **Diffuse Lighting:** Light scattered by surfaces
3. **Specular Lighting:** Mirror-like reflections
4. **Shadow:** Areas blocked from light

**Shadow Implementation:**

```
Shadow Map Generation:
    ↓
Create binary mask of design
    ↓
Gaussian blur to soften edges
    ↓
Offset for directional shadow
    ↓
Apply opacity for shadow intensity
    ↓
Composite onto background
```

**Highlight Preservation:**

1. **Identify Highlight Areas:** Bright regions on the template
2. **Mask Highlights:** Create a mask of highlight regions
3. **Reduce Design Opacity:** Darken design in highlight areas
4. **Add Gloss:** Apply specular highlights over design

#### 2.2.5 Blending Modes

Blending modes define how layers interact. Understanding them is crucial for realistic compositing.

**Common Blending Modes:**

| Mode | Formula | Use Case |
|------|---------|----------|
| Normal | Out = Foreground | Default |
| Multiply | Out = Foreground * Background | Shadows |
| Screen | Out = 1 - (1-F) * (1-B) | Highlights |
| Overlay | Complex | Contrast enhancement |
| Soft Light | Complex | Subtle lighting |
| Color Dodge | Complex | Brightening |

**Implementation:**

```python
def blend_multiply(foreground, background):
    return (foreground * background) / 255.0

def blend_screen(foreground, background):
    return 255 - ((255 - foreground) * (255 - background)) / 255.0

def blend_overlay(foreground, background):
    mask = background < 128
    result = np.where(
        mask,
        (2 * foreground * background) / 255,
        255 - (2 * (255 - foreground) * (255 - background)) / 255
    )
    return result
```

### 2.3 Color Theory and Color Spaces

Color is fundamental to image processing. Understanding color spaces and how to work with them is essential for realistic mockups.

#### 2.3.1 RGB Color Space

RGB represents color as combinations of Red, Green, and Blue light. It's an additive color model, meaning adding light creates lighter colors.

**Characteristics:**
- 8 bits per channel (0-255) typically
- Device-dependent
- Suitable for displays
- Not perceptually uniform

**Gamma Correction:**

The human eye perceives light non-linearly. Gamma correction compensates for this:

```
Output = Input ^ γ

Where γ ≈ 2.2 (common for displays)
```

**Why Gamma Matters:**

1. **Perception:** Linear RGB doesn't match human perception
2. **Interpolation:** Interpolating linear RGB gives wrong results
3. **Display:** Monitors apply gamma to display images

#### 2.3.2 LAB Color Space

LAB is a perceptually uniform color space. It separates color information from luminance information.

**Components:**
- **L*:** Lightness (0-100)
- **a*:** Green-Red axis (-128 to 127)
- **b*:** Blue-Yellow axis (-128 to 127)

**Why LAB Matters for Color Matching:**

1. **Perceptual Uniformity:** A change of 1 in LAB is roughly equally noticeable everywhere
2. **Separated Luminance:** Can adjust brightness without affecting color
3. **Color Matching:** Easier to match colors perceptually

**LAB Color Matching Algorithm:**

```
Convert Background to LAB
Convert Design to LAB
For each pixel:
    L_design = L_background * (L_design / L_design_mean)
    a_design = a_background * (a_design / a_design_mean)
    b_design = b_background * (b_design / b_design_mean)
```

#### 2.3.3 HSV Color Space

HSV (Hue, Saturation, Value) is a cylindrical color model that's intuitive for color manipulation.

**Components:**
- **Hue:** Color angle (0-360°)
- **Saturation:** Color intensity (0-100%)
- **Value:** Brightness (0-100%)

**Use Cases:**

1. **Color Adjustment:** Modify hue without affecting brightness
2. **Intensity Control:** Adjust saturation independently
3. **Shadow Generation:** Darken by reducing value

#### 2.3.4 CMYK vs RGB

CMYK (Cyan, Magenta, Yellow, Key/Black) is a subtractive color model used in printing.

**Key Differences:**

| Aspect | RGB | CMYK |
|--------|-----|------|
| Model | Additive | Subtractive |
| Use | Displays | Print |
| Gamut | Larger | Smaller |
| Conversion | Default | Lossy |

**Conversion Logic:**

```python
def rgb_to_cmyk(r, g, b):
    c = 1 - (r / 255)
    m = 1 - (g / 255)
    y = 1 - (b / 255)
    k = min(c, m, y)
    if k == 1:
        return (0, 0, 0, 100)
    c = (c - k) / (1 - k)
    m = (m - k) / (1 - k)
    y = (y - k) / (1 - k)
    return (c*100, m*100, y*100, k*100)
```

#### 2.3.5 ICC Profiles

ICC profiles define how colors should be interpreted and displayed. They ensure color consistency across devices.

**Why ICC Profiles Matter:**

1. **Color Consistency:** Same image looks same on different devices
2. **Gamut Mapping:** Handle colors that can't be displayed
3. **Print Accuracy:** Simulate print colors on screen

### 2.4 Image Quality and Export

The quality of exported images is critical for a mockup generator. Users need print-ready files.

#### 2.4.1 DPI (Dots Per Inch)

DPI determines image resolution for print:

```
DPI = Pixel Count / Physical Size (inches)
```

**Standard DPIs:**

| Use Case | DPI | Pixels for A4 |
|----------|-----|---------------|
| Web | 72 | 595 x 842 |
| Standard Print | 300 | 2480 x 3508 |
| High Quality | 600 | 4960 x 7016 |

**Calculating Required Resolution:**

```
Width_pixels = DPI * Width_inches
Height_pixels = DPI * Height_inches
```

#### 2.4.2 Export Formats

**PNG:**
- Lossless compression
- Supports transparency
- Excellent for web use
- Larger file sizes

**JPEG:**
- Lossy compression
- No transparency
- Smaller file sizes
- Artifacts at high compression

**WebP:**
- Lossy or lossless
- Supports transparency
- Better compression than JPEG/PNG
- Modern browser support

**AVIF:**
- Excellent compression
- Supports transparency
- HDR support
- Newer format

**TIFF:**
- Lossless
- Supports CMYK
- Professional printing
- Very large files

**Format Selection:**

```
def choose_format(use_case):
    if use_case == "web_preview":
        return "WebP", {"quality": 85}
    elif use_case == "web_transparent":
        return "PNG", {"compression": 6}
    elif use_case == "print_cmyk":
        return "TIFF", {"compression": "lzw"}
    elif use_case == "social_media":
        return "JPEG", {"quality": 90}
    else:
        return "PNG", {"compression": 6}
```

#### 2.4.3 JPEG Artifacts

JPEG compression creates artifacts that degrade image quality:

1. **Blockiness:** 8x8 blocks become visible
2. **Color Smearing:** Colors bleed across edges
3. **Ringing:** Artifacts around edges

**Minimizing Artifacts:**
- Use higher quality settings (85+)
- Use progressive JPEG
- Avoid multiple re-saves
- Consider alternatives for text/line art

### 2.5 Mathematical Foundations

#### 2.5.1 Linear Algebra

Images are transformed using linear algebra operations.

**Matrix Representation:**

A 1000x1000 RGB image is a 3D tensor:
```
shape: (1000, 1000, 3)
```

**Common Operations:**

```
Scaling: O = I * scalar
Translation: O = I + offset
Rotation: O = rotate_matrix * I
Perspective: O = perspective_matrix * I
```

**Why Linear Algebra Matters:**

1. **Efficiency:** Matrix operations are optimized in hardware
2. **Composition:** Transformations can be combined
3. **Inversion:** Can reverse transformations

#### 2.5.2 Interpolation

When transforming images, we need to sample pixels at new positions.

**Nearest Neighbor:**
```
Choose closest pixel: O(x,y) = I(round(x), round(y))
Pros: Fast, preserves sharp edges
Cons: Blocky, poor quality
```

**Bilinear:**
```
Weighted average of 4 nearest pixels
Pros: Smooth, acceptable quality
Cons: Can blur
```

**Bicubic:**
```
Weighted average of 16 nearest pixels
Pros: Smooth, good quality
Cons: Computationally expensive
```

**Lanczos:**
```
Lanczos kernel for reconstruction
Pros: Excellent quality
Cons: Ringing artifacts
```

#### 2.5.3 Convolution

Convolution is used for blurring, sharpening, edge detection, and more.

**Definition:**

```
(f * g)(x,y) = Σ_i Σ_j f(i,j) * g(x-i, y-j)
```

**Common Kernels:**

```
Gaussian Blur:
[1 2 1]
[2 4 2] / 16
[1 2 1]

Edge Detection (Sobel):
[-1 0 1]
[-2 0 2] / 8
[-1 0 1]

Sharpen:
[ 0 -1  0]
[-1  5 -1]
[ 0 -1  0]
```

#### 2.5.4 Fourier Transform

The Fourier Transform converts images from spatial domain to frequency domain.

**Why This Matters:**

1. **Frequency Analysis:** Identify high-frequency (edges) vs low-frequency (smooth) content
2. **Filtering:** Remove specific frequencies
3. **Compression:** JPEG uses Fourier transforms
4. **Correlation:** Template matching

---

## 3. Computer Vision Fundamentals

Computer Vision enables machines to understand and process visual information. For our mockup generator, CV is essential for automatically detecting template features and applying realistic transformations.

### 3.1 What is Computer Vision?

Computer Vision is a field of artificial intelligence that enables computers to derive meaningful information from digital images and videos. It combines techniques from physics, mathematics, signal processing, and machine learning.

**The Pipeline:**

```
Input Image → Preprocessing → Feature Extraction → Analysis → Output
```

**Why CV for Mockup Generation:**

1. **Automatic Detection:** Find printable areas without manual labeling
2. **Understanding:** Understand surface geometry from images
3. **Adaptability:** Work with user-uploaded templates
4. **Consistency:** Apply the same detection logic consistently

### 3.2 Image Matrix Fundamentals

#### 3.2.1 The Image Matrix

A digital image is fundamentally a matrix of numbers.

**Grayscale Image:**
```
Matrix: 2D array of single values (0-255)
size: (height, width)
```

**RGB Image:**
```
Matrix: 3D array of three values per pixel
size: (height, width, 3)
```

**RGBA Image:**
```
Matrix: 3D array of four values per pixel
size: (height, width, 4)
```

**Memory Layout:**

```
Row-major order (C-style):
[Row0_Channel0, Row0_Channel1, Row0_Channel2, Row1_Channel0, ...]

Column-major order (Fortran-style):
[Row0_Channel0, Row1_Channel0, Row2_Channel0, Row0_Channel1, ...]

OpenCV uses row-major, interleaved format:
[B, G, R, B, G, R, ...]
```

#### 3.2.2 Pixels

A pixel is the smallest unit of an image.

**Properties:**
- Position: (x, y) coordinates
- Value: Color or intensity
- Subpixels: Individual color channels

**Subpixel Accuracy:**

For subpixel accuracy, we consider positions with fractional coordinates:

```
Image at (1.5, 2.5) = Interpolation of nearby pixels
```

**Why Subpixel Matters:**
- Precise transformations
- Smooth animations
- High-quality resampling

#### 3.2.3 Color Channels

**RGB Channels:**
- Red: 0-255
- Green: 0-255  
- Blue: 0-255

**HSV Channels:**
- Hue: 0-179 (OpenCV) or 0-360
- Saturation: 0-255
- Value: 0-255

**LAB Channels:**
- L: 0-255 (scaled from 0-100)
- A: 0-255 (scaled from -128 to 127)
- B: 0-255 (scaled from -128 to 127)

#### 3.2.4 Alpha Channel

The alpha channel represents transparency or opacity.

**Interpretation:**
- 0: Completely transparent
- 255: Completely opaque

**Mathematical Operations:**

```python
def composite(foreground, background, alpha):
    # Alpha channel as float between 0 and 1
    a = alpha / 255.0
    
    # Compositing
    result = foreground * a + background * (1 - a)
    return result
```

**Types of Alpha:**
- **Straight Alpha:** Alpha channel is independent of RGB
- **Premultiplied Alpha:** RGB values are multiplied by alpha

### 3.3 Image Operations

#### 3.3.1 Arithmetic Operations

**Addition:**
```python
result = np.clip(image1 + image2, 0, 255)
```

**Subtraction:**
```python
result = np.clip(image1 - image2, 0, 255)
```

**Multiplication:**
```python
result = np.clip(image1 * image2 / 255, 0, 255)
```

**Division:**
```python
result = np.clip(image1 / (image2 + epsilon) * 255, 0, 255)
```

#### 3.3.2 Logical Operations

**AND:**
```python
result = cv2.bitwise_and(image1, image2)
```

**OR:**
```python
result = cv2.bitwise_or(image1, image2)
```

**XOR:**
```python
result = cv2.bitwise_xor(image1, image2)
```

**NOT:**
```python
result = cv2.bitwise_not(image1)
```

#### 3.3.3 Mathematical Operations

**Histogram Equalization:**
```python
def histogram_equalization(image):
    # Calculate histogram
    hist = np.zeros(256)
    for pixel in image.ravel():
        hist[pixel] += 1
    
    # Calculate cumulative distribution
    cdf = np.cumsum(hist)
    cdf_normalized = cdf / cdf[-1] * 255
    
    # Map pixel values
    return cdf_normalized[image]
```

**Gamma Correction:**
```python
def gamma_correct(image, gamma=1.8):
    lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 
                             for i in range(256)]).astype("uint8")
    return cv2.LUT(image, lookup_table)
```

### 3.4 Masking and Thresholding

#### 3.4.1 Thresholding

Thresholding converts grayscale images to binary masks.

**Simple Threshold:**
```python
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
```

**Adaptive Threshold:**
```python
binary = cv2.adaptiveThreshold(gray, 255, 
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
```

**Otsu's Method:**
```python
_, binary = cv2.threshold(gray, 0, 255, 
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

#### 3.4.2 Morphological Operations

Morphology processes binary images based on shapes.

**Erosion:**
```python
kernel = np.ones((5,5), np.uint8)
eroded = cv2.erode(binary, kernel, iterations=1)
```

**Dilation:**
```python
kernel = np.ones((5,5), np.uint8)
dilated = cv2.dilate(binary, kernel, iterations=1)
```

**Opening:** Erosion followed by dilation
```python
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
```

**Closing:** Dilation followed by erosion
```python
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
```

**Gradient:** Dilation - Erosion
```python
gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
```

**Top Hat:** Original - Opening
```python
tophat = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, kernel)
```

**Black Hat:** Closing - Original
```python
blackhat = cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, kernel)
```

### 3.5 Contours and Shape Analysis

#### 3.5.1 Contour Detection

Contours trace boundaries of connected regions.

```python
contours, hierarchy = cv2.findContours(binary, 
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
```

**Contour Properties:**

| Property | Function | Use |
|----------|----------|-----|
| Area | cv2.contourArea() | Size filtering |
| Perimeter | cv2.arcLength() | Shape filtering |
| Center | cv2.moments() | Positioning |
| Bounding Box | cv2.boundingRect() | Cropping |
| Enclosing Circle | cv2.minEnclosingCircle() | Shape matching |
| Convex Hull | cv2.convexHull() | Shape simplification |

#### 3.5.2 Contour Approximation

Simplify contours using Douglas-Peucker algorithm:

```python
epsilon = 0.01 * cv2.arcLength(contour, True)
approx = cv2.approxPolyDP(contour, epsilon, True)
```

**Applications:**
- Shape simplification
- Corner detection
- Rectangle detection

#### 3.5.3 Shape Matching

Match shapes using Hu moments:

```python
moments = cv2.moments(contour)
hu_moments = cv2.HuMoments(moments)
```

Hu moments are invariant to translation, rotation, and scale.

### 3.6 Perspective Detection and Homography

#### 3.6.1 Homography

Homography is a projective transformation between two planes.

**Mathematical Definition:**
```
x' = H * x
where H is a 3x3 matrix
```

**Computing Homography:**

```python
# Given matching points in two images
src_points = np.array([...])  # Points in source image
dst_points = np.array([...])  # Points in destination image

H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC)
```

#### 3.6.2 Perspective Transform

Applying perspective transform:

```python
transformed = cv2.warpPerspective(image, H, (width, height))
```

**Practical Implementation:**

```python
def apply_perspective(image, src_points, dst_points):
    # src_points: four corners of the area to transform
    # dst_points: four corners of the destination area
    
    # Compute homography
    H, _ = cv2.findHomography(src_points, dst_points)
    
    # Apply perspective transform
    height, width = image.shape[:2]
    result = cv2.warpPerspective(image, H, (width, height))
    
    return result
```

#### 3.6.3 Affine Transform

Affine transforms preserve parallelism and ratios:

```python
M = cv2.getAffineTransform(src_points, dst_points)
transformed = cv2.warpAffine(image, M, (width, height))
```

### 3.7 Edge Detection

#### 3.7.1 Gradient-Based Edge Detection

Edges are where pixel intensity changes rapidly.

**Sobel Operator:**
```python
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
```

**Scharr Operator:**
```python
scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
```

#### 3.7.2 Canny Edge Detection

Canny is a multi-stage edge detection algorithm.

**Steps:**
1. **Gaussian Blur:** Smooth image to reduce noise
2. **Gradient Calculation:** Compute gradient magnitude and direction
3. **Non-Maximum Suppression:** Thin edges to one pixel wide
4. **Double Threshold:** Classify edges as strong, weak, or non-edges
5. **Edge Tracking:** Connect strong edges to weak edges

```python
edges = cv2.Canny(gray, threshold1, threshold2)
```

**Parameter Selection:**
- **threshold1:** Lower bound for edge detection
- **threshold2:** Upper bound for edge detection

**Rule of Thumb:**
- For high-contrast images: high thresholds
- For low-contrast images: low thresholds
- Typically: threshold2 = 2-3 * threshold1

### 3.8 Feature Detection and Matching

#### 3.8.1 Corner Detection

Corners are points with significant intensity variation in multiple directions.

**Harris Corner Detection:**
```python
corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
```

**Shi-Tomasi Corner Detection:**
```python
corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, 
                                  qualityLevel=0.01, minDistance=10)
```

#### 3.8.2 SIFT (Scale-Invariant Feature Transform)

SIFT detects and describes local features invariant to scale and rotation.

```python
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(image, None)
```

**SIFT Properties:**
- Scale invariant
- Rotation invariant
- Illumination invariant
- Distinctive descriptors

#### 3.8.3 SURF (Speeded-Up Robust Features)

SURF is a faster alternative to SIFT:

```python
surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
keypoints, descriptors = surf.detectAndCompute(image, None)
```

#### 3.8.4 ORB (Oriented FAST and Rotated BRIEF)

ORB is a fast, free alternative to SIFT/SURF:

```python
orb = cv2.ORB_create(nfeatures=500)
keypoints, descriptors = orb.detectAndCompute(image, None)
```

#### 3.8.5 Feature Matching

**FLANN Matcher:**
```python
flann = cv2.FlannBasedMatcher()
matches = flann.knnMatch(descriptors1, descriptors2, k=2)
```

**Brute-Force Matcher:**
```python
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
```

**Good Match Filtering:**
```python
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)
```

### 3.9 Image Registration

Image registration aligns different images of the same scene.

**Steps:**

1. **Feature Detection:** Find keypoints in both images
2. **Feature Matching:** Find corresponding points
3. **Transform Estimation:** Compute transformation matrix
4. **Image Warping:** Apply transformation to align images

```python
def register_images(template, image):
    # Detect features
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(image, None)
    
    # Match features
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Filter matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    # Extract matching points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    
    # Compute homography
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # Warp image
    h, w = template.shape[:2]
    result = cv2.warpPerspective(image, H, (w, h))
    
    return result
```

### 3.10 Semantic Segmentation

Semantic segmentation labels every pixel in an image with a class.

**Applications in Mockup Generation:**
- Detect printable areas
- Separate foreground from background
- Identify different surfaces

**Common Models:**
- U-Net
- Mask R-CNN
- DeepLab
- SAM2

### 3.11 Object Detection

Object detection identifies and locates objects in images.

**Applications:**
- Detect where a design should be placed
- Identify objects for template matching
- Validate user-uploaded templates

**Common Models:**
- YOLO (You Only Look Once)
- SSD (Single Shot Detector)
- Faster R-CNN
- RetinaNet

### 3.12 Instance Segmentation

Instance segmentation combines object detection with semantic segmentation.

**Applications:**
- Separate overlapping objects
- Identify specific instances
- Complex scene understanding

**Common Models:**
- Mask R-CNN
- YOLACT
- SOLO

### 3.13 Pose Detection

Pose detection identifies the position and orientation of objects in 3D.

**Applications:**
- Determine camera angle from template image
- Estimate 3D surface orientation
- Improve perspective estimation

**Common Models:**
- MediaPipe Pose
- OpenPose
- HRNet

### 3.14 Depth Estimation

Depth estimation predicts the distance of each pixel from the camera.

**Applications:**
- Surface geometry understanding
- Displacement map generation
- Realistic lighting simulation

**Common Models:**
- MiDaS
- Depth Anything
- DPT (Dense Prediction Transformer)

### 3.15 Optical Flow

Optical flow computes motion between consecutive frames.

**Applications:**
- Video mockup generation
- Motion-based segmentation
- Dynamic template analysis

**Common Algorithms:**
- Lucas-Kanade
- Farnebäck
- Dense Optical Flow

---

## 4. NumPy & Image Processing Fundamentals

NumPy is the foundation of image processing in Python. Understanding how images work as arrays is crucial.

### 4.1 Images as Arrays

**Memory Layout:**

```python
import numpy as np

# Grayscale image: 2D array
gray = np.zeros((height, width), dtype=np.uint8)

# RGB image: 3D array (height, width, channels)
rgb = np.zeros((height, width, 3), dtype=np.uint8)

# RGBA image: 3D array (height, width, 4)
rgba = np.zeros((height, width, 4), dtype=np.uint8)
```

**Memory Storage:**

```
For RGB image:
Memory: [R1, G1, B1, R2, G2, B2, ...]
Stride: width * 3 bytes
```

### 4.2 Matrix Operations

**Element-wise Operations:**

```python
# Addition
result = image1 + image2

# Subtraction
result = image1 - image2

# Multiplication
result = image1 * image2

# Division
result = image1 / (image2 + 1e-10)
```

**Matrix Operations:**

```python
# Dot product
result = np.dot(matrix1, matrix2)

# Transpose
transposed = image.T

# Reshape
reshaped = image.reshape(-1, 3)

# Flatten
flattened = image.flatten()
```

### 4.3 Broadcasting

Broadcasting allows operations on arrays of different shapes.

**Rules:**
1. If arrays don't have same number of dimensions, prepend ones
2. If shapes differ in a dimension, broadcast to match
3. If a dimension is 1, it can be broadcast to any size

**Examples:**

```python
# Add constant to entire image
brightened = image + 50

# Multiply each channel by different factor
factors = np.array([1.2, 0.8, 1.0])
scaled = image * factors

# Compare with mask
mask = np.ones((height, width), dtype=bool)
masked = image[mask]
```

### 4.4 Vectorization

Vectorization applies operations to entire arrays without explicit loops.

**Before (Slow):**
```python
for i in range(height):
    for j in range(width):
        image[i, j] = image[i, j] * 1.2
```

**After (Fast):**
```python
image = image * 1.2
```

**Why Vectorization Matters:**
- 100-1000x speedup
- Uses optimized C code
- SIMD instructions

### 4.5 Pixel Manipulation

**Accessing Pixels:**

```python
# Get pixel at (x, y)
pixel = image[y, x]

# Set pixel at (x, y)
image[y, x] = [255, 0, 0]

# Get region
region = image[y:y+h, x:x+w]

# Set region
image[y:y+h, x:x+w] = new_values
```

**Fancy Indexing:**

```python
# Get all pixels where red > 128
red_pixels = image[image[:, :, 0] > 128]

# Set all pixels where red > 128 to white
image[image[:, :, 0] > 128] = [255, 255, 255]
```

### 4.6 Mask Multiplication

```python
# Create mask
mask = np.zeros((height, width), dtype=np.uint8)
mask[100:200, 100:200] = 255

# Apply mask to image
masked = cv2.bitwise_and(image, image, mask=mask)

# Or using NumPy
masked = image * (mask[:, :, np.newaxis] / 255.0)
```

### 4.7 Image Blending

**Alpha Blending:**

```python
def alpha_blend(img1, img2, alpha):
    return img1 * alpha + img2 * (1 - alpha)
```

**Weighted Blending:**

```python
def weighted_blend(img1, img2, weight1, weight2):
    return (img1 * weight1 + img2 * weight2) / (weight1 + weight2)
```

### 4.8 Interpolation

**Nearest Neighbor:**

```python
def nearest_neighbor(image, scale_x, scale_y):
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale_y), int(w * scale_x)
    
    # Create coordinate grid
    y = np.floor(np.arange(new_h) / scale_y).astype(int)
    x = np.floor(np.arange(new_w) / scale_x).astype(int)
    
    return image[y[:, np.newaxis], x]
```

**Bilinear:**

```python
def bilinear(image, scale_x, scale_y):
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale_y), int(w * scale_x)
    
    # Coordinates in original image
    y = np.arange(new_h) / scale_y
    x = np.arange(new_w) / scale_x
    
    # Four nearest neighbors
    y0 = np.floor(y).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)
    x0 = np.floor(x).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    
    # Weights
    dy = y - y0
    dx = x - x0
    
    # Interpolation
    result = (1 - dy)[:, np.newaxis] * (1 - dx) * image[y0[:, np.newaxis], x0] + \
             (1 - dy)[:, np.newaxis] * dx * image[y0[:, np.newaxis], x1] + \
             dy[:, np.newaxis] * (1 - dx) * image[y1[:, np.newaxis], x0] + \
             dy[:, np.newaxis] * dx * image[y1[:, np.newaxis], x1]
    
    return result
```

### 4.9 Sampling

**Grid Sampling:**

```python
def grid_sample(image, coords):
    """
    Sample image at arbitrary coordinates.
    coords: (H, W, 2) where last dimension is (x, y)
    """
    x = coords[:, :, 0]
    y = coords[:, :, 1]
    
    # Nearest neighbor
    x_int = np.round(x).astype(int)
    y_int = np.round(y).astype(int)
    
    # Clamp
    x_int = np.clip(x_int, 0, image.shape[1] - 1)
    y_int = np.clip(y_int, 0, image.shape[0] - 1)
    
    return image[y_int, x_int]
```

### 4.10 Coordinate Systems

**Image Coordinates:**
- Origin: Top-left corner
- x: Right
- y: Down

**Math Coordinates:**
- Origin: Center
- x: Right
- y: Up

**Converting Between Systems:**

```python
def image_to_math(x, y, width, height):
    return x - width/2, -(y - height/2)

def math_to_image(x, y, width, height):
    return x + width/2, -y + height/2
```

### 4.11 Memory Layout Optimization

**Contiguous Arrays:**

```python
# Ensure contiguous memory
image = np.ascontiguousarray(image)
```

**Stride Information:**

```python
strides = image.strides
# For RGB: (width*3, 3, 1)
# For RGBA: (width*4, 4, 1)
```

**Optimizing Access Patterns:**

```python
# Cache-friendly (row-major)
for y in range(height):
    for x in range(width):
        process(image[y, x])

# Cache-unfriendly (column-major)
for x in range(width):
    for y in range(height):
        process(image[y, x])
```

### 4.12 Performance Optimization

**Memory Views:**

```python
# Create view without copying
view = image[100:200, 100:200]
```

**In-place Operations:**

```python
# In-place addition
image += 50

# In-place multiplication
image *= 1.2
```

**Avoiding Copies:**

```python
# Bad: Creates copy
image2 = image[::2, ::2]

# Good: Creates view (if possible)
image2 = image[:, ::2]
```

**Using np.einsum:**

```python
# Matrix multiplication
result = np.einsum('ij,jk->ik', matrix1, matrix2)

# Channel operations
result = np.einsum('...ij, ...ij->...ij', image, kernel)
```

---

## 5. OpenCV Mastery for Mockup Generation

OpenCV (Open Source Computer Vision Library) is the primary tool for image processing in our pipeline. This section covers every OpenCV function needed.

### 5.1 Core OpenCV Functions

#### 5.1.1 Reading and Writing Images

```python
import cv2
import numpy as np

# Read image
image = cv2.imread('path/to/image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Read with alpha channel
image = cv2.imread('path/to/image.png', cv2.IMREAD_UNCHANGED)

# Write image
cv2.imwrite('path/to/output.jpg', image)
cv2.imwrite('path/to/output.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
cv2.imwrite('path/to/output.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])

# Read from memory
image = cv2.imdecode(np.fromfile('path/to/image.jpg', dtype=np.uint8), cv2.IMREAD_COLOR)

# Write to memory
_, buffer = cv2.imencode('.jpg', image)
data = buffer.tobytes()
```

#### 5.1.2 Display and Visualization

```python
# Display image
cv2.imshow('Window', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Draw on image
cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.circle(image, (cx, cy), radius, (0, 0, 255), -1)
cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
cv2.putText(image, 'Text', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Draw contours
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# Draw keypoints
cv2.drawKeypoints(image, keypoints, image, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Draw matches
cv2.drawMatches(img1, kp1, img2, kp2, matches, out)
```

### 5.2 Image Transformations

#### 5.2.1 Resize

```python
# Simple resize
resized = cv2.resize(image, (new_width, new_height))

# Resize with interpolation
resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

# Scale factor
resized = cv2.resize(image, None, fx=2.0, fy=2.0)

# Interpolation methods:
# cv2.INTER_NEAREST: Nearest neighbor (fast, blocky)
# cv2.INTER_LINEAR: Bilinear (good balance)
# cv2.INTER_CUBIC: Bicubic (smooth, slower)
# cv2.INTER_LANCZOS4: Lanczos (high quality)
# cv2.INTER_AREA: Area resampling (good for downscaling)
```

#### 5.2.2 Rotation

```python
# Get rotation matrix
center = (width/2, height/2)
angle = 45  # degrees
scale = 1.0
M = cv2.getRotationMatrix2D(center, angle, scale)

# Apply rotation
rotated = cv2.warpAffine(image, M, (width, height))

# Rotate with cropping
rotated = cv2.warpAffine(image, M, (width, height), borderMode=cv2.BORDER_CONSTANT)

# Rotate without cropping
cos = np.abs(M[0, 0])
sin = np.abs(M[0, 1])
new_width = int(height * sin + width * cos)
new_height = int(height * cos + width * sin)
M[0, 2] += (new_width / 2) - center[0]
M[1, 2] += (new_height / 2) - center[1]
rotated = cv2.warpAffine(image, M, (new_width, new_height))
```

#### 5.2.3 Flip

```python
# Flip horizontally
flipped = cv2.flip(image, 1)

# Flip vertically
flipped = cv2.flip(image, 0)

# Flip both
flipped = cv2.flip(image, -1)
```

#### 5.2.4 Crop

```python
# Crop region
cropped = image[y1:y2, x1:x2]

# Crop and resize
cropped = cv2.resize(image[y1:y2, x1:x2], (target_width, target_height))
```

#### 5.2.5 Translation

```python
# Translation matrix
M = np.float32([[1, 0, dx], [0, 1, dy]])

# Apply translation
translated = cv2.warpAffine(image, M, (width, height))
```

### 5.3 Perspective and Affine Transforms

#### 5.3.1 Affine Transform

```python
# Get affine transform from 3 points
src_pts = np.float32([[x1, y1], [x2, y2], [x3, y3]])
dst_pts = np.float32([[x1', y1'], [x2', y2'], [x3', y3']])
M = cv2.getAffineTransform(src_pts, dst_pts)

# Apply affine transform
transformed = cv2.warpAffine(image, M, (width, height))

# Create affine transform from parameters
M = cv2.getRotationMatrix2D(center, angle, scale)
```

#### 5.3.2 Perspective Transform

```python
# Get perspective transform from 4 points
src_pts = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
dst_pts = np.float32([[x1', y1'], [x2', y2'], [x3', y3'], [x4', y4']])
H = cv2.getPerspectiveTransform(src_pts, dst_pts)

# Apply perspective transform
transformed = cv2.warpPerspective(image, H, (width, height))

# Perspective transform with border
transformed = cv2.warpPerspective(image, H, (width, height), 
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0, 0))

# Compute homography from points
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
```

#### 5.3.3 Image Stitching

```python
# Stitch images
stitcher = cv2.Stitcher_create()
status, panorama = stitcher.stitch([img1, img2])
```

### 5.4 Image Filtering

#### 5.4.1 Blur and Smoothing

```python
# Gaussian blur
blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)

# Median blur
blurred = cv2.medianBlur(image, ksize)

# Bilateral filter
blurred = cv2.bilateralFilter(image, d, sigma_color, sigma_space)

# Box blur
blurred = cv2.blur(image, (ksize, ksize))

# Custom kernel
kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
blurred = cv2.filter2D(image, -1, kernel)
```

#### 5.4.2 Sharpening

```python
# Sharpening kernel
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpened = cv2.filter2D(image, -1, kernel)

# Unsharp mask
blurred = cv2.GaussianBlur(image, (0, 0), sigma)
sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
```

#### 5.4.3 Edge Detection

```python
# Sobel edges
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
edges = np.sqrt(sobel_x**2 + sobel_y**2)

# Scharr edges
scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)

# Laplacian edges
edges = cv2.Laplacian(gray, cv2.CV_64F)

# Canny edges
edges = cv2.Canny(gray, threshold1, threshold2)
```

#### 5.4.4 Morphological Operations

```python
# Get structuring element
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

# Morphological operations
eroded = cv2.erode(image, kernel, iterations=1)
dilated = cv2.dilate(image, kernel, iterations=1)
opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
```

### 5.5 Thresholding and Segmentation

#### 5.5.1 Global Thresholding

```python
# Simple threshold
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Inverse threshold
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# Truncation threshold
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)

# Zero threshold
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)

# Otsu's threshold
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

#### 5.5.2 Adaptive Thresholding

```python
# Adaptive mean
binary = cv2.adaptiveThreshold(gray, 255, 
                               cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 11, 2)

# Adaptive Gaussian
binary = cv2.adaptiveThreshold(gray, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
```

#### 5.5.3 Color Thresholding

```python
# HSV thresholding
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([h_min, s_min, v_min])
upper = np.array([h_max, s_max, v_max])
mask = cv2.inRange(hsv, lower, upper)

# RGB thresholding
lower = np.array([r_min, g_min, b_min])
upper = np.array([r_max, g_max, b_max])
mask = cv2.inRange(image, lower, upper)
```

### 5.6 Contours and Shape Analysis

#### 5.6.1 Contour Detection

```python
# Find contours
contours, hierarchy = cv2.findContours(binary, 
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

# Contour modes:
# cv2.RETR_EXTERNAL: Only outermost contours
# cv2.RETR_LIST: All contours, no hierarchy
# cv2.RETR_CCOMP: Two-level hierarchy
# cv2.RETR_TREE: Full hierarchy tree

# Contour approximation methods:
# cv2.CHAIN_APPROX_NONE: All points
# cv2.CHAIN_APPROX_SIMPLE: Compressed horizontal/vertical/diagonal
```

#### 5.6.2 Contour Properties

```python
# Area
area = cv2.contourArea(contour)

# Perimeter
perimeter = cv2.arcLength(contour, closed=True)

# Bounding rectangle
x, y, w, h = cv2.boundingRect(contour)

# Minimum area rectangle
rect = cv2.minAreaRect(contour)
box = cv2.boxPoints(rect)

# Minimum enclosing circle
(cx, cy), radius = cv2.minEnclosingCircle(contour)

# Fitting ellipse
ellipse = cv2.fitEllipse(contour)

# Fitting line
vx, vy, x0, y0 = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)

# Convex hull
hull = cv2.convexHull(contour)

# Convexity defects
defects = cv2.convexityDefects(contour, hull)

# Moments
moments = cv2.moments(contour)
cx = int(moments['m10'] / moments['m00'])
cy = int(moments['m01'] / moments['m00'])
```

#### 5.6.3 Contour Approximation

```python
# Approximate contour
epsilon = 0.01 * cv2.arcLength(contour, True)
approx = cv2.approxPolyDP(contour, epsilon, True)

# Match shapes
ret = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0)

# Point in contour
inside = cv2.pointPolygonTest(contour, (x, y), measureDist=False)
```

### 5.7 Feature Detection

#### 5.7.1 Corner Detection

```python
# Harris corners
corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
corners = cv2.dilate(corners, None)
threshold = 0.01 * corners.max()
corner_positions = np.where(corners > threshold)

# Shi-Tomasi corners
corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, 
                                  qualityLevel=0.01, minDistance=10)
corners = np.int0(corners)

# FAST corners
fast = cv2.FastFeatureDetector_create()
keypoints = fast.detect(gray, None)
```

#### 5.7.2 Blob Detection

```python
# Simple blob detector
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 100
params.filterByCircularity = True
params.minCircularity = 0.5
params.filterByConvexity = True
params.minConvexity = 0.5
params.filterByInertia = True
params.minInertiaRatio = 0.5

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(image)
```

### 5.8 Feature Matching

#### 5.8.1 SIFT Features

```python
# Create SIFT detector
sift = cv2.SIFT_create()

# Detect and compute
keypoints, descriptors = sift.detectAndCompute(image, None)

# Draw keypoints
image_with_kp = cv2.drawKeypoints(image, keypoints, None, 
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
```

#### 5.8.2 SURF Features

```python
# Create SURF detector
surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)

# Detect and compute
keypoints, descriptors = surf.detectAndCompute(image, None)
```

#### 5.8.3 ORB Features

```python
# Create ORB detector
orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)

# Detect and compute
keypoints, descriptors = orb.detectAndCompute(image, None)
```

#### 5.8.4 Feature Matching

```python
# Brute force matcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# FLANN matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# Ratio test for good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)
```

### 5.9 Image Registration and Alignment

```python
def align_images(img1, img2):
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Detect features
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    # Match features
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Filter matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    # Extract points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    
    # Find homography
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # Warp image
    h, w = img2.shape[:2]
    result = cv2.warpPerspective(img1, H, (w, h))
    
    return result
```

### 5.10 Image Composition

#### 5.10.1 Alpha Blending

```python
def alpha_blend(foreground, background, alpha):
    # Foreground and background must have same dimensions
    result = foreground * alpha + background * (1 - alpha)
    return result.astype(np.uint8)

def overlay_image(background, foreground, position, alpha=1.0):
    # Position is (x, y) for top-left corner
    x, y = position
    h, w = foreground.shape[:2]
    
    # Extract region from background
    roi = background[y:y+h, x:x+w]
    
    # Blend
    if alpha < 1.0:
        foreground = (foreground * alpha).astype(np.uint8)
    
    # Composite
    if foreground.shape[2] == 4:  # RGBA
        alpha = foreground[:, :, 3] / 255.0
        for c in range(3):
            roi[:, :, c] = (foreground[:, :, c] * alpha + 
                           roi[:, :, c] * (1 - alpha))
    else:
        roi[:, :] = foreground
    
    background[y:y+h, x:x+w] = roi
    return background
```

#### 5.10.2 Image Composition with Mask

```python
def composite_with_mask(background, foreground, mask):
    # Mask is binary (0 or 255)
    mask_float = mask / 255.0
    
    # Expand mask to 3 channels
    mask_3ch = np.stack([mask_float] * 3, axis=2)
    
    # Composite
    result = foreground * mask_3ch + background * (1 - mask_3ch)
    return result.astype(np.uint8)
```

### 5.11 Color Space Conversions

```python
# BGR to RGB
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# BGR to HSV
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

# BGR to LAB
lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

# BGR to YCrCb
ycbcr = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)

# BGR to Grayscale
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

# BGR to RGBA
rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)

# Custom conversions
def bgr_to_hsv_custom(bgr):
    hsv = np.zeros_like(bgr)
    # ... custom implementation
    return hsv
```

### 5.12 Image Pyramid

```python
# Gaussian pyramid
gaussian_pyramid = [image]
for i in range(levels):
    gaussian_pyramid.append(cv2.pyrDown(gaussian_pyramid[-1]))

# Laplacian pyramid
laplacian_pyramid = []
for i in range(levels):
    laplacian = gaussian_pyramid[i] - cv2.pyrUp(gaussian_pyramid[i+1])
    laplacian_pyramid.append(laplacian)

# Applications:
# - Multi-scale processing
# - Image blending
# - Feature detection at different scales
```

### 5.13 Template Matching

```python
def template_matching(image, template):
    # Methods:
    # cv2.TM_CCOEFF: Cross-correlation
    # cv2.TM_CCOEFF_NORMED: Normalized cross-correlation
    # cv2.TM_CCORR: Correlation
    # cv2.TM_CCORR_NORMED: Normalized correlation
    # cv2.TM_SQDIFF: Squared difference
    # cv2.TM_SQDIFF_NORMED: Normalized squared difference
    
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return max_loc, max_val
```

### 5.14 Image Inpainting

```python
# Remove artifacts or fill missing areas
inpainted = cv2.inpaint(image, mask, inpaintRadius, cv2.INPAINT_TELEA)
# Or cv2.INPAINT_NS (Navier-Stokes)
```

### 5.15 Histogram Operations

```python
# Calculate histogram
hist = cv2.calcHist([image], [channel], mask, [256], [0, 256])

# Histogram equalization
equalized = cv2.equalizeHist(gray)

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
equalized = clahe.apply(gray)

# Compare histograms
correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
```

### 5.16 Image Gradients

```python
# Compute gradient magnitude and direction
def compute_gradients(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    direction = np.arctan2(sobel_y, sobel_x)
    return magnitude, direction
```

### 5.17 Camera Calibration

While not directly used for mockup generation, understanding camera calibration helps with perspective estimation.

```python
# Camera calibration parameters
# Requires checkerboard images
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    object_points, image_points, image_size, None, None
)

# Undistort image
undistorted = cv2.undistort(image, mtx, dist, None, mtx)
```

### 5.18 Performance Optimizations

```python
# Use CUDA when available
gpu_image = cv2.cuda_GpuMat()
gpu_image.upload(image)
gpu_result = cv2.cuda.gaussianBlur(gpu_image, (5, 5), 0)
result = gpu_result.download()

# Use OpenMP for parallel processing
cv2.setNumThreads(4)

# Use specific SIMD optimizations
cv2.UMat()  # Unified memory
```

---

## 6. Displacement Mapping & Smart Object Mechanics

This section explains the fundamental mechanics behind Photoshop Smart Objects and how we replicate them programmatically.

### 6.1 What is a Displacement Map?

A displacement map is a grayscale image where each pixel's intensity corresponds to the amount of displacement at that position.

**Visual Representation:**
```
Displacement Map:
[50, 100, 150, 200, ...]
[75, 125, 175, 225, ...]
[...]

Interpretation:
50 = 50/255 = 0.196 displacement
100 = 100/255 = 0.392 displacement
150 = 150/255 = 0.588 displacement
200 = 200/255 = 0.784 displacement
```

**Why This Matters:**
A flat design printed on a flat surface (like paper) requires only perspective transform. But a design printed on fabric, especially with folds, needs displacement.

### 6.2 Photoshop Smart Object Mechanics

#### 6.2.1 Internal Representation

A Photoshop Smart Object contains:
1. **Source Data:** The original image data (embedded or linked)
2. **Transform Matrix:** The transformation applied to the source data
3. **Render Settings:** Blending modes, opacity, effects

**Transform Matrix Components:**

```
[scale_x, shear_x, translate_x]
[shear_y, scale_y, translate_y]
[perspective_x, perspective_y, 1]
```

#### 6.2.2 Rendering Pipeline

```
Source Image
    ↓
Preprocess (color space, alpha)
    ↓
Apply Transform Matrix
    ↓
Apply Displacement (if any)
    ↓
Apply Blending
    ↓
Composite
    ↓
Output
```

#### 6.2.3 Smart Object Conversion

When you convert a layer to a Smart Object:
1. The layer's current content is stored as source data
2. The current transform becomes the transform matrix
3. The layer becomes a "container" for the Smart Object

### 6.3 How Displacement Works

#### 6.3.1 Mathematical Foundation

For a displacement map D(x,y) and source image S(x,y):

```
dx = D(x,y) * scale - offset
dy = D(x,y) * scale - offset
Output(x,y) = S(x + dx, y + dy)
```

**Implementation:**

```python
def apply_displacement(source, displacement, scale=1.0, offset=0):
    h, w = source.shape[:2]
    
    # Normalize displacement to [0, 1]
    disp_norm = displacement / 255.0
    
    # Create coordinate grid
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    
    # Compute displaced coordinates
    dx = (disp_norm - 0.5) * 2 * scale
    dy = (disp_norm - 0.5) * 2 * scale
    x_new = x_coords + dx
    y_new = y_coords + dy
    
    # Sample source at new coordinates
    result = cv2.remap(source, x_new.astype(np.float32), 
                       y_new.astype(np.float32), 
                       cv2.INTER_LINEAR)
    
    return result
```

#### 6.3.2 Displacement Intensity

The intensity of displacement determines how much pixels are shifted.

**Low Intensity (0-30):**
- Subtle texture preservation
- Minimal distortion
- Good for slight surface irregularities

**Medium Intensity (30-100):**
- Moderate distortion
- Visible fabric folds
- Good for clothing mockups

**High Intensity (100-255):**
- Significant distortion
- Strong folds and wrinkles
- Good for extreme cases

#### 6.3.3 Directional Displacement

Displacement can be directional:

```python
def apply_directional_displacement(source, disp_x, disp_y):
    # x and y displacement maps separate
    h, w = source.shape[:2]
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    x_new = x_coords + disp_x / 255.0 * scale
    y_new = y_coords + disp_y / 255.0 * scale
    return cv2.remap(source, x_new.astype(np.float32), y_new.astype(np.float32), cv2.INTER_LINEAR)
```

### 6.4 Generating Displacement Maps

#### 6.4.1 Manual Creation

Displacement maps can be created in any image editor:
1. Create a grayscale image
2. Paint white where displacement should be high
3. Paint black where displacement should be low
4. Save as PNG or JPEG

#### 6.4.2 From Surface Normal Maps

Normal maps encode surface orientation as RGB values:
- Red: X direction
- Green: Y direction
- Blue: Z direction

Convert normal map to displacement:

```python
def normal_to_displacement(normal_map):
    # Extract x and y components
    nx = normal_map[:, :, 0] / 255.0 * 2 - 1
    ny = normal_map[:, :, 1] / 255.0 * 2 - 1
    
    # Compute displacement intensity
    intensity = np.sqrt(nx**2 + ny**2)
    
    # Scale to 0-255
    displacement = (intensity * 255).astype(np.uint8)
    return displacement
```

#### 6.4.3 From Depth Maps

Depth maps store distance from camera:
- White: Close
- Black: Far

Convert depth to displacement:

```python
def depth_to_displacement(depth_map):
    # Invert depth (closer = more displacement)
    inverted = 255 - depth_map
    
    # Apply gamma to emphasize features
    gamma = 1.5
    table = np.array([((i / 255.0) ** gamma) * 255 
                     for i in range(256)]).astype("uint8")
    displacement = cv2.LUT(inverted, table)
    return displacement
```

#### 6.4.4 From AI Models

AI models can generate displacement maps from images:

1. **Depth Estimation Models:** MiDaS, Depth Anything
2. **Surface Normal Models:** GeoNet, Pix2Pix
3. **Image-to-Image Models:** ControlNet, Pix2Pix

**Example Pipeline:**

```python
def generate_displacement_from_ai(image):
    # 1. Estimate depth
    depth = depth_estimation_model(image)
    
    # 2. Normalize to 0-255
    depth_norm = ((depth - depth.min()) / 
                 (depth.max() - depth.min()) * 255).astype(np.uint8)
    
    # 3. Extract gradients
    grad_x = cv2.Sobel(depth_norm, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(depth_norm, cv2.CV_64F, 0, 1)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    displacement = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8)
    
    return displacement
```

### 6.5 How Folds Are Preserved

Folds in fabric create depth variations that affect how a design appears.

**The Problem:**
A design printed on a folded fabric should appear:
1. Wrinkled with the fabric
2. Darker in valleys (shadow)
3. Lighter on peaks (highlight)

**The Solution:**
1. **Displacement Map:** Defines where pixels move
2. **Shadow Map:** Darkens valleys
3. **Highlight Map:** Lightens peaks

**Implementation:**

```python
def apply_fold_effect(design, displacement, shadow_map, highlight_map):
    # 1. Apply displacement
    displaced = apply_displacement(design, displacement)
    
    # 2. Apply shadows
    shadow_strength = shadow_map / 255.0
    shaded = displaced * (1 - shadow_strength * 0.5)
    
    # 3. Apply highlights
    highlight_strength = highlight_map / 255.0
    final = shaded * (1 + highlight_strength * 0.3)
    
    return final.astype(np.uint8)
```

### 6.6 Wrinkle Preservation

Wrinkles are localized folds that create complex patterns.

**Approach:**

1. **Detect Wrinkle Regions:** Use edge detection
2. **Extract Wrinkle Patterns:** Use texture analysis
3. **Apply Wrinkle-Aware Displacement:** Variable displacement

```python
def preserve_wrinkles(design, wrinkle_map, displacement):
    # wrinkle_map: high intensity at wrinkles
    wrinkle_mask = wrinkle_map / 255.0
    
    # Apply more displacement at wrinkles
    weighted_displacement = displacement * (1 + wrinkle_mask * 2)
    
    # Apply displacement
    result = apply_displacement(design, weighted_displacement)
    
    return result
```

### 6.7 Highlight Preservation

Highlights on the base surface should be preserved over the design.

**The Problem:**
When we overlay a design, we lose the original surface highlights.

**The Solution:**
Use the original template's highlights to modulate the design.

```python
def preserve_highlights(design, template_highlights):
    # template_highlights: grayscale, bright regions are highlights
    highlight_ratio = template_highlights / 255.0
    
    # Enhance design where template has highlights
    enhanced = design * (1 + highlight_ratio * 0.5)
    enhanced = np.clip(enhanced, 0, 255)
    
    return enhanced.astype(np.uint8)
```

### 6.8 Realistic Texture Transfer

Texture transfer applies surface texture to the design.

**Approach:**

1. **Extract Texture:** Separate texture from content
2. **Apply Texture to Design:** Add texture as overlay

```python
def transfer_texture(design, texture_map):
    # texture_map: grayscale texture pattern
    texture_norm = texture_map / 255.0
    
    # Apply texture as overlay
    result = design * (0.5 + texture_norm * 0.5)
    result = np.clip(result, 0, 255)
    
    return result.astype(np.uint8)
```

### 6.9 Edge Feathering

Edges should blend smoothly into the surface.

```python
def feather_edges(image, mask, radius=5):
    # Create feathered mask
    feathered = cv2.GaussianBlur(mask.astype(np.float32), (radius*2+1, radius*2+1), 0)
    feathered = np.clip(feathered, 0, 255)
    
    # Apply mask
    result = image * (feathered / 255.0)[:, :, np.newaxis]
    return result.astype(np.uint8)
```

### 6.10 Anti-Aliasing

Anti-aliasing smooths jagged edges.

```python
def anti_alias(image, mask, radius=1):
    # Apply slight blur to edges
    blurred = cv2.GaussianBlur(image, (radius*2+1, radius*2+1), 0)
    
    # Blend based on edge detection
    edges = cv2.Canny(mask, 50, 150)
    edge_mask = cv2.dilate(edges, np.ones((3,3), np.uint8))
    edge_mask = edge_mask / 255.0
    
    result = image * (1 - edge_mask)[:, :, np.newaxis] + blurred * edge_mask[:, :, np.newaxis]
    return result.astype(np.uint8)
```

### 6.11 Color Correction

Design colors should match the surface lighting.

```python
def color_correct(design, template):
    # Convert to LAB
    design_lab = cv2.cvtColor(design, cv2.COLOR_RGB2LAB)
    template_lab = cv2.cvtColor(template, cv2.COLOR_RGB2LAB)
    
    # Adjust luminance
    design_lab[:, :, 0] = template_lab[:, :, 0]
    
    # Match color distribution
    for c in [1, 2]:
        d_mean = np.mean(design_lab[:, :, c])
        t_mean = np.mean(template_lab[:, :, c])
        design_lab[:, :, c] = (design_lab[:, :, c] - d_mean) + t_mean
    
    # Convert back
    result = cv2.cvtColor(design_lab, cv2.COLOR_LAB2RGB)
    return result
```

### 6.12 Implementing Smart Object Replacement

The core algorithm for replacing a Smart Object content:

```python
def replace_smart_object(design, template_data):
    """
    template_data contains:
    - transform_matrix: The perspective/affine transform
    - displacement_map: Optional displacement
    - mask: The printable area mask
    - shadow_map: Optional shadow map
    - highlight_map: Optional highlight map
    - texture_map: Optional texture
    """
    
    # 1. Get transform from template
    transform = template_data['transform_matrix']
    
    # 2. Apply transform to design
    transformed = cv2.warpPerspective(design, transform, 
                                     (template_data['width'], template_data['height']))
    
    # 3. Apply displacement if present
    if 'displacement_map' in template_data:
        transformed = apply_displacement(transformed, template_data['displacement_map'])
    
    # 4. Apply mask
    mask = template_data['mask'] / 255.0
    masked = transformed * mask[:, :, np.newaxis]
    
    # 5. Apply shadows
    if 'shadow_map' in template_data:
        shadows = template_data['shadow_map'] / 255.0
        masked = masked * (1 - shadows * 0.5)
    
    # 6. Apply highlights
    if 'highlight_map' in template_data:
        highlights = template_data['highlight_map'] / 255.0
        masked = masked * (1 + highlights * 0.3)
    
    # 7. Apply texture
    if 'texture_map' in template_data:
        masked = transfer_texture(masked, template_data['texture_map'])
    
    # 8. Feather edges
    if 'feather_radius' in template_data:
        masked = feather_edges(masked, mask, template_data['feather_radius'])
    
    # 9. Anti-alias
    if 'anti_alias_radius' in template_data:
        masked = anti_alias(masked, mask, template_data['anti_alias_radius'])
    
    # 10. Color correction
    if 'color_correct' in template_data:
        template_image = template_data['background']
        masked = color_correct(masked, template_image)
    
    # 11. Composite
    result = composite_with_mask(template_data['background'], masked, mask)
    
    return result
```

### 6.13 Template Data Structure

Each template should include all necessary data for realistic compositing:

```json
{
  "metadata": {
    "name": "T-Shirt Front",
    "category": "apparel",
    "resolution": [2000, 2000],
    "type": "psd_smart_object"
  },
  "transform": {
    "type": "perspective",
    "src_points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
    "dst_points": [[x1', y1'], [x2', y2'], [x3', y3'], [x4', y4']]
  },
  "displacement": {
    "map": "displacement.png",
    "scale": 30,
    "offset": 0
  },
  "mask": "mask.png",
  "shadow": "shadow.png",
  "highlight": "highlight.png",
  "texture": "texture.png",
  "feathering": 5,
  "anti_aliasing": 2,
  "color_correction": true
}
```

### 6.14 AI-Generated Displacement Maps

While AI shouldn't replace deterministic processing, it can help generate displacement maps automatically.

**Approach:**

1. Load template image
2. Run depth estimation model
3. Extract surface gradients from depth map
4. Convert gradients to displacement map
5. Validate and adjust displacement
6. Store with template

```python
def generate_displacement_from_template(template_image):
    # 1. Estimate depth
    depth = estimate_depth(template_image)
    
    # 2. Normalize depth
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
    
    # 3. Compute displacement from depth gradients
    grad_x, grad_y = np.gradient(depth_norm)
    disp = np.sqrt(grad_x**2 + grad_y**2)
    
    # 4. Scale to 0-255
    disp = (disp / disp.max() * 255).astype(np.uint8)
    
    # 5. Smooth displacement
    disp = cv2.GaussianBlur(disp, (3, 3), 0)
    
    return disp
```

### 6.15 Limitations of Displacement

Displacement maps have limitations:

1. **No Hidden Surface Removal:** Displacement doesn't handle self-occlusion
2. **No Shadow Casting:** Displacement doesn't create shadows
3. **No Interpolation Artifacts:** Can create visible artifacts
4. **Scale Dependent:** Works best with specific resolution

**Solutions:**
- Use additional shadow maps for realistic shadowing
- Use high-resolution displacement maps
- Apply appropriate interpolation methods
- Combine with other compositing techniques

---

## 7. AI Integration Strategy

### 7.1 Where AI Should Be Used

AI should augment the compositing pipeline, not replace it.

**Recommended AI Use Cases:**

1. **Automatic Object Detection**
   - Find printable areas in templates
   - Identify object types
   - Extract bounding regions

2. **Automatic Perspective Estimation**
   - Detect vanishing points
   - Estimate surface orientation
   - Compute transformation matrices

3. **Automatic Mask Generation**
   - Segment printable areas
   - Separate foreground/background
   - Generate alpha masks

4. **Automatic Displacement Map Generation**
   - Infer surface geometry
   - Generate from depth maps
   - Compute from surface normals

5. **Automatic Shadow Generation**
   - Predict shadow locations
   - Estimate shadow intensity
   - Generate realistic shadows

6. **Automatic Color Correction**
   - Match color profiles
   - Adjust white balance
   - Apply proper color spaces

7. **Automatic Upscaling**
   - Enhance low-resolution designs
   - Preserve details
   - Improve quality

8. **Automatic Inpainting**
   - Remove unwanted elements
   - Fill gaps in masks
   - Clean up edges

### 7.2 Where AI Should NOT Be Used

AI should NOT be used for:

1. **Core Compositing**
   - Transform application
   - Pixel blending
   - Image warping

2. **Deterministic Operations**
   - Color space conversions
   - Arithmetic operations
   - Mask processing

3. **Quality-Critical Operations**
   - Output rendering
   - High-resolution exports
   - Print-ready generation

4. **Real-Time UI**
   - Preview generation
   - Interactive editing
   - Responsive feedback

### 7.3 AI Pipeline Design

**Architecture:**

```
User Uploads Template
    ↓
AI Analysis Pipeline:
    ├── Object Detection (YOLO/SAM2)
    ├── Segmentation (SAM2/Florence-2)
    ├── Depth Estimation (Depth Anything)
    ├── Displacement Generation
    ├── Shadow Estimation
    └── Color Analysis
    ↓
Validate AI Results
    ↓
Store Template Metadata
    ↓
User Edits Template (Manual Correction)
    ↓
Save Template
```

### 7.4 AI Model Selection Criteria

When selecting AI models, consider:

1. **Accuracy:** How well does it perform?
2. **Speed:** How fast is inference?
3. **Memory:** How much RAM/VRAM?
4. **License:** Can we use it commercially?
5. **Deployment:** Can it run locally?
6. **Maintenance:** How active is development?

### 7.5 Edge Cases for AI

**When AI Fails:**
- Unusual lighting conditions
- Uncommon objects
- Extreme angles
- Poor image quality

**Fallback Strategy:**
1. Detect AI failure
2. Notify user
3. Provide manual tools
4. Fall back to deterministic methods
5. Log failure for improvement

### 7.6 Privacy and Security

**AI Inference Locally:**
- No data leaves the user's machine
- No API costs
- Works offline
- Faster for repeated use

**AI Inference via API:**
- No local GPU required
- Always up-to-date models
- Higher accuracy
- Data privacy concerns

**Recommendation:** Provide both options with local as default.

---

## 8. Model Research & Selection

### 8.1 SAM2 (Segment Anything Model 2)

**Description:** Meta's segmentation model that can segment anything in an image using prompts (points, boxes, text).

**Capabilities:**
- Zero-shot segmentation
- Interactive refinement
- High-quality masks
- Real-time inference

**Use in Mockup Generator:**
- Automatic mask generation for templates
- User-guided segmentation
- Printable area detection

**Strengths:**
- Excellent accuracy
- Works on any object
- Interactive refinement
- Open-source (Apache 2.0)

**Weaknesses:**
- Large model (2GB)
- Requires GPU
- Sometimes oversegments
- Complex installation

**Recommended Use:** Primary segmentation tool for automatic mask generation.

**GPU Requirements:** 8GB+ VRAM for high-resolution images.

**Performance:** ~5-10 seconds for 1024x1024 image on RTX 3090.

### 8.2 Grounding DINO

**Description:** Open-set object detection model that can detect any object based on text prompts.

**Capabilities:**
- Zero-shot detection
- Text-guided detection
- High accuracy
- Fast inference

**Use in Mockup Generator:**
- Automatic template detection
- Object recognition
- Text-based search for templates

**Strengths:**
- Any object with text prompt
- High recall
- Open-source (Apache 2.0)

**Weaknesses:**
- Less accurate for uncommon objects
- Requires text prompt
- GPU-dependent

**Recommended Use:** Template categorization and search.

**GPU Requirements:** 6GB+ VRAM.

**Performance:** ~0.5 seconds per image on RTX 3090.

### 8.3 Florence-2

**Description:** Microsoft's vision-language model that handles multiple vision tasks.

**Capabilities:**
- Object detection
- Segmentation
- Captioning
- OCR
- Depth estimation

**Use in Mockup Generator:**
- Multi-purpose analysis
- Template understanding
- Automatic captioning

**Strengths:**
- Multiple tasks in one model
- Microsoft research
- Good accuracy

**Weaknesses:**
- Large model
- Complex usage
- Limited commercial use

**Recommended Use:** Secondary analysis for template metadata extraction.

**GPU Requirements:** 8GB+ VRAM.

**Performance:** ~2-3 seconds per image.

### 8.4 ControlNet

**Description:** Conditional diffusion model for controlled image generation.

**Capabilities:**
- Image generation with controls
- Edge maps, depth maps, etc.
- Style transfer
- Image editing

**Use in Mockup Generator:**
- Generate displacement maps
- Enhance mockups
- Style transfer
- Background generation

**Strengths:**
- High-quality outputs
- Precise control
- Open-source

**Weaknesses:**
- Requires Stable Diffusion
- GPU-intensive
- Slow inference

**Recommended Use:** Enhanced mockup generation (optional feature).

**GPU Requirements:** 8GB+ VRAM.

**Performance:** ~10-30 seconds per image.

### 8.5 IPAdapter

**Description:** Image prompt adapter for Stable Diffusion.

**Capabilities:**
- Image-conditioned generation
- Style transfer
- Identity preservation

**Use in Mockup Generator:**
- Style transfer for designs
- Generate mockups from text+image

**Strengths:**
- Flexible
- High-quality
- Open-source

**Weaknesses:**
- Requires Stable Diffusion
- Complex setup

**Recommended Use:** Advanced style transfer (future feature).

### 8.6 Flux

**Description:** Next-generation text-to-image model.

**Capabilities:**
- High-quality image generation
- Text rendering
- Style versatility

**Use in Mockup Generator:**
- Generate design suggestions
- Create mockup backgrounds

**Strengths:**
- Excellent quality
- Fast inference
- Open weights

**Weaknesses:**
- Large model
- GPU-intensive

**Recommended Use:** Design suggestion feature.

### 8.7 SDXL (Stable Diffusion XL)

**Description:** Large-scale text-to-image diffusion model.

**Capabilities:**
- High-resolution generation
- Photorealism
- Diverse styles

**Use in Mockup Generator:**
- Background generation
- Design enhancement
- Upscaling

**Strengths:**
- High quality
- Open-source
- Good community

**Weaknesses:**
- GPU-intensive
- Can be slow
- Large model

**Recommended Use:** Optional enhancement features.

### 8.8 RealESRGAN

**Description:** Real-world super-resolution model.

**Capabilities:**
- Upscaling low-resolution images
- Face enhancement
- Detail recovery

**Use in Mockup Generator:**
- Upscaling user designs
- Improving template quality
- Enhancing previews

**Strengths:**
- Good quality
- Open-source
- Relatively fast

**Weaknesses:**
- Sometimes oversharpens
- Artifacts on extreme upscaling

**Recommended Use:** Default upscaling method.

**GPU Requirements:** 4GB+ VRAM.

**Performance:** ~2-5 seconds per image.

### 8.9 LaMa (Large Mask Inpainting)

**Description:** Fast and high-quality inpainting model.

**Capabilities:**
- Mask-based inpainting
- Fast inference
- Good quality

**Use in Mockup Generator:**
- Removing unwanted elements
- Cleaning masks
- Filling gaps

**Strengths:**
- Very fast
- Good quality
- Open-source

**Weaknesses:**
- Limited to masks
- Some artifacts

**Recommended Use:** Default inpainting method.

**GPU Requirements:** 4GB+ VRAM.

**Performance:** ~0.5-2 seconds per image.

### 8.10 MediaPipe

**Description:** Google's framework for building multimodal applied ML pipelines.

**Capabilities:**
- Face detection
- Pose detection
- Hand tracking
- Image classification

**Use in Mockup Generator:**
- Face detection for template verification
- Pose estimation for apparel templates
- Object detection in images

**Strengths:**
- Fast
- Lightweight
- Cross-platform
- Open-source

**Weaknesses:**
- Less accurate than specialized models

**Recommended Use:** Real-time detection and preview features.

**GPU Requirements:** None (CPU inference).

**Performance:** Real-time (30+ fps).

### 8.11 YOLO (You Only Look Once)

**Description:** Real-time object detection system.

**Capabilities:**
- Real-time detection
- Multiple objects
- Good accuracy/speed tradeoff

**Use in Mockup Generator:**
- Template detection
- Object counting
- Scene understanding

**Strengths:**
- Very fast
- Good accuracy
- Open-source
- Many versions

**Weaknesses:**
- Bounding boxes only
- Less accurate than segmentation

**Recommended Use:** Quick detection for UI previews.

**GPU Requirements:** 4GB+ VRAM.

**Performance:** Real-time (30+ fps).

### 8.12 Depth Anything

**Description:** Monocular depth estimation model.

**Capabilities:**
- Single-image depth estimation
- Good generalization
- Fine details

**Use in Mockup Generator:**
- Depth map generation
- Displacement map generation
- Surface geometry understanding

**Strengths:**
- Excellent quality
- Good generalization
- Open-source

**Weaknesses:**
- GPU-intensive
- Requires preprocessing

**Recommended Use:** Displacement map generation.

**GPU Requirements:** 6GB+ VRAM.

**Performance:** ~2-5 seconds per image.

### 8.13 MiDaS

**Description:** Monocular depth estimation from mixed datasets.

**Capabilities:**
- Depth estimation
- Multiple resolution support
- Good generalization

**Use in Mockup Generator:**
- Depth map generation
- Displacement map generation

**Strengths:**
- Reliable
- Well-tested
- Open-source

**Weaknesses:**
- Older model
- Less detailed than Depth Anything

**Recommended Use:** Backup depth estimation.

### 8.14 Model Comparison Table

| Model | Task | Speed | VRAM | License | Recommended |
|-------|------|-------|------|---------|-------------|
| SAM2 | Segmentation | Medium | 8GB | Apache 2.0 | Primary |
| Grounding DINO | Detection | Fast | 6GB | Apache 2.0 | Secondary |
| Florence-2 | Multi-task | Medium | 8GB | Microsoft | Optional |
| ControlNet | Generation | Slow | 8GB | Stability AI | Future |
| IPAdapter | Generation | Slow | 8GB | Stability AI | Future |
| Flux | Generation | Medium | 10GB | Flux | Future |
| SDXL | Generation | Slow | 8GB | Stability AI | Future |
| RealESRGAN | Upscaling | Fast | 4GB | BSD-3 | Primary |
| LaMa | Inpainting | Fast | 4GB | Apache 2.0 | Primary |
| MediaPipe | Detection | Real-time | None | Apache 2.0 | UI |
| YOLO | Detection | Real-time | 4GB | GPL | UI |
| Depth Anything | Depth | Medium | 6GB | MIT | Primary |
| MiDaS | Depth | Medium | 4GB | MIT | Backup |

### 8.15 Model Selection Strategy

**For MVP (Minimum Viable Product):**
1. **YOLO** for object detection (fast, lightweight)
2. **RealESRGAN** for upscaling
3. **LaMa** for inpainting
4. **MiDaS** for depth estimation

**For Phase 2:**
1. **SAM2** for segmentation (better masks)
2. **Depth Anything** for better depth maps
3. **Grounding DINO** for text-based detection

**For Phase 3:**
1. **ControlNet** for displacement map generation
2. **SDXL** for enhancement features
3. **Florence-2** for metadata extraction

**For Phase 4:**
1. **Flux** for design suggestions
2. **IPAdapter** for style transfer
3. **MediaPipe** for real-time features

---

## 9. System Architecture

### 9.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  Design View │  │  Canvas View │  │  Preview View│        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │ Template UI  │  │ Export UI    │  │ Settings UI  │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API LAYER (FastAPI)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  Templates   │  │  Designs     │  │  Export      │        │
│  │  Endpoints   │  │  Endpoints   │  │  Endpoints   │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PROCESSING PIPELINE (Python)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  Image       │  │  AI          │  │  Template    │        │
│  │  Processing  │  │  Pipeline    │  │  Engine      │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  Compositing │  │  Export      │  │  Batch       │        │
│  │  Engine      │  │  Engine      │  │  Engine      │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STORAGE LAYER (SQLite + Filesystem)          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  Database    │  │  Image Cache │  │  Template    │        │
│  │  (Metadata)  │  │  (Blobs)     │  │  Storage     │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 Frontend Architecture

**Technology Stack:**
- **React 18+** for UI components
- **TypeScript** for type safety
- **Fabric.js** for canvas manipulation
- **Zustand** for state management
- **React Router v6** for navigation
- **Vite** for build tooling

**Component Hierarchy:**

```
App
├── Layout
│   ├── Header
│   │   ├── Logo
│   │   ├── Navigation
│   │   └── User Menu
│   ├── MainContent
│   │   ├── Home
│   │   │   ├── TemplateGrid
│   │   │   ├── TemplateCard
│   │   │   └── SearchBar
│   │   ├── Editor
│   │   │   ├── Canvas
│   │   │   │   ├── DesignOverlay
│   │   │   │   ├── TemplateOverlay
│   │   │   │   └── ControlWidget
│   │   │   ├── Toolbar
│   │   │   │   ├── TransformTools
│   │   │   │   ├── ColorTools
│   │   │   │   └── ExportTools
│   │   │   └── Properties
│   │   │       ├── Position
│   │   │       ├── Scale
│   │   │       └── Rotation
│   │   ├── Settings
│   │   │   ├── General
│   │   │   ├── Export
│   │   │   └── AI
│   │   └── About
│   └── Footer
│       ├── Links
│       └── Version
└── Modals
    ├── UploadModal
    ├── ExportModal
    └── TemplateEditor
```

**State Management:**

```typescript
interface AppState {
  // Templates
  templates: Template[];
  selectedTemplate: string | null;
  
  // Design
  design: Design | null;
  designPosition: Position;
  designScale: number;
  designRotation: number;
  
  // UI
  currentView: 'home' | 'editor' | 'settings';
  isLoading: boolean;
  error: string | null;
  
  // Settings
  settings: UserSettings;
  
  // Export
  exportFormat: 'png' | 'jpg' | 'webp';
  exportDPI: number;
}
```

**Canvas Implementation (Fabric.js):**

```typescript
import { fabric } from 'fabric';

class MockupCanvas {
  private canvas: fabric.Canvas;
  private templateImage: fabric.Image | null;
  private designImage: fabric.Image | null;
  
  constructor(canvasElement: HTMLCanvasElement) {
    this.canvas = new fabric.Canvas(canvasElement, {
      selection: false,
      backgroundColor: '#f0f0f0'
    });
    
    this.setupEvents();
  }
  
  loadTemplate(imageData: string) {
    fabric.Image.fromURL(imageData, (img) => {
      this.templateImage = img;
      this.canvas.add(img);
      this.canvas.renderAll();
    });
  }
  
  loadDesign(imageData: string) {
    fabric.Image.fromURL(imageData, (img) => {
      this.designImage = img;
      this.designImage.set({
        left: 100,
        top: 100,
        scaleX: 1,
        scaleY: 1
      });
      this.canvas.add(img);
      this.canvas.renderAll();
    });
  }
  
  applyPerspective(srcPoints: Point[], dstPoints: Point[]) {
    // Perspective transform using Fabric
    const matrix = this.calculatePerspectiveMatrix(srcPoints, dstPoints);
    this.designImage.applyTransformMatrix(matrix);
    this.canvas.renderAll();
  }
}
```

### 9.3 Backend Architecture

**Technology Stack:**
- **Python 3.10+** for processing
- **FastAPI** for API framework
- **Uvicorn** for ASGI server
- **SQLite** for database
- **OpenCV** for image processing
- **NumPy** for numerical operations
- **Pillow** for image handling
- **ONNX Runtime** for AI inference
- **PyTorch** for AI models
- **Celery** for task queue
- **Redis** for caching

**API Layer:**

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Crevr Mockup Generator API", version="1.0.0")

class TemplateCreate(BaseModel):
    name: str
    category: str
    description: Optional[str] = None

class DesignUpload(BaseModel):
    template_id: str
    file_path: str
    transformations: dict

@app.post("/api/templates/")
async def create_template(template: TemplateCreate):
    # Create template metadata
    pass

@app.post("/api/templates/upload")
async def upload_template(file: UploadFile):
    # Upload template image and generate metadata
    pass

@app.post("/api/designs/process")
async def process_design(design: DesignUpload):
    # Process design and generate mockup
    pass

@app.get("/api/templates/")
async def get_templates(category: Optional[str] = None):
    # Get all templates with optional filtering
    pass

@app.post("/api/export/")
async def export_mockup(mockup_id: str, format: str = "png"):
    # Export mockup in requested format
    pass
```

**Processing Pipeline:**

```python
class MockupProcessor:
    def __init__(self):
        self.template_engine = TemplateEngine()
        self.compositing_engine = CompositingEngine()
        self.ai_pipeline = AIPipeline()
        
    def process(self, design_image: np.ndarray, template_id: str) -> np.ndarray:
        # 1. Load template
        template = self.template_engine.load(template_id)
        
        # 2. Analyze template
        template_metadata = self.ai_pipeline.analyze_template(template)
        
        # 3. Preprocess design
        design = self.preprocess_design(design_image, template_metadata)
        
        # 4. Apply transformations
        transformed = self.compositing_engine.transform(
            design, template_metadata['transform']
        )
        
        # 5. Apply displacement
        if 'displacement' in template_metadata:
            transformed = self.compositing_engine.apply_displacement(
                transformed, template_metadata['displacement']
            )
        
        # 6. Composite
        result = self.compositing_engine.composite(
            transformed, template, template_metadata
        )
        
        return result
    
    def preprocess_design(self, design: np.ndarray, template_metadata: dict) -> np.ndarray:
        # Resize, color correct, etc.
        return design
```

### 9.4 Template Engine

```python
class TemplateEngine:
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.db = Database()
        self.cache = {}
        
    def load(self, template_id: str) -> Template:
        # Check cache
        if template_id in self.cache:
            return self.cache[template_id]
        
        # Load from storage
        template_data = self.db.get_template(template_id)
        
        # Load images
        images = self.load_images(template_data['paths'])
        
        # Build template object
        template = Template(
            id=template_id,
            metadata=template_data['metadata'],
            background=images['background'],
            mask=images['mask'],
            displacement=images.get('displacement'),
            shadow=images.get('shadow'),
            highlight=images.get('highlight'),
            texture=images.get('texture')
        )
        
        # Cache
        self.cache[template_id] = template
        
        return template
    
    def save(self, template: Template) -> str:
        # Generate ID if needed
        if not template.id:
            template.id = self.generate_id()
        
        # Save metadata
        self.db.save_template(template)
        
        # Save images
        paths = self.save_images(template)
        
        # Update metadata with paths
        self.db.update_template_paths(template.id, paths)
        
        # Update cache
        self.cache[template.id] = template
        
        return template.id
```

### 9.5 Compositing Engine

```python
class CompositingEngine:
    def __init__(self):
        self.displacement_engine = DisplacementEngine()
        self.color_engine = ColorEngine()
        self.shadow_engine = ShadowEngine()
        
    def transform(self, design: np.ndarray, transform_params: dict) -> np.ndarray:
        """Apply perspective transform to design."""
        src_points = np.array(transform_params['src_points'], dtype=np.float32)
        dst_points = np.array(transform_params['dst_points'], dtype=np.float32)
        
        H = cv2.getPerspectiveTransform(src_points, dst_points)
        result = cv2.warpPerspective(
            design, H,
            (transform_params['width'], transform_params['height']),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        return result
    
    def apply_displacement(self, design: np.ndarray, displacement_map: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Apply displacement map to design."""
        return self.displacement_engine.apply(design, displacement_map, scale)
    
    def composite(self, design: np.ndarray, template: Template, metadata: dict) -> np.ndarray:
        """Composite design onto template."""
        # 1. Apply mask
        masked = design * (template.mask / 255.0)[:, :, np.newaxis]
        
        # 2. Apply shadows
        if template.shadow is not None:
            shadow_strength = template.shadow / 255.0
            masked = masked * (1 - shadow_strength * 0.5)
        
        # 3. Apply highlights
        if template.highlight is not None:
            highlight_strength = template.highlight / 255.0
            masked = masked * (1 + highlight_strength * 0.3)
        
        # 4. Apply color correction
        if metadata.get('color_correct', True):
            masked = self.color_engine.correct(masked, template.background)
        
        # 5. Composite
        result = template.background.copy()
        mask = (template.mask > 0).astype(np.float32)[:, :, np.newaxis]
        result = result * (1 - mask) + masked * mask
        
        return result.astype(np.uint8)
```

### 9.6 AI Pipeline

```python
class AIPipeline:
    def __init__(self):
        self.models = self.load_models()
        
    def load_models(self):
        return {
            'detection': self.load_detection_model(),
            'segmentation': self.load_segmentation_model(),
            'depth': self.load_depth_model(),
            'upscale': self.load_upscale_model(),
            'inpaint': self.load_inpaint_model()
        }
    
    def analyze_template(self, image: np.ndarray) -> dict:
        """Analyze template image and extract metadata."""
        result = {}
        
        # 1. Detect objects
        detections = self.models['detection'].predict(image)
        result['objects'] = detections
        
        # 2. Segment printable area
        mask = self.models['segmentation'].predict(image)
        result['mask'] = mask
        
        # 3. Estimate depth
        depth = self.models['depth'].predict(image)
        result['depth'] = depth
        
        # 4. Generate displacement from depth
        result['displacement'] = self.generate_displacement(depth)
        
        # 5. Estimate perspective
        result['perspective'] = self.estimate_perspective(mask)
        
        # 6. Extract color info
        result['colors'] = self.extract_colors(image)
        
        return result
    
    def generate_displacement(self, depth: np.ndarray) -> np.ndarray:
        """Generate displacement map from depth map."""
        # Normalize depth
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
        
        # Compute gradients
        grad_x, grad_y = np.gradient(depth_norm)
        disp = np.sqrt(grad_x**2 + grad_y**2)
        
        # Scale
        disp = (disp / disp.max() * 255).astype(np.uint8)
        
        # Smooth
        disp = cv2.GaussianBlur(disp, (3, 3), 0)
        
        return disp
    
    def estimate_perspective(self, mask: np.ndarray) -> dict:
        """Estimate perspective transform from mask."""
        # Find contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Approximate to quadrilateral
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:
            # Get corners
            corners = approx.reshape(4, 2)
            
            # Sort corners
            corners = self.sort_corners(corners)
            
            return {
                'src_points': corners.tolist(),
                'dst_points': [
                    [0, 0],
                    [mask.shape[1], 0],
                    [mask.shape[1], mask.shape[0]],
                    [0, mask.shape[0]]
                ]
            }
        
        return None
    
    def sort_corners(self, corners: np.ndarray) -> np.ndarray:
        """Sort corners in clockwise order starting from top-left."""
        # Calculate centroid
        center = np.mean(corners, axis=0)
        
        # Sort by angle
        def angle_from_center(point):
            return np.arctan2(point[1] - center[1], point[0] - center[0])
        
        corners = sorted(corners, key=angle_from_center)
        
        # Ensure top-left is first
        if corners[0][0] + corners[0][1] > corners[1][0] + corners[1][1]:
            corners = corners[1:] + corners[:1]
        
        return np.array(corners)
```

### 9.7 Task Queue Architecture

```python
from celery import Celery
from celery.result import AsyncResult

celery_app = Celery(
    'crevr',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

@celery_app.task
def process_design_task(design_path: str, template_id: str, options: dict):
    """Async task for processing design."""
    processor = MockupProcessor()
    
    # Load design
    design = cv2.imread(design_path)
    
    # Process
    result = processor.process(design, template_id)
    
    # Save result
    output_path = f"output/{uuid.uuid4()}.png"
    cv2.imwrite(output_path, result)
    
    return {'output_path': output_path}

@celery_app.task
def batch_process_task(designs: List[str], template_id: str, options: dict):
    """Batch process multiple designs."""
    results = []
    for design_path in designs:
        result = process_design_task.delay(design_path, template_id, options)
        results.append(result.id)
    
    return {'task_ids': results}
```

### 9.8 Database Schema

```sql
-- SQLite schema

CREATE TABLE templates (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    description TEXT,
    thumbnail_path TEXT,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE template_files (
    id TEXT PRIMARY KEY,
    template_id TEXT REFERENCES templates(id),
    file_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_size INTEGER,
    width INTEGER,
    height INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE designs (
    id TEXT PRIMARY KEY,
    name TEXT,
    template_id TEXT REFERENCES templates(id),
    design_path TEXT NOT NULL,
    output_path TEXT,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE export_jobs (
    id TEXT PRIMARY KEY,
    design_id TEXT REFERENCES designs(id),
    format TEXT NOT NULL,
    dpi INTEGER,
    status TEXT DEFAULT 'pending',
    output_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE TABLE users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_templates_category ON templates(category);
CREATE INDEX idx_designs_template ON designs(template_id);
CREATE INDEX idx_export_jobs_status ON export_jobs(status);
```

### 9.9 Caching Architecture

```python
import redis
import hashlib
import pickle

class Cache:
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
        self.default_ttl = 3600  # 1 hour
        
    def get(self, key: str):
        """Get cached value."""
        data = self.redis.get(key)
        if data:
            return pickle.loads(data)
        return None
    
    def set(self, key: str, value, ttl: int = None):
        """Cache value."""
        ttl = ttl or self.default_ttl
        data = pickle.dumps(value)
        self.redis.setex(key, ttl, data)
    
    def generate_key(self, *args, **kwargs):
        """Generate cache key from arguments."""
        key = f"{args}{kwargs}".encode()
        return hashlib.md5(key).hexdigest()
    
    def invalidate(self, pattern: str):
        """Invalidate cache keys matching pattern."""
        keys = self.redis.keys(pattern)
        if keys:
            self.redis.delete(*keys)

class ImageCache(Cache):
    def get_image(self, path: str):
        """Get cached image."""
        key = self.generate_key('image', path)
        data = self.get(key)
        if data:
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        return None
    
    def set_image(self, path: str, image: np.ndarray):
        """Cache image."""
        key = self.generate_key('image', path)
        _, encoded = cv2.imencode('.png', image)
        self.set(key, encoded.tobytes())
```

### 9.10 Docker Architecture

**Dockerfile:**

```dockerfile
# Frontend
FROM node:18-alpine AS frontend
WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Backend
FROM python:3.10-slim AS backend
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY backend/ ./

# Final image
FROM python:3.10-slim
WORKDIR /app

# Copy frontend build
COPY --from=frontend /app/dist ./frontend/

# Copy backend
COPY --from=backend /app ./backend/

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set environment
ENV PYTHONPATH=/app/backend
ENV FRONTEND_DIR=/app/frontend

# Run
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Docker Compose:**

```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  celery:
    build: .
    command: celery -A backend.tasks worker --loglevel=info
    depends_on:
      - redis
    volumes:
      - ./backend:/app/backend
      - ./data:/app/data
    environment:
      - REDIS_URL=redis://redis:6379/0

  api:
    build: .
    command: uvicorn backend.main:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    depends_on:
      - redis
    volumes:
      - ./backend:/app/backend
      - ./data:/app/data
    environment:
      - REDIS_URL=redis://redis:6379/0

  frontend:
    build:
      context: ./frontend
      target: development
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app
      - /app/node_modules

volumes:
  redis_data:
```

### 9.11 Monitoring and Logging

```python
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class Metrics:
    request_count: int = 0
    error_count: int = 0
    total_processing_time: float = 0.0
    
class Logger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Console handler
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console.setFormatter(formatter)
        self.logger.addHandler(console)
        
    def info(self, message: str):
        self.logger.info(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def debug(self, message: str):
        self.logger.debug(message)
    
    def warn(self, message: str):
        self.logger.warning(message)

class MetricsCollector:
    def __init__(self):
        self.metrics = Metrics()
    
    def track_request(self, processing_time: float):
        self.metrics.request_count += 1
        self.metrics.total_processing_time += processing_time
    
    def track_error(self):
        self.metrics.error_count += 1
    
    def get_metrics(self):
        avg_time = (self.metrics.total_processing_time / 
                   self.metrics.request_count if self.metrics.request_count > 0 else 0)
        return {
            'requests': self.metrics.request_count,
            'errors': self.metrics.error_count,
            'error_rate': self.metrics.error_count / self.metrics.request_count if self.metrics.request_count > 0 else 0,
            'avg_processing_time': avg_time
        }
```

---

## 10. Template System Design

### 10.1 Template Structure

Each template is stored in a dedicated folder:

```
templates/
├── [template-id]/
│   ├── metadata.json
│   ├── preview.png
│   ├── background.png
│   ├── mask.png
│   ├── displacement.png
│   ├── shadow.png
│   ├── highlight.png
│   └── texture.png
```

### 10.2 Metadata Schema

```json
{
  "id": "unique-template-id",
  "name": "Classic T-Shirt Front",
  "category": "apparel",
  "subcategory": "t-shirt",
  "description": "Front view of a classic t-shirt",
  "version": "1.0.0",
  "created_at": "2026-01-15T10:30:00Z",
  "updated_at": "2026-07-15T14:20:00Z",
  "author": "Crevr Team",
  "tags": ["t-shirt", "apparel", "front", "classic"],
  
  "source": {
    "type": "photograph",
    "original_width": 2000,
    "original_height": 2000,
    "orientation": "portrait"
  },
  
  "technical": {
    "recommended_width": 1200,
    "recommended_height": 1200,
    "min_width": 300,
    "min_height": 300,
    "max_width": 4096,
    "max_height": 4096,
    "resolution": 300,
    "color_space": "RGB"
  },
  
  "transform": {
    "type": "perspective",
    "src_points": [
      [200, 400],
      [800, 400],
      [800, 800],
      [200, 800]
    ],
    "dst_points": [
      [0, 0],
      [1000, 0],
      [1000, 1000],
      [0, 1000]
    ],
    "width": 1000,
    "height": 1000
  },
  
  "displacement": {
    "enabled": true,
    "map": "displacement.png",
    "scale": 30,
    "offset": 0,
    "direction": "both"
  },
  
  "mask": {
    "file": "mask.png",
    "feather_radius": 5,
    "anti_aliasing": true
  },
  
  "shadow": {
    "enabled": true,
    "map": "shadow.png",
    "intensity": 0.5,
    "offset": [5, 5],
    "blur": 10
  },
  
  "highlight": {
    "enabled": true,
    "map": "highlight.png",
    "intensity": 0.3
  },
  
  "texture": {
    "enabled": true,
    "map": "texture.png",
    "intensity": 0.2,
    "blend_mode": "overlay"
  },
  
  "editable_zones": [
    {
      "id": "front",
      "name": "Front Area",
      "bounds": [100, 100, 800, 800],
      "transform": "perspective",
      "min_scale": 0.5,
      "max_scale": 2.0
    },
    {
      "id": "sleeve_left",
      "name": "Left Sleeve",
      "bounds": [0, 200, 150, 400],
      "transform": "perspective",
      "min_scale": 0.5,
      "max_scale": 2.0
    }
  ],
  
  "supported_formats": ["png", "jpg", "webp", "avif"],
  
  "ai_analysis": {
    "segmentation": true,
    "depth": true,
    "displacement_generated": true,
    "confidence": 0.92
  },
  
  "statistics": {
    "usage_count": 1234,
    "rating": 4.8,
    "downloads": 567
  }
}
```

### 10.3 Template Editor UI

**Template Editor Interface:**

```
┌─────────────────────────────────────────────────────────────┐
│ Template Editor - "Classic T-Shirt Front"                 │
├─────────────────────────────────────────────────────────────┤
│ ┌──────────────────┐  ┌──────────────────────────────────┐ │
│ │   Toolbar        │  │   Preview Area                   │ │
│ │  ┌──────────────┐│  │   ┌──────────────────────────┐   │ │
│ │  │ Transform    ││  │   │                          │   │ │
│ │  │   └─ Points  ││  │   │  [Template Image]        │   │ │
│ │  │   └─ Grid    ││  │   │                          │   │ │
│ │  │   └─ Hand    ││  │   │                          │   │ │
│ │  ├──────────────┤│  │   │  [Editable Zone]        │   │ │
│ │  │ Mask         ││  │   │                          │   │ │
│ │  │   └─ Manual  ││  │   │  ┌──────────────────┐   │   │ │
│ │  │   └─ Auto    ││  │   │  │  Design Here     │   │   │ │
│ │  │   └─ Refine  ││  │   │  └──────────────────┘   │   │ │
│ │  ├──────────────┤│  │   │                          │   │ │
│ │  │ Displacement ││  │   │                          │   │ │
│ │  │   └─ Manual  ││  │   │                          │   │ │
│ │  │   └─ Auto    ││  │   │                          │   │ │
│ │  │   └─ Adjust  ││  │   │                          │   │ │
│ │  ├──────────────┤│  │   │                          │   │ │
│ │  │ Shadow       ││  │   │                          │   │ │
│ │  │   └─ Manual  ││  │   │                          │   │ │
│ │  │   └─ Auto    ││  │   │                          │   │ │
│ │  │   └─ Adjust  ││  │   │                          │   │ │
│ │  └──────────────┘│  │   │                          │   │ │
│ │  ┌──────────────┐│  │   │                          │   │ │
│ │  │ Save         ││  │   │                          │   │ │
│ │  │ Cancel       ││  │   │                          │   │ │
│ │  └──────────────┘│  │   │                          │   │ │
│ └──────────────────┘  │   └──────────────────────────┘   │ │
│                       └──────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 10.4 Automated Template Detection

```python
class TemplateAnalyzer:
    def __init__(self):
        self.segmentation_model = load_segmentation_model()
        self.depth_model = load_depth_model()
        
    def analyze(self, image: np.ndarray) -> TemplateMetadata:
        """Analyze image and extract template metadata."""
        result = TemplateMetadata()
        
        # 1. Segment printable areas
        masks = self.segmentation_model.predict(image)
        result.editable_zones = self.extract_zones(masks)
        
        # 2. Estimate depth
        depth = self.depth_model.predict(image)
        result.displacement = self.generate_displacement(depth)
        
        # 3. Detect perspective
        perspective = self.detect_perspective(image, masks)
        result.transform = perspective
        
        # 4. Extract shadows
        shadows = self.extract_shadows(image, depth)
        result.shadow = shadows
        
        # 5. Extract highlights
        highlights = self.extract_highlights(image)
        result.highlight = highlights
        
        # 6. Detect object type
        result.category = self.classify_object(image)
        
        return result
    
    def extract_zones(self, masks: np.ndarray) -> List[EditableZone]:
        """Extract editable zones from segmentation masks."""
        zones = []
        contours, _ = cv2.findContours(masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours):
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Get perspective
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:
                points = approx.reshape(4, 2).tolist()
            else:
                points = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
            
            zone = EditableZone(
                id=f"zone_{i}",
                name=f"Zone {i+1}",
                bounds=[x, y, w, h],
                perspective_points=points,
                mask=masks[:, :, i] if masks.ndim > 2 else masks
            )
            zones.append(zone)
        
        return zones
    
    def detect_perspective(self, image: np.ndarray, masks: np.ndarray) -> Transform:
        """Detect perspective from image and masks."""
        # Find largest mask
        if masks.ndim > 2:
            mask = masks[:, :, 0]
        else:
            mask = masks
        
        # Find contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        contour = max(contours, key=cv2.contourArea)
        
        # Approximate to quadrilateral
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:
            corners = approx.reshape(4, 2)
            corners = self.sort_corners(corners)
            
            # Get image dimensions
            h, w = image.shape[:2]
            
            # Transform to rectangle
            dst_points = np.array([
                [0, 0],
                [w, 0],
                [w, h],
                [0, h]
            ], dtype=np.float32)
            
            # Compute homography
            H, _ = cv2.findHomography(corners.astype(np.float32), dst_points)
            
            return Transform(
                src_points=corners.tolist(),
                dst_points=dst_points.tolist(),
                width=w,
                height=h,
                matrix=H.tolist()
            )
        
        return None
```

### 10.5 Template Validation

```python
class TemplateValidator:
    def __init__(self):
        self.rules = [
            self.validate_metadata,
            self.validate_images,
            self.validate_masks,
            self.validate_transforms,
            self.validate_resolution
        ]
    
    def validate(self, template: Template) -> ValidationResult:
        errors = []
        warnings = []
        
        for rule in self.rules:
            result = rule(template)
            if result.errors:
                errors.extend(result.errors)
            if result.warnings:
                warnings.extend(result.warnings)
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def validate_metadata(self, template: Template) -> ValidationResult:
        errors = []
        warnings = []
        
        # Check required fields
        required = ['id', 'name', 'category']
        for field in required:
            if not getattr(template.metadata, field, None):
                errors.append(f"Missing required metadata field: {field}")
        
        # Check category
        valid_categories = ['apparel', 'devices', 'packaging', 'print', 'other']
        if template.metadata.category not in valid_categories:
            warnings.append(f"Unknown category: {template.metadata.category}")
        
        return ValidationResult(errors=errors, warnings=warnings)
    
    def validate_images(self, template: Template) -> ValidationResult:
        errors = []
        warnings = []
        
        # Check required images
        required = ['background', 'mask']
        for field in required:
            if getattr(template, field, None) is None:
                errors.append(f"Missing required image: {field}")
        
        # Check dimensions
        if template.background is not None:
            h, w = template.background.shape[:2]
            if h < 100 or w < 100:
                warnings.append(f"Image too small: {w}x{h}")
        
        return ValidationResult(errors=errors, warnings=warnings)
```

### 10.6 Template Packaging

```python
class TemplatePackager:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        
    def package(self, template: Template) -> Path:
        """Package template into a single file."""
        # Create directory
        template_dir = self.output_dir / template.id
        template_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        with open(template_dir / 'metadata.json', 'w') as f:
            json.dump(template.metadata.to_dict(), f, indent=2)
        
        # Save images
        for name, image in template.images.items():
            if image is not None:
                path = template_dir / f"{name}.png"
                cv2.imwrite(str(path), image)
        
        # Create preview
        preview = self.generate_preview(template)
        cv2.imwrite(str(template_dir / 'preview.png'), preview)
        
        # Create ZIP
        import zipfile
        zip_path = self.output_dir / f"{template.id}.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for file in template_dir.rglob('*'):
                zf.write(file, file.relative_to(template_dir))
        
        return zip_path
    
    def generate_preview(self, template: Template) -> np.ndarray:
        """Generate preview image with design placeholder."""
        # Start with background
        preview = template.background.copy()
        
        # Add design placeholder
        if template.mask is not None:
            mask = template.mask / 255.0
            placeholder = np.ones_like(preview) * 128
            preview = preview * (1 - mask[:, :, np.newaxis]) + placeholder * mask[:, :, np.newaxis]
            preview = preview.astype(np.uint8)
        
        return preview
```

---

## 11. User Experience & Interface Design

### 11.1 Design Philosophy

**Core Principles:**
1. **Simplicity:** Clean, minimal interface focused on the task
2. **Speed:** Instant feedback, no waiting for simple operations
3. **Clarity:** Clear visual hierarchy, obvious actions
4. **Productivity:** Optimized workflows for power users
5. **Approachability:** Easy for beginners, powerful for experts

**Visual Design:**
- White background
- Clean typography
- Subtle shadows for depth
- Consistent spacing
- Clear iconography

**Color Palette:**
```
Primary: #2196F3 (Blue)
Secondary: #4CAF50 (Green)
Accent: #FF9800 (Orange)
Error: #F44336 (Red)
Text: #212121 (Dark)
Text Secondary: #757575 (Gray)
Background: #FFFFFF (White)
Surface: #F5F5F5 (Light Gray)
```

### 11.2 User Flow

#### 11.2.1 Flow 1: Quick Mockup

```
1. User visits home page
   ↓
2. Sees template grid
   ↓
3. Clicks on template
   ↓
4. Editor opens
   ↓
5. User uploads design
   ↓
6. System automatically processes
   ↓
7. Preview shown
   ↓
8. User adjusts if needed
   ↓
9. User exports
```

#### 11.2.2 Flow 2: Custom Template

```
1. User visits home page
   ↓
2. Clicks "Upload Template"
   ↓
3. Selects image from filesystem
   ↓
4. System analyzes template
   ↓
5. AI detects printable area
   ↓
6. User verifies and adjusts
   ↓
7. System saves template
   ↓
8. User creates mockup
```

#### 11.2.3 Flow 3: Batch Processing

```
1. User selects template
   ↓
2. User clicks "Batch Upload"
   ↓
3. User selects multiple designs
   ↓
4. System processes all
   ↓
5. Progress shown
   ↓
6. All mockups generated
   ↓
7. User downloads ZIP
```

### 11.3 Screen Designs

#### 11.3.1 Home Screen

```
┌─────────────────────────────────────────────────────────────┐
│ Logo   Templates   Upload   About   Settings   [User]      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Welcome to Crevr Mockup Generator                         │
│  Create professional mockups in seconds                   │
│                                                             │
│  🔍 Search templates...                                    │
│                                                             │
│  Categories:  [All] [Apparel] [Devices] [Packaging] [Print]│
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Template Grid                                      │   │
│  │                                                     │   │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐            │   │
│  │  │      │ │      │ │      │ │      │            │   │
│  │  │ T-   │ │ T-   │ │ Hoodie│ │ Mac- │            │   │
│  │  │ Shirt │ │ Shirt │ │       │ │ book │            │   │
│  │  │ 1    │ │ 2    │ │       │ │      │            │   │
│  │  └──────┘ └──────┘ └──────┘ └──────┘            │   │
│  │                                                     │   │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐            │   │
│  │  │      │ │      │ │      │ │      │            │   │
│  │  │ Phone │ │ Laptop│ │ Box  │ │ Poster│            │   │
│  │  │       │ │       │ │      │ │      │            │   │
│  │  └──────┘ └──────┘ └──────┘ └──────┘            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Upload Custom Template →                                  │
│                                                             │
│  [Footer Links]                                            │
└─────────────────────────────────────────────────────────────┘
```

#### 11.3.2 Editor Screen

```
┌─────────────────────────────────────────────────────────────┐
│ ← Back   Crevr   [Template Name]   Export  Settings        │
├─────────────────────────────────────────────────────────────┤
│ ┌──────────────┐  ┌──────────────────────────────────────┐ │
│ │   Toolbar    │  │   Canvas Area                       │ │
│ │              │  │                                      │ │
│ │  ┌──────────┐│  │   ┌────────────────────────────┐    │ │
│ │  │ Select   ││  │   │                            │    │ │
│ │  ├──────────┤│  │   │      Template Image         │    │ │
│ │  │ Transform││  │   │                            │    │ │
│ │  │   └─ Move││  │   │    ┌────────────────┐      │    │ │
│ │  │   └─ Scale││  │   │    │  Design        │      │    │ │
│ │  │   └─ Rotate││  │   │    │  Overlay       │      │    │ │
│ │  │   └─ Reset││  │   │    └────────────────┘      │    │ │
│ │  ├──────────┤│  │   │                            │    │ │
│ │  │ Colors   ││  │   │                            │    │ │
│ │  │   └─ Pick││  │   │                            │    │ │
│ │  │   └─ Match││  │   │                            │    │ │
│ │  ├──────────┤│  │   │                            │    │ │
│ │  │ Effects  ││  │   │                            │    │ │
│ │  │   └─ Shadow││  │   │                            │    │ │
│ │  │   └─ Blur││  │   │                            │    │ │
│ │  │   └─ Glow││  │   │                            │    │ │
│ │  ├──────────┤│  │   │                            │    │ │
│ │  │ Undo     ││  │   │                            │    │ │
│ │  │ Redo     ││  │   │                            │    │ │
│ │  └──────────┘│  │   └────────────────────────────┘    │ │
│ └──────────────┘  │                                      │ │
│                   │  ┌──────────────────────────────────┐ │ │
│                   │  │   Properties Panel              │ │ │
│                   │  │                                 │ │ │
│                   │  │   Position:  (x, y)            │ │ │
│                   │  │   Scale:     [1.0]             │ │ │
│                   │  │   Rotation:  [0°]              │ │ │
│                   │  │   Opacity:   [100%]            │ │ │
│                   │  │                                 │ │ │
│                   │  │   Template:  [Name]            │ │ │
│                   │  │   Resolution: [2000x2000]      │ │ │
│                   │  │                                 │ │ │
│                   │  │   [Export] [Reset] [Save]      │ │ │
│                   │  └──────────────────────────────────┘ │ │
│                   └──────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

#### 11.3.3 Export Dialog

```
┌─────────────────────────────────────────────────────────────┐
│  Export Mockup                                             │
│                                                             │
│  Format:                                                    │
│  ○ PNG (Recommended)                                       │
│  ● JPEG                                                    │
│  ○ WebP                                                    │
│  ○ AVIF                                                    │
│  ○ TIFF                                                    │
│                                                             │
│  Resolution:                                                │
│  [DPI:  300] [Width:  2000] [Height:  2000]               │
│                                                             │
│  Quality:                                                   │
│  [━━━━━━━━━━━━━━━●━━━━━━━━]  90%                          │
│                                                             │
│  Options:                                                   │
│  ☑ Preserve transparency                                    │
│  ☑ Include shadow                                           │
│  ☐ Include watermark                                        │
│                                                             │
│  [Export] [Cancel]                                         │
└─────────────────────────────────────────────────────────────┘
```

### 11.4 Responsive Design

**Breakpoints:**
- Mobile: < 768px
- Tablet: 768px - 1024px
- Desktop: > 1024px

**Mobile Adaptations:**
- Stack toolbar vertically
- Hide properties panel
- Touch-friendly controls
- Larger buttons

### 11.5 Accessibility

**WCAG 2.1 AA Compliance:**
- Color contrast ratios
- Keyboard navigation
- Screen reader support
- Focus indicators
- Alt text for images

### 11.6 Performance Goals

**Loading:**
- First contentful paint: < 1s
- Time to interactive: < 2s
- Largest contentful paint: < 2.5s

**Interaction:**
- Input latency: < 100ms
- Canvas updates: 60fps
- Export generation: < 5s for standard images

---

## 12. API Design & Backend Architecture

### 12.1 API Overview

**Base URL:** `/api/v1`

**Authentication:** JWT-based (optional for local-first)

**Headers:**
```
Content-Type: application/json
Authorization: Bearer <token>
```

### 12.2 Endpoints

#### 12.2.1 Templates

```
GET /api/templates
GET /api/templates/{id}
POST /api/templates
PUT /api/templates/{id}
DELETE /api/templates/{id}
POST /api/templates/upload
POST /api/templates/{id}/analyze
```

**Request/Response Examples:**

```json
// GET /api/templates?category=apparel&limit=20
{
  "templates": [
    {
      "id": "tpl_123",
      "name": "Classic T-Shirt",
      "category": "apparel",
      "thumbnail": "/api/templates/tpl_123/thumbnail",
      "description": "Front view t-shirt",
      "usage_count": 1234,
      "created_at": "2026-01-15T10:30:00Z"
    }
  ],
  "total": 100,
  "page": 1,
  "limit": 20
}

// POST /api/templates/upload
// Multipart form data with image file
{
  "name": "My Custom Template",
  "category": "apparel",
  "description": "Custom t-shirt template"
}
// Response:
{
  "id": "tpl_456",
  "status": "analyzing",
  "message": "Template uploaded and being analyzed"
}
```

#### 12.2.2 Designs

```
POST /api/designs/process
POST /api/designs/batch
GET /api/designs/{id}
GET /api/designs/{id}/status
DELETE /api/designs/{id}
```

**Request/Response Examples:**

```json
// POST /api/designs/process
{
  "template_id": "tpl_123",
  "design_file": "base64_encoded_image",
  "transformations": {
    "scale": 1.0,
    "rotation": 0,
    "position": [0, 0]
  },
  "options": {
    "quality": 90,
    "format": "png",
    "resolution": 300
  }
}
// Response:
{
  "job_id": "job_789",
  "status": "processing",
  "estimated_time": 5
}

// GET /api/designs/{id}/status
{
  "job_id": "job_789",
  "status": "completed",
  "result": {
    "output_url": "/api/designs/job_789/output",
    "preview_url": "/api/designs/job_789/preview",
    "format": "png",
    "width": 2000,
    "height": 2000,
    "size_bytes": 2450000
  }
}
```

#### 12.2.3 Export

```
POST /api/export
GET /api/export/{job_id}
GET /api/export/{job_id}/download
```

**Request/Response Examples:**

```json
// POST /api/export
{
  "design_id": "design_123",
  "format": "png",
  "dpi": 300,
  "width": 2000,
  "height": 2000,
  "options": {
    "transparent": true,
    "include_shadow": true,
    "quality": 90
  }
}
// Response:
{
  "job_id": "exp_456",
  "status": "pending"
}
```

### 12.3 WebSocket Events

```
ws://localhost:8000/ws/{job_id}

Events:
- progress: { percentage: 50, message: "Processing..." }
- complete: { output_url: "...", preview_url: "..." }
- error: { error: "Invalid image format" }
```

### 12.4 Processing Pipeline API

```python
from pydantic import BaseModel
from typing import Optional, List
from enum import Enum

class TransformType(str, Enum):
    PERSPECTIVE = "perspective"
    AFFINE = "affine"
    WARP = "warp"

class TransformParameters(BaseModel):
    type: TransformType
    src_points: List[List[float]]
    dst_points: List[List[float]]
    width: int
    height: int

class ProcessingOptions(BaseModel):
    quality: int = 90
    format: str = "png"
    resolution: int = 300
    color_correct: bool = True
    apply_shadow: bool = True
    apply_highlight: bool = True
    apply_displacement: bool = True
    feather_radius: int = 5
    anti_aliasing: bool = True

class ProcessRequest(BaseModel):
    template_id: str
    design_file: str  # base64 or URL
    transform: Optional[TransformParameters] = None
    options: ProcessingOptions = ProcessingOptions()
```

### 12.5 Error Responses

```json
{
  "error": {
    "code": "INVALID_IMAGE",
    "message": "The uploaded image is not in a supported format",
    "details": {
      "supported_formats": ["png", "jpg", "webp", "gif"],
      "received": "bmp"
    }
  }
}
```

**Error Codes:**

| Code | Description | HTTP Status |
|------|-------------|-------------|
| INVALID_IMAGE | Image format not supported | 400 |
| IMAGE_TOO_LARGE | Image exceeds size limits | 400 |
| TEMPLATE_NOT_FOUND | Template ID doesn't exist | 404 |
| PROCESSING_FAILED | Image processing error | 500 |
| EXPORT_FAILED | Export generation error | 500 |
| RATE_LIMITED | Too many requests | 429 |
| UNAUTHORIZED | Authentication required | 401 |
| FORBIDDEN | Insufficient permissions | 403 |

---

## 13. Folder Structure & Naming Conventions

### 13.1 Project Root Structure

```
crevr-mockup-generator/
├── frontend/                          # React frontend
│   ├── public/
│   │   ├── favicon.ico
│   │   └── index.html
│   ├── src/
│   │   ├── components/
│   │   │   ├── common/
│   │   │   ├── editor/
│   │   │   ├── templates/
│   │   │   └── settings/
│   │   ├── hooks/
│   │   ├── services/
│   │   ├── store/
│   │   ├── styles/
│   │   ├── types/
│   │   ├── utils/
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── README.md
├── backend/                           # Python backend
│   ├── src/
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   ├── schemas/
│   │   │   └── dependencies/
│   │   ├── core/
│   │   │   ├── compositing/
│   │   │   ├── templates/
│   │   │   ├── processing/
│   │   │   └── export/
│   │   ├── ai/
│   │   │   ├── models/
│   │   │   ├── pipelines/
│   │   │   └── inference/
│   │   ├── storage/
│   │   │   ├── database/
│   │   │   ├── cache/
│   │   │   └── filesystem/
│   │   ├── workers/
│   │   │   ├── tasks/
│   │   │   └── queue/
│   │   ├── utils/
│   │   └── main.py
│   ├── tests/
│   │   ├── unit/
│   │   ├── integration/
│   │   └── fixtures/
│   ├── requirements.txt
│   ├── pyproject.toml
│   └── README.md
├── models/                            # AI models
│   ├── segmentation/
│   ├── depth/
│   ├── detection/
│   ├── upscale/
│   └── inpainting/
├── data/                              # Persistent data
│   ├── templates/
│   ├── designs/
│   ├── exports/
│   └── cache/
├── templates/                         # Template definitions
│   ├── apparel/
│   ├── devices/
│   ├── packaging/
│   ├── print/
│   └── other/
├── docs/                              # Documentation
│   ├── architecture.md
│   ├── api.md
│   ├── deployment.md
│   └── development.md
├── scripts/                           # Development scripts
│   ├── build.py
│   ├── deploy.py
│   └── test.py
├── docker/
│   ├── Dockerfile.frontend
│   ├── Dockerfile.backend
│   └── docker-compose.yml
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── release.yml
├── .gitignore
└── README.md
```

### 13.2 Backend Module Structure

```
backend/src/
├── api/
│   ├── routes/
│   │   ├── templates.py
│   │   ├── designs.py
│   │   ├── export.py
│   │   └── health.py
│   ├── schemas/
│   │   ├── template.py
│   │   ├── design.py
│   │   └── export.py
│   └── dependencies/
│       ├── auth.py
│       └── database.py
├── core/
│   ├── compositing/
│   │   ├── engine.py
│   │   ├── transform.py
│   │   ├── displacement.py
│   │   ├── shadow.py
│   │   ├── color.py
│   │   └── blend.py
│   ├── templates/
│   │   ├── loader.py
│   │   ├── analyzer.py
│   │   ├── validator.py
│   │   └── packager.py
│   ├── processing/
│   │   ├── pipeline.py
│   │   ├── preprocess.py
│   │   ├── postprocess.py
│   │   └── batch.py
│   └── export/
│       ├── generator.py
│       ├── formats.py
│       └── quality.py
├── ai/
│   ├── models/
│   │   ├── segmentation.py
│   │   ├── depth.py
│   │   ├── detection.py
│   │   ├── upscale.py
│   │   └── inpainting.py
│   ├── pipelines/
│   │   ├── template_analyzer.py
│   │   ├── design_enhancer.py
│   │   └── background_remover.py
│   └── inference/
│       ├── local.py
│       └── remote.py
├── storage/
│   ├── database/
│   │   ├── models.py
│   │   ├── session.py
│   │   └── migrations/
│   ├── cache/
│   │   ├── redis.py
│   │   └── filesystem.py
│   └── filesystem/
│       ├── manager.py
│       └── paths.py
├── workers/
│   ├── tasks/
│   │   ├── process.py
│   │   ├── batch.py
│   │   └── cleanup.py
│   └── queue/
│       ├── celery.py
│       └── scheduler.py
├── utils/
│   ├── image.py
│   ├── colors.py
│   ├── math.py
│   └── validators.py
└── main.py
```

### 13.3 Frontend Module Structure

```
frontend/src/
├── components/
│   ├── common/
│   │   ├── Button.tsx
│   │   ├── Input.tsx
│   │   ├── Modal.tsx
│   │   ├── Spinner.tsx
│   │   └── Toast.tsx
│   ├── editor/
│   │   ├── Canvas.tsx
│   │   ├── Toolbar.tsx
│   │   ├── Properties.tsx
│   │   ├── Controls.tsx
│   │   └── Preview.tsx
│   ├── templates/
│   │   ├── TemplateGrid.tsx
│   │   ├── TemplateCard.tsx
│   │   ├── TemplateUpload.tsx
│   │   └── TemplateEditor.tsx
│   ├── settings/
│   │   ├── General.tsx
│   │   ├── Export.tsx
│   │   └── AI.tsx
│   └── layout/
│       ├── Header.tsx
│       ├── Footer.tsx
│       └── Layout.tsx
├── hooks/
│   ├── useCanvas.ts
│   ├── useTemplates.ts
│   ├── useExport.ts
│   └── useSettings.ts
├── services/
│   ├── api.ts
│   ├── templates.ts
│   ├── designs.ts
│   ├── export.ts
│   └── websocket.ts
├── store/
│   ├── app.ts
│   ├── templates.ts
│   ├── editor.ts
│   ├── settings.ts
│   └── export.ts
├── styles/
│   ├── globals.css
│   ├── variables.css
│   └── themes/
├── types/
│   ├── template.ts
│   ├── design.ts
│   ├── export.ts
│   └── settings.ts
├── utils/
│   ├── image.ts
│   ├── file.ts
│   ├── color.ts
│   └── transform.ts
├── App.tsx
└── main.tsx
```

### 13.4 Naming Conventions

**Python (Backend):**
- Modules: snake_case (e.g., `template_engine.py`)
- Classes: PascalCase (e.g., `TemplateEngine`)
- Functions: snake_case (e.g., `process_design()`)
- Constants: UPPER_SNAKE_CASE (e.g., `MAX_IMAGE_SIZE`)
- Private methods: `_underscore` prefix

**TypeScript/React (Frontend):**
- Components: PascalCase (e.g., `TemplateCard`)
- Hooks: camelCase with `use` prefix (e.g., `useTemplates`)
- Services: camelCase (e.g., `apiService`)
- Types: PascalCase (e.g., `Template`)
- Constants: UPPER_SNAKE_CASE

**Files:**
- Python: snake_case
- TypeScript: camelCase
- CSS: kebab-case
- JSON: snake_case for keys

---

## 14. Error Handling Architecture

### 14.1 Error Hierarchy

```python
class CrevrError(Exception):
    """Base exception for all Crevr errors."""
    pass

class ImageError(CrevrError):
    """Image processing related errors."""
    pass

class TemplateError(CrevrError):
    """Template related errors."""
    pass

class ValidationError(CrevrError):
    """Validation related errors."""
    pass

class ProcessingError(CrevrError):
    """Processing pipeline errors."""
    pass

class ExportError(CrevrError):
    """Export related errors."""
    pass

class AICreationError(CrevrError):
    """AI pipeline errors."""
    pass
```

### 14.2 Error Codes

```python
ERROR_CODES = {
    # 1xxx: Image Errors
    'E1001': 'Invalid image format',
    'E1002': 'Image too large',
    'E1003': 'Image too small',
    'E1004': 'Corrupt image file',
    'E1005': 'Missing alpha channel',
    
    # 2xxx: Template Errors
    'E2001': 'Template not found',
    'E2002': 'Invalid template metadata',
    'E2003': 'Template image missing',
    'E2004': 'Template mask invalid',
    'E2005': 'Template category unknown',
    
    # 3xxx: Validation Errors
    'E3001': 'Invalid design file',
    'E3002': 'Design dimensions invalid',
    'E3003': 'Transform parameters invalid',
    'E3004': 'Export options invalid',
    
    # 4xxx: Processing Errors
    'E4001': 'Processing pipeline failed',
    'E4002': 'Displacement generation failed',
    'E4003': 'Color correction failed',
    'E4004': 'Compositing failed',
    
    # 5xxx: Export Errors
    'E5001': 'Export format unsupported',
    'E5002': 'Export resolution too high',
    'E5003': 'Export generation failed',
    'E5004': 'Export file too large',
    
    # 6xxx: AI Errors
    'E6001': 'AI model not loaded',
    'E6002': 'AI inference failed',
    'E6003': 'AI model not found',
}
```

### 14.3 Error Response Format

```python
class ErrorResponse(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    request_id: Optional[str] = None

def create_error_response(error: Exception, request_id: Optional[str] = None) -> ErrorResponse:
    if isinstance(error, CrevrError):
        code = get_error_code(error)
        message = str(error)
        details = get_error_details(error)
    else:
        code = 'E9999'
        message = 'Internal server error'
        details = None
    
    return ErrorResponse(
        code=code,
        message=message,
        details=details,
        request_id=request_id
    )
```

### 14.4 Error Handling Decorator

```python
from functools import wraps
import traceback

def handle_errors(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except CrevrError as e:
            logger.error(f"Expected error: {e}")
            return create_error_response(e)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            logger.error(traceback.format_exc())
            return create_error_response(
                ProcessingError(f"Unexpected error: {str(e)}")
            )
    return wrapper

# Usage:
@app.post("/api/designs/process")
@handle_errors
async def process_design(request: ProcessRequest):
    # ...
```

### 14.5 Validation Errors

```python
from pydantic import BaseModel, validator

class ProcessRequest(BaseModel):
    template_id: str
    design_file: str
    options: ProcessingOptions
    
    @validator('template_id')
    def validate_template_id(cls, v):
        if not v.startswith('tpl_'):
            raise ValidationError('Template ID must start with "tpl_"')
        return v
    
    @validator('design_file')
    def validate_design_file(cls, v):
        # Check if base64 or URL
        if v.startswith('data:image'):
            return v
        elif v.startswith('http'):
            return v
        else:
            raise ValidationError('Design file must be base64 or URL')
```

### 14.6 Recovery Strategies

| Error Type | Strategy |
|------------|----------|
| Image corruption | Try to recover, fallback to original |
| Template missing | Return 404, suggest alternatives |
| Processing timeout | Retry with lower quality |
| Export failure | Fallback to lower resolution |
| AI model error | Use deterministic fallback |
| Memory error | Process in chunks, free memory |

### 14.7 Logging Errors

```python
import logging
from datetime import datetime

class ErrorLogger:
    def __init__(self):
        self.logger = logging.getLogger('crevr.error')
        self.error_counts = {}
        
    def log_error(self, error: Exception, context: dict = None):
        error_type = type(error).__name__
        error_code = getattr(error, 'code', 'UNKNOWN')
        
        # Increment count
        key = f"{error_type}:{error_code}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
        
        # Log
        self.logger.error(
            f"Error: {error_type} ({error_code})",
            extra={
                'error': str(error),
                'context': context,
                'timestamp': datetime.utcnow().isoformat()
            }
        )
        
        # Alert if threshold exceeded
        if self.error_counts[key] > 10:
            self.send_alert(key, self.error_counts[key])
```

### 14.8 User-Friendly Errors

```python
def get_user_message(error_code: str) -> str:
    user_messages = {
        'E1001': "The image format isn't supported. Please use PNG, JPG, or WebP.",
        'E1002': "The image is too large. Please reduce the size to under 50MB.",
        'E1003': "The image is too small. Please use an image at least 300x300 pixels.",
        'E2001': "The template you're trying to use was not found. Please select another template.",
        'E3001': "The design file is invalid. Please upload a valid image file.",
        'E4001': "We couldn't generate the mockup. Please try again or use a different design.",
        'E5001': "The export format is not supported. Please choose PNG, JPG, or WebP.",
    }
    return user_messages.get(error_code, "An unexpected error occurred. Please try again.")
```

---

## 15. Performance Optimization

### 15.1 Image Processing Optimization

#### 15.1.1 NumPy Optimization

```python
# Use vectorized operations instead of loops
# ❌ Slow
for i in range(height):
    for j in range(width):
        result[i, j] = image[i, j] * 2.0

# ✅ Fast
result = image * 2.0

# Use view instead of copy when possible
# ❌ Copy
cropped = image[y1:y2, x1:x2].copy()

# ✅ View (no copy)
cropped = image[y1:y2, x1:x2]

# Use np.einsum for complex operations
# ❌ Slow
result = np.zeros((h, w, 3))
for i in range(h):
    for j in range(w):
        result[i, j] = np.dot(matrix, image[i, j])

# ✅ Fast
result = np.einsum('ij,klj->kli', matrix, image)
```

#### 15.1.2 OpenCV Optimization

```python
# Use appropriate interpolation
# For downscaling: cv2.INTER_AREA
resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)

# For upscaling: cv2.INTER_CUBIC or cv2.INTER_LANCZOS4
resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)

# Use UMat for GPU acceleration
gpu_image = cv2.UMat(image)
gpu_result = cv2.GaussianBlur(gpu_image, (5, 5), 0)
result = gpu_result.get()

# Use OpenMP
cv2.setNumThreads(4)
```

#### 15.1.3 Memory Management

```python
import gc

def process_large_image(image_path: str):
    # Load in chunks
    chunks = []
    for chunk in load_in_chunks(image_path):
        processed = process_chunk(chunk)
        chunks.append(processed)
        # Free memory
        del chunk
        gc.collect()
    
    return merge_chunks(chunks)

# Use memory mapping for large files
import mmap

def mmap_image(path: str):
    with open(path, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            data = np.frombuffer(mm, dtype=np.uint8)
            # Process data
```

### 15.2 Async Processing

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncProcessor:
    def __init__(self, max_workers=4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    async def process_async(self, image: np.ndarray) -> np.ndarray:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self.process_sync,
            image
        )
        return result
    
    def process_sync(self, image: np.ndarray) -> np.ndarray:
        # CPU-bound processing
        return image * 2
```

### 15.3 Caching Strategy

```python
# Multi-level caching
class CacheManager:
    def __init__(self):
        self.l1_cache = {}  # In-memory
        self.l2_cache = redis.Redis()  # Redis
        self.l3_cache = FileSystemCache()  # Disk
        
    def get(self, key: str):
        # Check L1
        if key in self.l1_cache:
            return self.l1_cache[key]
        
        # Check L2
        result = self.l2_cache.get(key)
        if result:
            self.l1_cache[key] = result
            return result
        
        # Check L3
        result = self.l3_cache.get(key)
        if result:
            self.l2_cache.set(key, result)
            self.l1_cache[key] = result
            return result
        
        return None
    
    def set(self, key: str, value, ttl: int = 3600):
        self.l1_cache[key] = value
        self.l2_cache.setex(key, ttl, value)
        self.l3_cache.set(key, value)

# Cache image transforms
@lru_cache(maxsize=100)
def get_perspective_transform(src_points, dst_points):
    return cv2.getPerspectiveTransform(src_points, dst_points)
```

### 15.4 Parallel Processing

```python
from multiprocessing import Pool, cpu_count

class BatchProcessor:
    def __init__(self):
        self.pool = Pool(processes=cpu_count())
        
    def process_batch(self, items: List[dict]) -> List[np.ndarray]:
        # Use multiprocessing for CPU-bound tasks
        results = self.pool.map(self.process_item, items)
        return results
    
    def process_item(self, item: dict) -> np.ndarray:
        # Process single item
        return item['image'] * 2

# GPU batch processing
import torch

def process_batch_gpu(images: np.ndarray) -> np.ndarray:
    # Move to GPU
    tensor = torch.from_numpy(images).cuda()
    
    # Process in parallel
    result = tensor * 2
    
    # Move back to CPU
    return result.cpu().numpy()
```

### 15.5 Streaming Processing

```python
def process_image_stream(input_path: str, output_path: str, chunk_size: int = 100):
    """Process large images in chunks."""
    # Read image
    image = cv2.imread(input_path)
    h, w = image.shape[:2]
    
    # Process in chunks
    with open(output_path, 'wb') as out:
        for y in range(0, h, chunk_size):
            chunk = image[y:y+chunk_size]
            processed = process_chunk(chunk)
            out.write(processed.tobytes())
```

### 15.6 Performance Metrics

```python
from time import time
from contextlib import contextmanager

@contextmanager
def measure_performance(name: str):
    start = time()
    yield
    duration = time() - start
    print(f"{name}: {duration:.3f}s")

# Usage:
with measure_performance("perspective_transform"):
    result = cv2.warpPerspective(image, H, (w, h))

# Performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}
        
    def record(self, name: str, duration: float):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(duration)
        
    def get_stats(self, name: str) -> dict:
        if name not in self.metrics:
            return {}
        data = self.metrics[name]
        return {
            'avg': sum(data) / len(data),
            'min': min(data),
            'max': max(data),
            'count': len(data)
        }
```

### 15.7 SIMD Optimization

```python
# Use NumPy's SIMD optimizations
import numpy as np

# NumPy automatically uses SIMD where possible
def process_fast(image):
    # These operations use SIMD
    result = image * 2
    result = np.clip(result, 0, 255)
    result = result.astype(np.uint8)
    return result

# For more control, use OpenCV
def process_cv2(image):
    # OpenCV uses SIMD optimizations
    result = cv2.addWeighted(image, 2, image, 0, 0)
    return result
```

---

## 16. Security Architecture

### 16.1 Upload Security

```python
from pathlib import Path
import magic  # python-magic

class UploadValidator:
    def __init__(self):
        self.max_size = 50 * 1024 * 1024  # 50MB
        self.allowed_types = [
            'image/jpeg',
            'image/png',
            'image/webp',
            'image/gif',
            'image/bmp',
            'image/tiff'
        ]
        
    def validate(self, file_data: bytes, filename: str) -> bool:
        # Check size
        if len(file_data) > self.max_size:
            raise ValidationError('File too large')
        
        # Check MIME type
        mime = magic.from_buffer(file_data, mime=True)
        if mime not in self.allowed_types:
            raise ValidationError(f'Invalid file type: {mime}')
        
        # Check for malicious content
        self.check_for_malicious(file_data)
        
        return True
    
    def check_for_malicious(self, file_data: bytes):
        # Check for executable code
        # Check for embedded scripts
        # Check for zip bombs
        # Check for known malware signatures
        pass
```

### 16.2 Path Traversal Prevention

```python
from pathlib import Path
import os

class SecurePathManager:
    def __init__(self, base_path: Path):
        self.base_path = base_path.resolve()
        
    def get_path(self, user_path: str) -> Path:
        # Resolve user path
        path = (self.base_path / user_path).resolve()
        
        # Check if inside base path
        if not str(path).startswith(str(self.base_path)):
            raise SecurityError('Path traversal detected')
        
        return path
```

### 16.3 Image Sanitization

```python
def sanitize_image(image_data: bytes) -> bytes:
    """Remove potentially harmful data from images."""
    # Use PIL to re-encode image
    from PIL import Image
    import io
    
    try:
        # Open image
        img = Image.open(io.BytesIO(image_data))
        
        # Remove EXIF data
        img.info.clear()
        
        # Re-encode with safety
        output = io.BytesIO()
        img.save(output, format=img.format, quality=95)
        
        return output.getvalue()
    except Exception as e:
        raise SecurityError(f'Image sanitization failed: {e}')
```

### 16.4 Rate Limiting

```python
from functools import wraps
import time

class RateLimiter:
    def __init__(self, max_requests: int, window: int):
        self.max_requests = max_requests
        self.window = window
        self.requests = {}
        
    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        
        # Clean old requests
        if client_id in self.requests:
            self.requests[client_id] = [
                t for t in self.requests[client_id]
                if now - t < self.window
            ]
        else:
            self.requests[client_id] = []
        
        # Check limit
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        # Add request
        self.requests[client_id].append(now)
        return True

def rate_limit(max_requests: int = 100, window: int = 60):
    limiter = RateLimiter(max_requests, window)
    
    def decorator(func):
        @wraps(func)
        async def wrapper(request, *args, **kwargs):
            client_id = request.client.host
            if not limiter.is_allowed(client_id):
                raise RateLimitError('Rate limit exceeded')
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator
```

### 16.5 Input Validation

```python
from pydantic import BaseModel, validator
import re

class DesignUpload(BaseModel):
    name: str
    template_id: str
    file_data: str  # base64
    options: dict
    
    @validator('name')
    def validate_name(cls, v):
        if len(v) > 100:
            raise ValidationError('Name too long')
        if not re.match(r'^[\w\s\-.,]+$', v):
            raise ValidationError('Invalid name format')
        return v
    
    @validator('template_id')
    def validate_template_id(cls, v):
        if not re.match(r'^tpl_[a-zA-Z0-9]+$', v):
            raise ValidationError('Invalid template ID')
        return v
    
    @validator('file_data')
    def validate_file_data(cls, v):
        # Validate base64
        try:
            data = base64.b64decode(v)
        except:
            raise ValidationError('Invalid base64 data')
        return v
```

### 16.6 Output Sanitization

```python
def sanitize_filename(filename: str) -> str:
    """Remove dangerous characters from filenames."""
    # Allow only alphanumeric, dash, underscore, dot
    sanitized = re.sub(r'[^a-zA-Z0-9\-_.]', '_', filename)
    return sanitized

def sanitize_url(url: str) -> str:
    """Validate and sanitize URLs."""
    # Check protocol
    if not url.startswith(('http://', 'https://')):
        raise ValidationError('Invalid URL protocol')
    
    # Check for malicious content
    # ...
    
    return url
```

### 16.7 Authentication

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta

class AuthManager:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.security = HTTPBearer()
        
    def create_token(self, user_id: str) -> str:
        payload = {
            'sub': user_id,
            'exp': datetime.utcnow() + timedelta(days=7)
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_token(self, token: str) -> dict:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail='Token expired'
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail='Invalid token'
            )

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    auth_manager = AuthManager(settings.SECRET_KEY)
    payload = auth_manager.verify_token(credentials.credentials)
    return payload
```

---

## 17. Testing Strategy

### 17.1 Unit Testing

```python
import unittest
import numpy as np
import cv2

class TestDisplacementEngine(unittest.TestCase):
    def setUp(self):
        self.engine = DisplacementEngine()
        self.image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.displacement = np.zeros((100, 100), dtype=np.uint8)
        
    def test_apply_displacement(self):
        # Create a simple design
        design = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        # Apply displacement
        result = self.engine.apply(design, self.displacement)
        
        # Should be same as input
        np.testing.assert_array_equal(result, design)
        
    def test_apply_displacement_with_scale(self):
        design = np.ones((100, 100, 3), dtype=np.uint8) * 255
        displacement = np.ones((100, 100), dtype=np.uint8) * 50
        
        result = self.engine.apply(design, displacement, scale=0.1)
        
        # Should be slightly different
        self.assertFalse(np.array_equal(result, design))
        
    def test_displacement_handles_edges(self):
        # Test that displacement doesn't go out of bounds
        design = np.ones((10, 10, 3), dtype=np.uint8) * 255
        displacement = np.ones((10, 10), dtype=np.uint8) * 255
        
        result = self.engine.apply(design, displacement, scale=10)
        
        self.assertEqual(result.shape, design.shape)

class TestColorEngine(unittest.TestCase):
    def setUp(self):
        self.engine = ColorEngine()
        
    def test_rgb_to_lab(self):
        rgb = np.array([[[255, 0, 0]]], dtype=np.uint8)
        lab = self.engine.rgb_to_lab(rgb)
        # L should be high, a should be positive
        self.assertGreater(lab[0, 0, 0], 50)
        self.assertGreater(lab[0, 0, 1], 0)
        
    def test_color_correction(self):
        design = np.ones((100, 100, 3), dtype=np.uint8) * 200
        template = np.ones((100, 100, 3), dtype=np.uint8) * 100
        
        corrected = self.engine.correct(design, template)
        
        # Should be closer to template
        self.assertLess(np.mean(corrected), 200)
        self.assertGreater(np.mean(corrected), 50)
```

### 17.2 Integration Testing

```python
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

class TestAPI:
    def test_get_templates(self):
        response = client.get("/api/v1/templates")
        assert response.status_code == 200
        assert "templates" in response.json()
        
    def test_upload_template(self):
        with open("test_data/template.jpg", "rb") as f:
            files = {"file": ("template.jpg", f, "image/jpeg")}
            response = client.post("/api/v1/templates/upload", files=files)
            assert response.status_code == 200
            assert "id" in response.json()
            
    def test_process_design(self):
        data = {
            "template_id": "tpl_test",
            "design_file": "base64_encoded_image",
            "options": {"format": "png"}
        }
        response = client.post("/api/v1/designs/process", json=data)
        assert response.status_code == 200
        assert "job_id" in response.json()
        
    def test_invalid_image(self):
        data = {
            "template_id": "tpl_test",
            "design_file": "invalid_data",
            "options": {"format": "png"}
        }
        response = client.post("/api/v1/designs/process", json=data)
        assert response.status_code == 400
        assert "error" in response.json()
```

### 17.3 Image Comparison Testing

```python
class ImageComparator:
    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold
        
    def compare(self, actual: np.ndarray, expected: np.ndarray) -> bool:
        """Compare two images with tolerance."""
        if actual.shape != expected.shape:
            return False
        
        # Calculate MSE
        mse = np.mean((actual.astype(float) - expected.astype(float)) ** 2)
        
        # Normalize
        max_mse = 255 ** 2
        normalized_mse = mse / max_mse
        
        return normalized_mse < self.threshold
    
    def compare_with_ssim(self, actual: np.ndarray, expected: np.ndarray) -> float:
        """Compare using SSIM."""
        from skimage.metrics import structural_similarity
        return structural_similarity(actual, expected, multichannel=True)
    
    def compare_histogram(self, actual: np.ndarray, expected: np.ndarray) -> float:
        """Compare histograms."""
        hist_actual = cv2.calcHist([actual], [0], None, [256], [0, 256])
        hist_expected = cv2.calcHist([expected], [0], None, [256], [0, 256])
        return cv2.compareHist(hist_actual, hist_expected, cv2.HISTCMP_CORREL)
```

### 17.4 Regression Testing

```python
class RegressionTest:
    def __init__(self, golden_dir: Path):
        self.golden_dir = golden_dir
        self.comparator = ImageComparator()
        
    def run(self) -> bool:
        """Run all regression tests."""
        passed = True
        
        for test_file in self.golden_dir.glob("*.json"):
            with open(test_file) as f:
                test_case = json.load(f)
                
            # Run test
            result = self.run_test_case(test_case)
            
            if not result:
                passed = False
                print(f"Regression test failed: {test_file}")
                
        return passed
    
    def run_test_case(self, test_case: dict) -> bool:
        # Load input
        input_image = cv2.imread(test_case['input'])
        
        # Process
        result = self.process(input_image)
        
        # Load expected
        expected = cv2.imread(test_case['expected'])
        
        # Compare
        return self.comparator.compare(result, expected)
```

### 17.5 Performance Testing

```python
class PerformanceTest:
    def __init__(self):
        self.metrics = {}
        
    def test_processing_speed(self, processor, test_count: int = 100):
        """Test processing speed."""
        image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        times = []
        
        for _ in range(test_count):
            start = time.time()
            processor.process(image)
            times.append(time.time() - start)
            
        avg_time = sum(times) / len(times)
        print(f"Average processing time: {avg_time:.3f}s")
        
        assert avg_time < 1.0  # Should process in under 1 second
        
    def test_memory_usage(self, processor):
        """Test memory usage."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss
        
        image = np.random.randint(0, 255, (4000, 4000, 3), dtype=np.uint8)
        result = processor.process(image)
        
        mem_after = process.memory_info().rss
        mem_used = (mem_after - mem_before) / (1024 * 1024)
        
        print(f"Memory used: {mem_used:.1f} MB")
        assert mem_used < 500  # Should use under 500MB
```

### 17.6 Coverage Requirements

```yaml
coverage:
  overall: 85%
  api: 95%
  core: 90%
  ai: 80%
  utils: 95%
  storage: 85%
```

---

## 18. GitHub Research & Architectural Patterns

### 18.1 Open Source Mockup Projects

#### 18.1.1 automated_mockups (Python)
**Repository:** https://github.com/topics/mockup-generator

**Key Features:**
- Python-based PSD mockup automation
- Overlays designs onto apparel templates
- Bulk processing capability
- Uses Pillow for image manipulation

**Architectural Insights:**
- Template-based approach with configurable regions
- Batch processing for multiple designs
- Simple compositing pipeline
- Focus on reproducibility

**What We Can Learn:**
- Template configuration via JSON
- Batch processing architecture
- Simple compositing approach

#### 18.1.2 Mockingbird
**Repository:** https://github.com/topics/mockup-generator

**Key Features:**
- Bulk embeds UI screenshots into device frames
- Supports Apple devices
- Web-based interface

**Architectural Insights:**
- API-first design
- Device frame templates
- Image composition pipeline

**What We Can Learn:**
- API design for mockup generation
- Device template management
- Web-based generation

#### 18.1.3 Presenta
**Repository:** https://github.com/topics/mockup-generator

**Key Features:**
- Creates animated mockup videos
- Uses device frames
- JavaScript + FastAPI backend

**Architectural Insights:**
- Real-time processing in browser
- Video generation pipeline
- Frame-based animation

**What We Can Learn:**
- In-browser processing
- Animation pipeline design
- API integration patterns

### 18.2 Image Processing Libraries

#### 18.2.1 Pillow (PIL Fork)
**Repository:** https://github.com/python-pillow/Pillow

**Key Features:**
- Python imaging library
- Extensive format support
- Image manipulation operations
- Good performance

**Use Cases:**
- Basic image operations
- Format conversions
- Simple compositing

**Alternative Consideration:** OpenCV provides better performance for complex operations.

#### 18.2.2 OpenCV
**Repository:** https://github.com/opencv/opencv

**Key Features:**
- Computer vision library
- Real-time processing
- GPU acceleration
- Extensive algorithms

**Use Cases:**
- Image transformations
- Feature detection
- Perspective transform
- Matrix operations

**Alternative Consideration:** Pillow is simpler but OpenCV is more powerful.

#### 18.2.3 ImageMagick
**Repository:** https://github.com/ImageMagick/ImageMagick

**Key Features:**
- Command-line image processing
- Format conversion
- Batch processing
- Scriptable

**Use Cases:**
- Batch operations
- Format conversion
- Simple compositions

**Alternative Consideration:** Python libraries offer better integration.

### 18.3 Architectural Patterns

#### 18.3.1 Pipeline Pattern

```python
class Pipeline:
    def __init__(self):
        self.steps = []
        
    def add_step(self, step):
        self.steps.append(step)
        
    def process(self, data):
        for step in self.steps:
            data = step(data)
        return data

# Usage
pipeline = Pipeline()
pipeline.add_step(preprocess)
pipeline.add_step(transform)
pipeline.add_step(composite)
result = pipeline.process(design)
```

**Why This Pattern:**
- Clear separation of concerns
- Easy to add/remove steps
- Testable in isolation
- Reusable components

#### 18.3.2 Factory Pattern

```python
class TemplateFactory:
    def create(self, template_type: str, config: dict) -> Template:
        if template_type == 'apparel':
            return ApparelTemplate(config)
        elif template_type == 'device':
            return DeviceTemplate(config)
        elif template_type == 'packaging':
            return PackagingTemplate(config)
        else:
            raise ValueError(f'Unknown template type: {template_type}')
```

**Why This Pattern:**
- Encapsulates creation logic
- Supports different template types
- Easy to add new types

#### 18.3.3 Strategy Pattern

```python
class TransformStrategy:
    def transform(self, image: np.ndarray, params: dict) -> np.ndarray:
        pass

class PerspectiveStrategy(TransformStrategy):
    def transform(self, image: np.ndarray, params: dict) -> np.ndarray:
        H = cv2.getPerspectiveTransform(
            np.array(params['src']),
            np.array(params['dst'])
        )
        return cv2.warpPerspective(image, H, params['size'])

class AffineStrategy(TransformStrategy):
    def transform(self, image: np.ndarray, params: dict) -> np.ndarray:
        M = cv2.getAffineTransform(
            np.array(params['src']),
            np.array(params['dst'])
        )
        return cv2.warpAffine(image, M, params['size'])
```

**Why This Pattern:**
- Swap algorithms at runtime
- Clean extension points
- Separation of concerns

#### 18.3.4 Observer Pattern

```python
class ProgressObserver:
    def update(self, progress: float, message: str):
        pass

class WebSocketObserver(ProgressObserver):
    def update(self, progress: float, message: str):
        # Send progress to WebSocket
        websocket.send({
            'progress': progress,
            'message': message
        })

class ProgressSubject:
    def __init__(self):
        self.observers = []
        
    def attach(self, observer: ProgressObserver):
        self.observers.append(observer)
        
    def notify(self, progress: float, message: str):
        for observer in self.observers:
            observer.update(progress, message)
```

**Why This Pattern:**
- Real-time progress updates
- Multiple listeners
- Decoupled progress reporting

### 18.4 Design Patterns Applied

| Pattern | Use Case | Benefit |
|---------|----------|---------|
| Pipeline | Image processing chain | Modular, testable |
| Factory | Template creation | Flexible, extensible |
| Strategy | Transform algorithms | Swappable algorithms |
| Observer | Progress reporting | Real-time feedback |
| Singleton | Cache manager | Single cache instance |
| Decorator | Image enhancement | Add features without modifying |

### 18.5 Anti-Patterns to Avoid

1. **God Object:** A single class that does everything
2. **Spaghetti Code:** Tangled, unstructured code
3. **Copy-Paste:** Duplicated code across modules
4. **Premature Optimization:** Over-optimizing before needed
5. **Big Ball of Mud:** No clear architecture

### 18.6 Code Organization Best Practices

1. **Separation of Concerns:** Separate processing, API, storage
2. **Dependency Injection:** Inject dependencies for testability
3. **Single Responsibility:** Each module does one thing well
4. **Open/Closed:** Open for extension, closed for modification
5. **Loose Coupling:** Minimal dependencies between modules

---

## 19. Roadmap & Phased Development

### 19.1 MVP (Phase 1) - Foundation

**Duration:** 2-3 months

**Core Features:**
- [x] Basic image upload
- [x] Template selection
- [x] Perspective transform
- [x] Simple compositing
- [x] PNG/JPEG export
- [x] Basic template support
  - T-Shirt (front)
  - Laptop screen
  - Mobile phone

**Technical Goals:**
- [ ] Working prototype
- [ ] Local deployment
- [ ] 100 template support
- [ ] < 5s processing time

**Deliverables:**
- Web application
- Basic API
- Documentation
- 100+ templates
- Unit tests

**Success Metrics:**
- 1000 users
- 4.0+ user rating
- < 5% error rate

### 19.2 Phase 2 - Enhanced Features

**Duration:** 2-3 months

**New Features:**
- [ ] Batch processing
- [ ] WebP/AVIF export
- [ ] Color correction
- [ ] Shadow simulation
- [ ] Template editor
- [ ] AI auto-detection
- [ ] Additional templates
  - Hoodie
  - Oversized T-Shirt
  - Desktop monitor
  - Box packaging

**Technical Goals:**
- [ ] AI pipeline integration
- [ ] GPU acceleration
- [ ] Template analyzer
- [ ] 500+ templates

**Deliverables:**
- Template editor
- AI features
- Batch processing
- Template marketplace

**Success Metrics:**
- 5000 users
- 4.5+ user rating
- < 3% error rate

### 19.3 Phase 3 - Advanced Features

**Duration:** 2-3 months

**New Features:**
- [ ] Displacement maps
- [ ] Realistic folds
- [ ] Texture transfer
- [ ] Highlight preservation
- [ ] Video mockups
- [ ] 3D mockups
- [ ] Template marketplace
- [ ] User accounts

**Technical Goals:**
- [ ] Advanced compositing
- [ ] 3D rendering
- [ ] Video generation
- [ ] User management

**Deliverables:**
- Advanced compositing
- 3D support
- Video support
- Marketplace

**Success Metrics:**
- 10,000 users
- 4.7+ user rating
- < 1% error rate

### 19.4 Phase 4 - Enterprise Features

**Duration:** 2-3 months

**New Features:**
- [ ] White-label solution
- [ ] API platform
- [ ] Custom integrations
- [ ] Advanced analytics
- [ ] Team collaboration
- [ ] Version control
- [ ] Priority support

**Technical Goals:**
- [ ] Enterprise architecture
- [ ] High availability
- [ ] Horizontal scaling
- [ ] Disaster recovery

**Deliverables:**
- Enterprise API
- White-label licensing
- Integration SDK
- Admin dashboard

**Success Metrics:**
- 25,000 users
- 100 enterprise customers
- 4.9+ user rating

### 19.5 Phase 5 - Ecosystem

**Duration:** Ongoing

**New Features:**
- [ ] Plugin system
- [ ] Community templates
- [ ] Design marketplace
- [ ] AI model marketplace
- [ ] Developer SDK
- [ ] Cloud sync
- [ ] Mobile app

**Technical Goals:**
- [ ] Plugin architecture
- [ ] Community features
- [ ] Mobile deployment
- [ ] Cloud infrastructure

**Deliverables:**
- Plugin API
- Community platform
- Mobile apps
- Cloud service

**Success Metrics:**
- 50,000+ users
- 1000+ community templates
- 100+ plugins

### 19.6 Timeline

```
Phase 1: Jan 2026 - Mar 2026
Phase 2: Apr 2026 - Jun 2026
Phase 3: Jul 2026 - Sep 2026
Phase 4: Oct 2026 - Dec 2026
Phase 5: 2027+
```

### 19.7 Resource Requirements

| Phase | Developers | Designers | QA | Timeline |
|-------|------------|-----------|-----|----------|
| MVP | 2 | 1 | 1 | 3 months |
| Phase 2 | 3 | 2 | 2 | 3 months |
| Phase 3 | 4 | 2 | 2 | 3 months |
| Phase 4 | 4 | 2 | 3 | 3 months |
| Phase 5 | 5 | 3 | 3 | Ongoing |

### 19.8 Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| AI model integration | Medium | High | Start with simple models, iterate |
| Performance issues | Medium | High | Optimize early, monitor |
| Template quality | Low | Medium | QC process for templates |
| Competition | High | Medium | Focus on local-first, free |
| User adoption | Medium | High | Marketing, community |

### 19.9 Success Metrics

**Technical:**
- Processing time < 5s
- Uptime > 99.9%
- Error rate < 1%
- Response time < 200ms

**Business:**
- 10,000+ MAU
- 4.5+ rating
- 50% retention
- 100+ templates/week

**Community:**
- 1000+ GitHub stars
- 50+ contributors
- 100+ issues resolved/week

---

## 20. Deployment & Operations

### 20.1 Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Load Balancer                          │
│                     (Nginx/HAProxy)                        │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│  Web Server    │  │  Web Server    │  │  Web Server    │
│  (React)       │  │  (React)       │  │  (React)       │
└────────────────┘  └────────────────┘  └────────────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     API Gateway                            │
│                   (FastAPI + Uvicorn)                      │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│  API Server    │  │  API Server    │  │  API Server    │
│  (FastAPI)     │  │  (FastAPI)     │  │  (FastAPI)     │
└────────────────┘  └────────────────┘  └────────────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Celery Worker Pool                      │
│              (Image Processing, AI, Export)                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  PostgreSQL  │  │    Redis     │  │   Storage    │   │
│  │  (Metadata)  │  │  (Cache)     │  │   (S3/NFS)   │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 20.2 Infrastructure Requirements

**Minimum Hardware:**
- CPU: 4 cores
- RAM: 8GB
- Storage: 50GB
- Network: 100Mbps

**Recommended Hardware:**
- CPU: 8+ cores
- RAM: 32GB+
- Storage: 1TB+ SSD
- Network: 1Gbps
- GPU: NVIDIA RTX 3060+

**Cloud Options:**
- AWS: EC2 + S3 + RDS + ElastiCache
- GCP: Compute Engine + Cloud Storage + Cloud SQL + Memorystore
- Azure: VMs + Blob Storage + SQL Database + Redis Cache

### 20.3 Deployment Process

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build Frontend
        run: |
          cd frontend
          npm ci
          npm run build
          
      - name: Build Backend
        run: |
          cd backend
          docker build -t crevr-backend .
          
      - name: Deploy to Server
        run: |
          scp -r frontend/dist user@server:/var/www/crevr
          ssh user@server "docker pull crevr-backend && docker-compose up -d"
```

### 20.4 Monitoring

**Metrics to Monitor:**

```python
class Metrics:
    # System metrics
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: float
    
    # Application metrics
    request_count: int
    error_count: int
    response_time: float
    processing_time: float
    
    # Business metrics
    users_active: int
    mockups_generated: int
    exports_created: int

# Prometheus integration
from prometheus_client import Counter, Histogram, Gauge

requests = Counter('http_requests_total', 'Total HTTP requests')
errors = Counter('http_errors_total', 'Total HTTP errors')
response_time = Histogram('http_response_time_seconds', 'HTTP response time')
active_users = Gauge('active_users', 'Active users')

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    
    requests.inc()
    response_time.observe(duration)
    
    if response.status_code >= 400:
        errors.inc()
    
    return response
```

### 20.5 Logging

```python
import structlog

logger = structlog.get_logger()

# Structured logging
logger.info("processing_started", 
            template_id="tpl_123",
            design_size=1024 * 1024)

logger.info("processing_completed",
            template_id="tpl_123",
            duration=1.5,
            output_size=2 * 1024 * 1024)

# Error logging with context
try:
    process_design()
except Exception as e:
    logger.error("processing_failed",
                error=str(e),
                traceback=traceback.format_exc())
```

### 20.6 Backup and Disaster Recovery

```python
class BackupManager:
    def __init__(self, backup_dir: Path):
        self.backup_dir = backup_dir
        
    def backup_database(self):
        """Backup PostgreSQL database."""
        import subprocess
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"db_backup_{timestamp}.sql"
        
        subprocess.run([
            "pg_dump",
            "-h", settings.DB_HOST,
            "-U", settings.DB_USER,
            "-d", settings.DB_NAME,
            "-f", str(backup_file)
        ])
        
    def backup_storage(self):
        """Backup file storage."""
        import shutil
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"storage_backup_{timestamp}.tar.gz"
        
        shutil.make_archive(
            str(backup_file).replace('.tar.gz', ''),
            'gztar',
            settings.STORAGE_PATH
        )
```

**Recovery Time Objectives:**
- RPO (Recovery Point Objective): 1 hour
- RTO (Recovery Time Objective): 4 hours

### 20.7 Scaling Strategy

**Horizontal Scaling:**
- API servers: Add more instances
- Worker nodes: Add more workers
- Database: Read replicas
- Cache: Redis cluster

**Vertical Scaling:**
- Add more CPU cores
- Add more RAM
- Add GPU resources
- Upgrade storage

**Auto-scaling Triggers:**
- CPU > 70%
- Memory > 80%
- Queue length > 1000
- Response time > 500ms

---

## 21. Monetization Strategy

### 21.1 Free Tier

**Features:**
- Unlimited templates access
- Basic mockup generation
- PNG export
- 720p resolution
- Watermarked output
- Community support

**Purpose:**
- User acquisition
- Showcase capabilities
- Build community

### 21.2 Pro Tier ($9.99/month)

**Features:**
- All free features
- HD resolution (2K)
- WebP/AVIF export
- No watermarks
- Batch processing (10 at a time)
- Priority support
- Commercial usage rights

### 21.3 Business Tier ($29.99/month)

**Features:**
- All Pro features
- 4K resolution
- Unlimited batch processing
- API access
- Custom templates
- White-label export
- Direct support
- Team management

### 21.4 Enterprise (Custom Pricing)

**Features:**
- All Business features
- On-premise deployment
- Custom branding
- SLA guarantee
- Training and onboarding
- Priority bug fixes
- Dedicated support

### 21.5 Alternative Pricing Models

**Pay-as-you-go:**
- $0.50 per export
- $0.10 per preview
- $1.00 per batch

**Token System:**
- 10 tokens = $5
- 100 tokens = $40
- 1000 tokens = $300

**Lifetime License:**
- Pro: $99 (one-time)
- Business: $249 (one-time)
- Enterprise: Custom

### 21.6 Revenue Projections

| Tier | Price | Users | Revenue |
|------|-------|-------|---------|
| Free | $0 | 10,000 | $0 |
| Pro | $9.99 | 1,000 | $9,990/mo |
| Business | $29.99 | 200 | $5,998/mo |
| Enterprise | $499 | 20 | $9,980/mo |
| **Total** | | | **$25,968/mo** |

### 21.7 Billing Integration

```python
class BillingService:
    def __init__(self):
        self.stripe = stripe.Stripe(settings.STRIPE_KEY)
        
    def create_customer(self, email: str) -> str:
        customer = self.stripe.Customer.create(email=email)
        return customer.id
        
    def create_subscription(self, customer_id: str, price_id: str) -> dict:
        subscription = self.stripe.Subscription.create(
            customer=customer_id,
            items=[{"price": price_id}],
            payment_behavior="default_incomplete",
            expand=["latest_invoice.payment_intent"]
        )
        return subscription
        
    def cancel_subscription(self, subscription_id: str):
        self.stripe.Subscription.delete(subscription_id)
        
    def webhook_handler(self, payload: dict):
        event = self.stripe.Event.construct_from(payload, settings.STRIPE_KEY)
        
        if event.type == 'invoice.paid':
            self.handle_payment_success(event.data.object)
        elif event.type == 'invoice.payment_failed':
            self.handle_payment_failure(event.data.object)
```

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| Alpha Channel | Transparency channel in an image |
| API | Application Programming Interface |
| BGR | Blue-Green-Red color order (OpenCV default) |
| CMYK | Cyan-Magenta-Yellow-Key (Black) color model |
| Compositing | Combining multiple images into one |
| Contour | Boundary of a shape in an image |
| DPI | Dots Per Inch (print resolution) |
| FastAPI | Modern Python web framework |
| GPU | Graphics Processing Unit |
| Homography | Transformation between two planes |
| HTTP | Hypertext Transfer Protocol |
| ICC | International Color Consortium (color profiles) |
| JPEG | Joint Photographic Experts Group (image format) |
| LAB | Lightness-A-B color space |
| Mask | Binary image used to select regions |
| MVP | Minimum Viable Product |
| OpenCV | Open Source Computer Vision Library |
| PNG | Portable Network Graphics (image format) |
| PSD | Photoshop Document (Adobe format) |
| RGB | Red-Green-Blue color model |
| S3 | Simple Storage Service (AWS) |
| SIMD | Single Instruction, Multiple Data |
| SQLite | Lightweight SQL database |
| SVG | Scalable Vector Graphics |
| UI | User Interface |
| UX | User Experience |
| VRAM | Video Random Access Memory |
| WebP | Modern web image format |
| YOLO | You Only Look Once (object detection) |

---

## Appendix B: References

### Academic Papers

1. Lucas, B. D., & Kanade, T. (1981). An iterative image registration technique with an application to stereo vision.

2. Harris, C., & Stephens, M. (1988). A combined corner and edge detector.

3. Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints.

4. Viola, P., & Jones, M. J. (2004). Robust real-time face detection.

5. Redmon, J., et al. (2016). You Only Look Once: Unified, Real-Time Object Detection.

### Documentation

1. OpenCV Documentation: https://docs.opencv.org
2. NumPy Documentation: https://numpy.org/doc
3. Pillow Documentation: https://pillow.readthedocs.io
4. FastAPI Documentation: https://fastapi.tiangolo.com
5. React Documentation: https://reactjs.org/docs

### Tools and Libraries

1. OpenCV: https://opencv.org
2. NumPy: https://numpy.org
3. Pillow: https://python-pillow.org
4. FastAPI: https://fastapi.tiangolo.com
5. React: https://reactjs.org
6. Fabric.js: http://fabricjs.com
7. Celery: https://docs.celeryproject.org
8. Redis: https://redis.io
9. SQLite: https://sqlite.org

---

## Appendix C: Contributing Guidelines

### Code of Conduct

1. Be respectful
2. Be inclusive
3. Be collaborative
4. Be constructive

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Write tests
4. Submit pull request
5. Code review
6. Merge

### Coding Standards

**Python:**
- PEP 8 compliance
- Type hints
- Docstrings
- 88 character line limit

**TypeScript:**
- ESLint
- Prettier
- TypeScript strict mode
- 100 character line limit

---

## Appendix D: License

**MIT License**

Copyright (c) 2026 Crevr Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

*This document represents the complete Product Requirements Document for the Crevr Mockup Generator project. It serves as the single source of truth for all development efforts and should be consulted for any implementation questions.*