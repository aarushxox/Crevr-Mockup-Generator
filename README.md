# Crevr Mockup Generator

Crevr is a local-first, deterministic **image compositing engine** for high-fidelity product mockups. It maps and warps custom designs onto flat or curved product surfaces, applies pre-calculated fold displacement fields, matches environmental lighting tones in LAB color space, and composites them with feathered edges to produce photorealistic results instantly on your local computer.

Explicitly, this is **NOT** a generative AI tool; it is a fast, free, and highly accurate computer vision pipeline powered by **OpenCV** and **NumPy**.

---

## 🚀 Quick Start Guide (How to Run Locally)

Follow these steps sequentially to set up and run Crevr on your local machine.

### 1. Prerequisites
Make sure you have the following installed:
* **Python 3.10+** (Python 3.12 recommended)
* **Node.js 18+** (with npm)

---

### 2. Backend Engine Setup (Python FastAPI)

1. Open your terminal and navigate to the project directory:
   ```bash
   cd Crevr-Mockup-Generator
   ```

2. (Optional but recommended) Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install the required Python packages:
   ```bash
   pip install opencv-python-headless numpy pillow fastapi uvicorn pydantic python-multipart httpx
   ```

4. Run the **automatic template pre-ingestion script**. This parses the blank images in `assets/` and auto-generates masks, displacement maps, lighting maps, and coordinate files under `templates/`:
   ```bash
   PYTHONPATH=. python3 scripts/pre_ingest.py
   ```

5. Start the Python FastAPI engine on port `8001`:
   ```bash
   python3 -m uvicorn engine.api.main:app --port 8001
   ```

---

### 3. API Gateway & Frontend Setup (Node.js Express)

1. Open a new terminal window, navigate back to the root of the project, and go to the `gateway/` folder:
   ```bash
   cd Crevr-Mockup-Generator/gateway
   ```

2. Install the gateway Node dependencies:
   ```bash
   npm install
   ```

3. Start the Express API Gateway on port `8000`:
   ```bash
   PORT=8000 node server.js
   ```

---

### 4. Access the Application 🎉

Once both services are running:
* **Frontend Web Dashboard:** Open your browser and navigate to **[http://localhost:8000](http://localhost:8000)**.
* You can select a template (Classic Flat Lay T-shirt, Macbook Pro, or Studio Model Portrait), upload any design image, interactively translate/scale/rotate the artwork on the Fabric.js canvas viewport, and render a photo-realistic high-resolution mockup with realistic fold mappings!

---

## 🧪 Running the Tests

To verify that the entire pipeline is working properly on your system, you can run the built-in test suites.

### Run Pipeline Unit Tests
Tests the individual OpenCV perspective warp, displacement shifting, multiply/screen blending, and feathering logic:
```bash
python3 -m unittest engine/tests/test_pipeline.py
```

### Run API Integration Tests
Performs a full API round-trip integration test (upload design -> auto remove background -> trigger high-fidelity render -> download output):
```bash
python3 -m unittest engine/tests/test_api.py
```

---

## 📂 Project Architecture

```
Crevr-Mockup-Generator/
  assets/                     # Raw source photos (pre-ingestion)
  templates/                  # Ingested, pre-calculated templates (masks, maps, metadata)
  frontend/
    index.html                # Highly polished, productivity-first React & Fabric.js SPA
  gateway/
    server.js                 # Node.js gateway (handles uploads, checks magic bytes, proxies API)
  engine/                     # Python FastAPI compositing engine
    api/
      main.py                 # FastAPI endpoints & SQLite DB initializer
    pipeline/
      warp.py                 # Perspective/homography warping
      displacement.py         # Per-pixel offset remap (folds/wrinkles)
      blend.py                # LAB space tone matching & multiply/screen blends
      mask.py                 # Mask edge feathering & compositing
      ingest.py               # Automated CV template analyzer
      render.py               # Orchestrator of rendering stages
    tests/                    # Programmatic testing suite
  data/                       # Local database, uploads, and export directories (ignored by git)
```
