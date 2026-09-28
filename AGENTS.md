# Crevr Mockup Generator — Base44 Dev Notes

## Architecture
Three-tier local-first app (no external services or credentials):
- **engine** (`engine/`): Python FastAPI on port 8001. OpenCV/NumPy image compositing pipeline. Uses SQLite (`data/crevr.db`) for render history. Templates are pre-ingested into `templates/`.
- **gateway** (`gateway/`): Node.js Express on port 8000. Serves `frontend/index.html` as static files and proxies all `/api/*` requests to the engine via `ENGINE_URL`.
- **frontend** (`frontend/`): Single `index.html` using React, Babel, Tailwind, and Fabric.js all via CDN (no build step).

## Running
`docker compose -f docker-compose.base44.yml up -d` — gateway is exposed on host port 3000.

## Key Details
- The gateway's `forwardToEngine` parses `ENGINE_URL` (env var) to reach the engine container. Default `http://127.0.0.1:8001` works for local non-Docker dev; compose sets `ENGINE_URL=http://engine:8001`.
- Templates list endpoint is `POST /api/templates` (not GET) — the frontend explicitly uses `method: 'POST'`.
- `templates/` directory ships pre-ingested (base.png, mask.png, displacement.png, lighting.png, metadata.json per template). Re-run `PYTHONPATH=. python3 scripts/pre_ingest.py` only if `assets/` source images change.
- The engine creates `data/designs/`, `data/exports/`, and `data/crevr.db` at startup — all gitignored.
- opencv-python-headless requires system libs: `libglib2.0-0 libsm6 libxext6 libxrender1 libgl1` (installed in compose).
- Engine uses `--reload` (watches `engine/` dir) for live Python edits. Frontend is static HTML served fresh per request — call `reload_preview` after frontend edits.
- No external credentials needed — fully local-first CV pipeline.
