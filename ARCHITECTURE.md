# NephroScan AI — Architecture

## Overview

Single-server Rust application (Axum + ONNX Runtime) serving both the API and the frontend from one origin. Designed for free-tier Docker hosting on Render or Railway.

```
Browser (index.html)
    │
    ├── GET  /                    → serves index.html
    ├── GET  /api/health          → model status JSON
    ├── POST /api/predict         → kidney classification
    ├── POST /api/predict-chest   → chest classification
    ├── POST /api/predict-brain   → brain classification
    ├── POST /api/predict-heart   → heart classification
    ├── POST /api/explain         → Grad-CAM heatmap
    ├── GET  /api/lab/health      → lab extraction availability
    ├── POST /api/lab/analyze     → lab report intake + guidance
    ├── GET  /api/ai/health       → optional AI layer status
    ├── POST /api/ai/analyze-image→ optional AI image description
    └── POST /api/ai/chat         → optional AI report Q&A
```

## Backend (`backend-rust/`)

- **Runtime:** single Axum process, `TOKIO_WORKER_THREADS=1`, no interpreter and no GC
- **Model loading:** all 4 exported ONNX graphs are loaded once at startup from `MODEL_DIR` into shared session state
- **Inference pipeline:** image → decode → per-manifest resize/grayscale → NCHW tensor → ONNX Runtime → softmax → result
- **Manifest-driven preprocessing:** each `<modality>.json` carries `input_size`, `grayscale`, `normalize_mean/std`, `classes`, `positive_class`, and `threshold`, so preprocessing stays pinned to the trained checkpoints
- **Calibrated thresholds:** chest (0.80) and heart (0.60) come from the manifests, not from code
- **Provenance:** every response includes model name, version, timestamp, device, and inference type
- **Upload validation:** content type, size (16 MB per local scan upload), and image-decode integrity checks
- **Lab route:** `/api/lab/analyze` accepts an upload but does not run OCR in this build; it returns a `needs_review` envelope with an empty `tests` array so the frontend's manual-entry table stays authoritative

## Optional Gemini Intelligence Layer (`/api/ai/*`)

The local classifiers only cover kidney, chest, brain, and heart scans. The
Gemini layer covers the rest — describing an out-of-scope image and answering
questions about a report — and is inert unless `AI_API_KEY` is set.

```
Browser ──same-origin POST──► Axum /api/ai/*
                                 │  validate + bound + encode
                                 │  (key never leaves the server)
                                 ▼
             Gemini OpenAI-compatible /chat/completions
```

- **Disabled by default:** without a key both POST routes return
  `503 {"status":"disabled","code":"ai_disabled"}` and the frontend keeps
  using local model output and offline guidance. `GET /api/ai/health` exposes the flag.
- **Transport:** one bounded outbound HTTPS call per request through a shared
  `reqwest` client, using Gemini's OpenAI-compatible chat-completions endpoint.
  `AI_BASE_URL` keeps the provider swappable without browser changes. No extra
  process, thread, or resident model.
- **Image path:** JPG/PNG/WEBP only, ≤ 4 MB, decoded and bounds-checked
  (longest edge ≤ 2048) before the bytes are base64-encoded into a data URL.
  Nothing is written to disk and the buffer is dropped when the call returns.
- **Chat path:** JSON body ≤ 64 KB; roles restricted to `user`/`assistant`;
  the newest 20 turns are forwarded under a 24 000-character budget with a
  4000-character per-message cap. Caller context is truncated to 4000
  characters and injected as untrusted data.
- **Prompt safety:** shared rules forbid diagnosis, certainty, and prescribing;
  they require explicit uncertainty, clinician review, and immediate emergency
  care for red-flag symptoms. Every response carries the educational disclaimer.
- **No fabrication:** `analyze-image` requires a parseable JSON object with a
  non-empty `summary`; malformed provider output becomes `502 upstream_malformed`
  with no clinical fields. Provider faults map to explicit error codes.
- **Observability:** each request gets an `ai_<hex>` request id returned to the
  client and logged with route, byte count, model, and latency. Image bytes,
  message text, and credentials are never logged.

## Frontend (`frontend/index.html` + `frontend/src`)

The existing HTML shell keeps the proven imaging, lab, history, and demo flows. A strict TypeScript module under `frontend/src` is compiled to `frontend/dist/main.js` and loaded after the legacy handlers. The Docker build compiles it in a Node stage; local development runs `cd frontend && npm ci && npm run build`. The module talks to same-origin `/api/*` only and never receives a credential.

### Views
- **Dashboard** — Model status, session stats, quick actions
- **New Analysis** — Upload images, select scan type, run inference
- **Report History** — Session reports with filtering
- **Patients** — Patient management (demo)
- **Compare Scans** — Side-by-side result comparison
- **Lab Test Analysis** — Lab upload, editable value table, generated report
- **Model Performance** — Model accuracy metrics
- **AI Assistant** — Clinical Q&A backed by `/api/ai/chat`
- **Why NephroScan?** — Method, scope, and limitation notes
- **Settings** — Theme, thresholds, export

### Navigation and motion
Below 920px the sidebar becomes a fixed drawer. Opening it is compositor-only:
`transform` on the drawer (320ms, `--ease-drawer`) and `opacity` on the scrim
(220ms), with `visibility` delayed so the closed drawer leaves the tab order.
Nav items stagger in at 30ms steps. `prefers-reduced-motion: reduce` collapses
durations and zeroes every delay; `forced-colors: active` restores system
borders. The AI outage panel at the bottom of the drawer is a
`role="status"` live region toggled from `renderAiCapabilityNotice()`.

## Models

| Model | File | Input | Classes | Calibrated |
|---|---|---|---|---|
| Kidney Stone | `kidney.onnx` | 128px, grayscale | Normal / stone | No |
| Chest Pneumonia | `chest.onnx` | 128px, RGB | normal / pneumonia | Yes (0.80) |
| Brain MRI | `brain.onnx` | 96px, grayscale | glioma / meningioma / notumor / pituitary | No |
| Heart Cardiomegaly | `heart.onnx` | 160px, RGB | false / true | Yes (0.60) |

Each graph ships with a `<modality>.json` manifest holding its input size,
grayscale flag, normalization constants, class order, positive class, and
threshold. `models.json` indexes the four. Raw `.pth` checkpoints stay out of
Git and out of the image.

## Deployment

- **Platform:** Render free tier or Railway Trial, Docker runtime
- **Railway config:** `railway.json` selects the Dockerfile, sets the `/api/health` healthcheck, and bounds the restart policy; the image's `CMD` starts the Rust binary
- **Render config:** `render.yaml` pins the free plan, one instance, `MODEL_DIR`/`FRONTEND_DIR`, and `TOKIO_WORKER_THREADS=1`
- **Docker:** Node stage compiles `frontend/src` → `frontend/dist/main.js`; Rust stage builds `nephroscan-backend`; the Debian slim runtime carries only the binary, the frontend, and `models/`. The runtime stage fails the build if any of the four `.onnx`/`.json` pairs is missing or if `models/` exceeds 500 MB.
- **Memory:** one process, one Tokio worker, four ONNX sessions resident. Free tiers cap at 512 MB, so keep a single instance and verify the deployed peak in platform metrics.
- **Auto-deploy:** Railway deploys from the linked project with `railway up`; Render rebuilds from GitHub using `render.yaml`
