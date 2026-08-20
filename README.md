# NephroScan AI

Medical imaging AI platform for expo demonstration. Educational prototype only — not a medical device.

## Quick Start

### Local Development

```bash
# 1. Build the typed frontend module (frontend/dist/main.js)
cd frontend && npm ci && npm run build && cd ..

# 2. Run the Rust service; it serves the API and the frontend on one origin
cargo run --release --manifest-path backend-rust/Cargo.toml
```

Server starts at `http://localhost:8080`. Open in browser.

`MODEL_DIR` (default `models`) must contain `kidney.onnx`, `chest.onnx`,
`brain.onnx`, `heart.onnx`, each modality's `<name>.json` manifest, and
`models.json`. The ONNX graphs are exported from the trained checkpoints
out-of-band and are never committed as raw `.pth` files.

### Docker

```bash
docker compose up --build
```

### Production (Render)

Push to GitHub. Render auto-deploys using `render.yaml`.

### Railway (Trial-friendly)

This repo includes `railway.json` and a multi-stage `Dockerfile`. Railway
builds the TypeScript frontend, compiles the Rust service, runs it on the
injected `$PORT`, and checks `/api/health` before routing traffic.

```bash
railway init --name nephroscan
railway up
railway domain
```

Keep one replica and `TOKIO_WORKER_THREADS=1` on free/trial plans. The four
exported ONNX graphs are the only model artifacts in the image and the runtime
stage asserts they stay under 500 MB. Remote AI calls add no resident model;
configure `AI_API_KEY` as a service variable to enable optional vision/chat
assistance.

## Project Structure

```
NephroScan-AI/
├── backend-rust/          # Axum + ONNX Runtime service (API + static frontend)
│   ├── Cargo.toml
│   └── src/
├── models/                # Exported ONNX graphs + JSON manifests (not in Git)
│   ├── kidney.onnx / kidney.json
│   ├── chest.onnx  / chest.json
│   ├── brain.onnx  / brain.json
│   ├── heart.onnx  / heart.json
│   └── models.json
├── frontend/index.html    # HTML shell and visual system
├── frontend/src/          # Strict TypeScript browser integration
├── Dockerfile
├── railway.json
├── docker-compose.yml
├── render.yaml
└── .env.example
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Server + model status |
| `/api/predict` | POST | Kidney stone classification |
| `/api/predict-chest` | POST | Chest pneumonia classification |
| `/api/predict-brain` | POST | Brain MRI classification |
| `/api/predict-heart` | POST | Heart cardiomegaly classification |
| `/api/explain` | POST | Grad-CAM attention heatmap |
| `/api/lab/health` | GET | Whether laboratory extraction is available |
| `/api/lab/analyze` | POST | Lab report intake + guidance (see note below) |
| `/api/ai/health` | GET | Whether the optional AI layer is configured |
| `/api/ai/analyze-image` | POST | Optional AI description of an image outside the trained modalities |
| `/api/ai/chat` | POST | Optional AI Q&A about a health report / result |

`/api/lab/health` reports `ocr_available: false` in this Rust build: the
service accepts a lab upload and returns a `needs_review` envelope with an
empty `tests` array instead of running OCR. The Lab Test Analysis view stays
usable — enter the values manually in the editable table before generating
the report.

## Optional AI Assistance

The four local ONNX classifiers cover kidney, chest, brain, and heart
scans only. The optional Gemini layer handles out-of-scope image descriptions
and conversational report explanations through Google's OpenAI-compatible
chat-completions endpoint. It is **off by default**.

### Enabling it

```bash
# server-side only — never exposed to the browser
export AI_PROVIDER=gemini
export AI_API_KEY=your-gemini-key
export AI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
export AI_VISION_MODEL=gemini-3.5-flash-lite
export AI_CHAT_MODEL=gemini-3.5-flash-lite
cargo run --release --manifest-path backend-rust/Cargo.toml
```

Create a key in [Google AI Studio](https://aistudio.google.com/apikey). The
free tier is rate-limited and Google states that free-tier content may be used
to improve its products; use synthetic/demo images unless your privacy and
consent requirements allow otherwise.

`GET /api/ai/health` reports `{"enabled": true|false}` so the frontend can
show the correct state. `AI_BASE_URL` remains configurable for another
OpenAI-compatible provider without changing the browser contract.

### Contracts

`POST /api/ai/analyze-image` — `multipart/form-data`: `image` (JPG/PNG/WEBP,
4 MB max), optional `scan_type`, optional `context`.

```json
{ "status": "ok", "provider": "gemini", "model": "gemini-3.5-flash-lite",
  "summary": "...", "findings": ["..."], "limitations": ["..."],
  "next_steps": ["..."], "disclaimer": "...", "request_id": "ai_…" }
```

`POST /api/ai/chat` — JSON: `messages` (array of `{role: "user"|"assistant",
content: string}`), optional `context`.

```json
{ "status": "ok", "provider": "gemini", "model": "gemini-3.5-flash-lite",
  "message": "...", "disclaimer": "...", "request_id": "ai_…" }
```

### Behaviour when disabled or failing

Every non-2xx answer is JSON with a `request_id` and **no** generated clinical
content. The frontend keeps local guidance available instead of showing
partial results.

| Situation | HTTP | Body |
|---|---|---|
| No `AI_API_KEY` | 503 | `status: "disabled"`, `code: "ai_disabled"` |
| Bad upload (missing/corrupt/empty) | 400 | `status: "error"` |
| Wrong file type (PDF, DICOM, …) | 415 | `code: "unsupported_type"` |
| Image or request body too large | 413 | `code: "image_too_large"` / `payload_too_large` |
| Provider rate limit | 429 | `code: "upstream_rate_limited"` |
| Provider unreachable, rejected, or unparseable | 502 | `code: "upstream_*"` |
| Provider timeout | 504 | `code: "upstream_timeout"` |

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | 8080 | Server port (Render/Railway inject their own) |
| `MODEL_DIR` | models | Directory of exported `.onnx` graphs + `.json` manifests |
| `FRONTEND_DIR` | frontend | Directory served for same-origin static assets |
| `AI_PROVIDER` | gemini | Provider label returned in AI responses |
| `AI_API_KEY` | *(unset)* | Enables `/api/ai/*`; server-side only |
| `AI_BASE_URL` | Gemini OpenAI-compatible endpoint | Provider API base URL |
| `AI_VISION_MODEL` | gemini-3.5-flash-lite | Model for `/api/ai/analyze-image` |
| `AI_CHAT_MODEL` | gemini-3.5-flash-lite | Model for `/api/ai/chat` |
| `AI_TIMEOUT` | 45 | Outbound AI timeout, seconds (5–120) |

Request bounds are compile-time constants in `backend-rust/src/main.rs`:
16 MB per local scan upload, 4 MB per AI image, 64 KB per `/api/ai/chat`
body, 20 forwarded turns, 4000 chars per message, 24 000 chars per
conversation, 4000 chars of report context.

## Safety

- Every result includes provenance metadata (model, timestamp, inference type)
- Grad-CAM attention maps are labeled as model attention, not a measurement
- No clinical diagnosis — all results are AI-assisted screening prototypes
- See `DEMO_SCRIPT.md` for presentation guidelines
- `AI_API_KEY` stays on the server; the browser only calls same-origin
  `/api/ai/*` routes and never receives a credential
- AI prompts forbid diagnosis, prescribing, and certainty; they require stated
  uncertainty, clinician review, and emergency escalation for red-flag symptoms
- Every AI response carries the educational disclaimer; if the provider returns
  nothing usable the route fails with an error instead of inventing findings
- AI logs record only a `request_id`, route, byte count, model, and latency —
  never image bytes, message text, or credentials (upstream error bodies are
  redacted before logging)
