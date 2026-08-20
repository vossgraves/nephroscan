# NephroScan AI

Medical imaging AI platform for expo demonstration. Educational prototype only — not a medical device.

## Quick Start

### Local Development

```bash
pip install -r requirements.txt
python app.py
```

Server starts at `http://localhost:5000`. Open in browser.

### Docker

```bash
docker compose up --build
```

### Production (Render)

Push to GitHub. Render auto-deploys using `render.yaml`.

### Railway (Trial-friendly)

This repo includes `railway.json` and a multi-stage `Dockerfile`. Railway builds the TypeScript frontend, serves the Flask app on its injected `$PORT`, and checks `/api/health` before routing traffic.

```bash
railway init --name nephroscan
railway up
railway domain
```

Keep one replica and one Gunicorn worker on the Trial plan. The four local checkpoints occupy about 171 MiB on disk. On this workstation the full Flask process measured about 495 MiB after loading/warming all four models and about 499 MiB after one prediction; leave headroom for concurrent requests and verify the deployed peak in Railway metrics. OpenAI calls add no resident model; configure `OPENAI_API_KEY` as a Railway service variable to enable optional vision/chat assistance.
The GitHub release intentionally omits raw `.pth` checkpoints to avoid a 200MB source upload. The Dockerfile downloads the four required checkpoints from `MODEL_BASE_URL` when `models/` is empty; local non-Docker inference still needs those files in `models/`.


## Project Structure

```
NephroScan-AI/
├── app.py                 # Unified Flask server (API + frontend)
├── ai/gradcam.py          # Grad-CAM explainability
├── models/                # ResNet-18 .pth checkpoints
│   ├── kidney_stone_resnet18.pth
│   ├── chest_pneumonia_resnet18.pth
│   ├── brain_mri_resnet18.pth
│   └── heart_cardiomegaly_resnet18_improved.pth
├── frontend/index.html    # HTML shell and visual system
├── frontend/src/          # Strict TypeScript browser integration
├── requirements.txt
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
| `/api/lab/analyze` | POST | Lab report OCR + guidance |
| `/api/ai/health` | GET | Whether the optional AI layer is configured |
| `/api/ai/analyze-image` | POST | Optional AI description of an image outside the trained modalities |
| `/api/ai/chat` | POST | Optional AI Q&A about a health report / result |

## Optional AI Assistance

The four local ResNet-18 classifiers cover kidney, chest, brain, and heart
scans only. An optional OpenAI-backed layer handles everything else: a plain
description of an image outside those modalities, and conversational
explanation of a report. It is **off by default**.

### Enabling it

```bash
# server-side only — never exposed to the browser
export OPENAI_API_KEY=sk-your-key-here
export OPENAI_VISION_MODEL=gpt-5.6-luna   # optional, cost-sensitive multimodal model
export OPENAI_CHAT_MODEL=gpt-5.6-luna     # optional, cost-sensitive chat model
python app.py
```

`GET /api/ai/health` reports `{"enabled": true|false}` so the frontend can show
the right affordance. `OPENAI_BASE_URL` accepts any OpenAI-compatible
`/chat/completions` endpoint, so the deployment can be pointed at a gateway
without code changes.

### Contracts

`POST /api/ai/analyze-image` — `multipart/form-data`: `image` (JPG/PNG/WEBP,
4 MB max), optional `scan_type`, optional `context`.

```json
{ "status": "ok", "provider": "openai", "model": "gpt-5.6-luna",
  "summary": "...", "findings": ["..."], "limitations": ["..."],
  "next_steps": ["..."], "disclaimer": "...", "request_id": "ai_…" }
```

`POST /api/ai/chat` — JSON: `messages` (array of `{role: "user"|"assistant",
content: string}`), optional `context`.

```json
{ "status": "ok", "provider": "openai", "model": "gpt-5.6-luna",
  "message": "...", "disclaimer": "...", "request_id": "ai_…" }
```

### Behaviour when disabled or failing

Every non-2xx answer is JSON with a `request_id` and **no** generated clinical
content — the frontend must fall back to local guidance instead of showing
partial results.

| Situation | HTTP | Body |
|---|---|---|
| No `OPENAI_API_KEY` | 503 | `status: "disabled"`, `code: "ai_disabled"` |
| Bad upload (missing/corrupt/empty) | 400 | `status: "error"` |
| Wrong file type (PDF, DICOM, …) | 415 | `code: "unsupported_type"` |
| Image or request body too large | 413 | `code: "image_too_large"` / `payload_too_large` |
| Provider rate limit | 429 | `code: "upstream_rate_limited"` |
| Provider unreachable, rejected, or unparseable | 502 | `code: "upstream_*"` |
| Provider timeout | 504 | `code: "upstream_timeout"` |

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | 5000 | Server port |
| `MODEL_DIR` | models | Model checkpoint directory |
| `CORS_ORIGINS` | * | Allowed CORS origins |
| `MAX_UPLOAD_BYTES` | 8388608 | Max upload size for the local model routes (8MB) |
| `INFERENCE_TIMEOUT` | 30 | Inference timeout (seconds) |
| `OPENAI_API_KEY` | *(unset)* | Enables `/api/ai/*`; server-side only |
| `OPENAI_BASE_URL` | https://api.openai.com/v1 | OpenAI-compatible base URL (gateway-friendly) |
| `OPENAI_VISION_MODEL` | gpt-5.6-luna | Model for `/api/ai/analyze-image` |
| `OPENAI_CHAT_MODEL` | gpt-5.6-luna | Model for `/api/ai/chat` |
| `OPENAI_TIMEOUT` | 45 | Outbound AI timeout, seconds (5–120) |
| `OPENAI_MAX_OUTPUT_TOKENS` | 700 | AI response cap (128–4096) |
| `AI_MAX_IMAGE_BYTES` | 4194304 | Max AI image upload (4MB) |
| `AI_MAX_IMAGE_DIM` | 1024 | Longest edge sent to the provider |
| `AI_MAX_JSON_BYTES` | 65536 | Max `/api/ai/chat` request body |
| `AI_MAX_MESSAGES` | 20 | Newest conversation turns forwarded |
| `AI_MAX_MESSAGE_CHARS` | 4000 | Per-message character limit |
| `AI_MAX_TOTAL_CHARS` | 24000 | Conversation character budget |
| `AI_MAX_CONTEXT_CHARS` | 4000 | Report-context character limit |

## Safety

- Every result includes provenance metadata (model, timestamp, inference type)
- Thermal proxy is labeled "VISUAL SIMULATION" — not an infrared measurement
- No clinical diagnosis — all results are AI-assisted screening prototypes
- See `DEMO_SCRIPT.md` for presentation guidelines
- `OPENAI_API_KEY` stays on the server; the browser only calls same-origin
  `/api/ai/*` routes and never receives a credential
- AI prompts forbid diagnosis, prescribing, and certainty; they require stated
  uncertainty, clinician review, and emergency escalation for red-flag symptoms
- Every AI response carries the educational disclaimer; if the provider returns
  nothing usable the route fails with an error instead of inventing findings
- AI logs record only a `request_id`, route, byte count, model, and latency —
  never image bytes, message text, or credentials (upstream error bodies are
  redacted before logging)
