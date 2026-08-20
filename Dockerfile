FROM node:22-alpine AS frontend-build

WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/tsconfig.json ./tsconfig.json
COPY frontend/src ./src
RUN npm run build

FROM rust:1.98-bookworm AS rust-build

WORKDIR /src
COPY rust-toolchain.toml ./rust-toolchain.toml
COPY backend-rust/Cargo.toml backend-rust/Cargo.toml
COPY backend-rust/src backend-rust/src
RUN CARGO_BUILD_JOBS=1 cargo build --release --manifest-path backend-rust/Cargo.toml

FROM debian:bookworm-slim AS runtime

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl tesseract-ocr poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY --from=rust-build /src/backend-rust/target/release/nephroscan-backend ./nephroscan-backend
COPY --from=frontend-build /frontend/dist ./frontend/dist
COPY frontend/index.html ./frontend/index.html
COPY frontend/landing.html ./frontend/landing.html
COPY frontend/demo_assets ./frontend/demo_assets

# ONNX graphs and JSON manifests are generated outside this image from the
# trained checkpoints.  They are the only model artifacts copied at runtime;
# no Python, checkpoints, or credentials are part of the service image.
COPY models/ ./models/
RUN set -eu; \
    for model in kidney chest brain heart; do \
      test -s "models/$model.onnx"; \
      test -s "models/$model.json"; \
    done; \
    test -s models/models.json; \
    size_mb="$(du -sm models | cut -f1)"; \
    test "$size_mb" -le 500

ENV PORT=8080 \
    MODEL_DIR=/app/models \
    FRONTEND_DIR=/app/frontend \
    RUST_LOG=info \
    TOKIO_WORKER_THREADS=1 \
    MALLOC_ARENA_MAX=2

EXPOSE 8080

# Rust runs as one process/instance to fit Render's 512 MB free tier.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD-SHELL curl --fail --silent "http://127.0.0.1:${PORT:-8080}/api/health" >/dev/null || exit 1

CMD ["./nephroscan-backend"]
