use axum::{
    body::Bytes,
    extract::{Multipart, Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use image::{imageops::FilterType, DynamicImage, ImageFormat, ImageReader, Rgb};
use regex::Regex;
use reqwest::Client;
use tempfile::NamedTempFile;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::{
    collections::HashMap,
    env,
    io::Cursor,
    fs,
    process::Command,
    path::{Path as FsPath, PathBuf},
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::net::TcpListener;
use tower_http::{
    compression::CompressionLayer,
    cors::CorsLayer,
    limit::RequestBodyLimitLayer,
    services::ServeDir,
    trace::TraceLayer,
};
use tracing::{info, warn};
use tract_onnx::prelude::*;

const API_DISCLAIMER: &str = "For education and decision support only; not a medical diagnosis. Review results with a qualified clinician.";
const AI_DISCLAIMER: &str = "AI output is informational and may be wrong. It is not a medical diagnosis; consult a qualified clinician.";
const LAB_DISCLAIMER: &str = "Educational lab extraction only. Confirm important values and reference ranges with the original report and a qualified clinician.";
const MAX_LOCAL_IMAGE_BYTES: usize = 16 * 1024 * 1024;
const MAX_AI_IMAGE_BYTES: usize = 4 * 1024 * 1024;
const MAX_AI_JSON_BYTES: usize = 64 * 1024;
const MAX_MESSAGES: usize = 20;
const MAX_MESSAGE_CHARS: usize = 4000;
const MAX_TOTAL_CHARS: usize = 24_000;
const MAX_CONTEXT_CHARS: usize = 4000;

type OnnxPlan = TypedSimplePlan;

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ModelManifest {
    #[serde(default, alias = "filename", alias = "file", alias = "onnx")]
    model: String,
    #[serde(default = "default_input_size")]
    input_size: usize,
    #[serde(default = "default_true")]
    grayscale: bool,
    #[serde(default = "default_channels")]
    channels: usize,
    #[serde(default = "default_mean")]
    normalize_mean: Vec<f32>,
    #[serde(default = "default_std")]
    normalize_std: Vec<f32>,
    #[serde(default)]
    classes: Vec<String>,
    #[serde(default)]
    positive_class: Option<String>,
    #[serde(default)]
    threshold: Option<f32>,
}

fn default_input_size() -> usize { 128 }
fn default_true() -> bool { true }
fn default_channels() -> usize { 3 }
fn default_mean() -> Vec<f32> { vec![0.485, 0.456, 0.406] }
fn default_std() -> Vec<f32> { vec![0.229, 0.224, 0.225] }

fn default_manifest(name: &str) -> ModelManifest {
    let (size, gray, classes, positive, threshold) = match name {
        "kidney" => (128, true, vec!["Normal".into(), "stone".into()], None, None),
        "chest" => (128, false, vec!["normal".into(), "pneumonia".into()], Some("pneumonia".into()), Some(0.80)),
        "brain" => (96, true, vec!["glioma".into(), "meningioma".into(), "notumor".into(), "pituitary".into()], None, None),
        "heart" => (160, false, vec!["false".into(), "true".into()], Some("true".into()), Some(0.60)),
        _ => (128, true, vec![], None, None),
    };
    ModelManifest {
        model: format!("{name}.onnx"), input_size: size, grayscale: gray, channels: 3,
        normalize_mean: default_mean(), normalize_std: default_std(), classes,
        positive_class: positive, threshold,
    }
}

#[derive(Clone)]
struct LoadedModel {
    manifest: ModelManifest,
    plan: Option<Arc<OnnxPlan>>,
}

#[derive(Clone)]
struct AppState {
    models: Arc<HashMap<String, LoadedModel>>,
    frontend_dir: PathBuf,
    started_at: String,
    http: Client,
    ai: AiConfig,
}

#[derive(Clone)]
struct AiConfig {
    provider: String,
    api_key: Option<String>,
    base_url: String,
    vision_model: String,
    chat_model: String,
    timeout: Duration,
}

#[derive(Debug)]
enum AppError {
    BadRequest(&'static str),
    TooLarge(&'static str),
    NotFound(&'static str),
    Unavailable(&'static str),
    Upstream(&'static str),
    Internal(&'static str),
}

impl AppError {
    fn status(&self) -> StatusCode {
        match self {
            Self::BadRequest(_) => StatusCode::BAD_REQUEST,
            Self::TooLarge(_) => StatusCode::PAYLOAD_TOO_LARGE,
            Self::NotFound(_) => StatusCode::NOT_FOUND,
            Self::Unavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
            Self::Upstream(_) => StatusCode::BAD_GATEWAY,
            Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
    fn code(&self) -> &'static str {
        match self {
            Self::BadRequest(_) => "invalid_request",
            Self::TooLarge(_) => "payload_too_large",
            Self::NotFound(_) => "not_found",
            Self::Unavailable(_) => "unavailable",
            Self::Upstream(_) => "upstream_error",
            Self::Internal(_) => "internal_error",
        }
    }
    fn message(&self) -> &'static str {
        match self {
            Self::BadRequest(m) | Self::TooLarge(m) | Self::NotFound(m) | Self::Unavailable(m) | Self::Upstream(m) | Self::Internal(m) => m,
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = self.status();
        let body = Json(json!({"status": if status == StatusCode::SERVICE_UNAVAILABLE { "disabled" } else { "error" }, "code": self.code(), "error": self.message(), "message": self.message(), "disclaimer": API_DISCLAIMER}));
        (status, body).into_response()
    }
}

fn env_usize(name: &str, default: usize, min: usize, max: usize) -> usize {
    env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default).clamp(min, max)
}
fn env_string(name: &str, default: &str) -> String { env::var(name).unwrap_or_else(|_| default.to_string()).trim().to_string() }
fn now_iso() -> String {
    let millis = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis();
    format!("unix-ms:{millis}")
}
fn truncate_chars(text: &str, max: usize) -> String { text.chars().take(max).collect() }
fn decode_bounded(bytes: &[u8], max_dimension: u32) -> Result<DynamicImage, AppError> {
    let reader = ImageReader::new(Cursor::new(bytes)).with_guessed_format().map_err(|_| AppError::BadRequest("uploaded file is not a supported image"))?;
    let (width, height) = reader.into_dimensions().map_err(|_| AppError::BadRequest("uploaded file is not a supported image"))?;
    if width == 0 || height == 0 || width > max_dimension || height > max_dimension {
        return Err(AppError::BadRequest("image dimensions exceed the safety limit"));
    }
    let reader = ImageReader::new(Cursor::new(bytes)).with_guessed_format().map_err(|_| AppError::BadRequest("uploaded file is not a supported image"))?;
    reader.decode().map_err(|_| AppError::BadRequest("uploaded file is not a supported image"))
}

fn merge_json(base: &mut Value, overlay: Value) {
    if let (Some(a), Some(b)) = (base.as_object_mut(), overlay.as_object()) {
        for (key, value) in b { a.insert(key.clone(), value.clone()); }
    }
}

fn load_manifests(model_dir: &FsPath) -> HashMap<String, ModelManifest> {
    let mut result = HashMap::new();
    let mut root = Value::Null;
    if let Ok(data) = std::fs::read_to_string(model_dir.join("models.json")) {
        if let Ok(value) = serde_json::from_str::<Value>(&data) { root = value; }
    }
    let root_models = root.get("models").unwrap_or(&root);
    for name in ["kidney", "chest", "brain", "heart"] {
        let mut value = root_models.get(name).cloned().unwrap_or_else(|| json!({}));
        if let Ok(data) = std::fs::read_to_string(model_dir.join(format!("{name}.json"))) {
            if let Ok(overlay) = serde_json::from_str::<Value>(&data) { merge_json(&mut value, overlay); }
        }
        let mut fallback = serde_json::to_value(default_manifest(name)).unwrap_or_else(|_| json!({}));
        merge_json(&mut fallback, value);
        let mut manifest: ModelManifest = serde_json::from_value(fallback).unwrap_or_else(|_| default_manifest(name));
        if manifest.model.is_empty() { manifest.model = format!("{name}.onnx"); }
        result.insert(name.to_string(), manifest);
    }
    result
}

fn load_models(model_dir: &FsPath) -> HashMap<String, LoadedModel> {
    let manifests = load_manifests(model_dir);
    let mut loaded = HashMap::new();
    for name in ["kidney", "chest", "brain", "heart"] {
        let manifest = manifests.get(name).cloned().unwrap_or_else(|| default_manifest(name));
        let path = model_dir.join(&manifest.model);
        let plan = match tract_onnx::onnx().model_for_path(&path).and_then(|m| m.into_optimized()).and_then(|m| m.into_runnable()) {
            Ok(plan) => Some(plan),
            Err(error) => {
                warn!(modality = name, model = %manifest.model, error = %error, "ONNX model unavailable");
                None
            }
        };
        if plan.is_some() { info!(modality = name, model = %manifest.model, "loaded ONNX model"); }
        loaded.insert(name.to_string(), LoadedModel { manifest, plan });
    }
    loaded
}

fn provenance(name: &str, model: &LoadedModel) -> Value {
    json!({"model": model.manifest.model, "version": "rust-onnx-1", "inference_type": if model.plan.is_some() { "REAL_MODEL_INFERENCE" } else { "MODEL_UNAVAILABLE" }, "timestamp": now_iso(), "device": "cpu", "preprocessing": format!("resize-{}-{}-normalize", model.manifest.input_size, if model.manifest.grayscale { "grayscale" } else { "rgb" }), "modality": name})
}

fn preprocess(image: &DynamicImage, manifest: &ModelManifest) -> Result<Vec<f32>, AppError> {
    let size = manifest.input_size;
    if size == 0 || manifest.channels != 3 || manifest.normalize_mean.len() < 3 || manifest.normalize_std.len() < 3 {
        return Err(AppError::Internal("model preprocessing configuration is invalid"));
    }
    let rgb = image.to_rgb8();
    let resized = image::imageops::resize(&rgb, size as u32, size as u32, FilterType::Triangle);
    let gray = if manifest.grayscale { Some(DynamicImage::ImageRgb8(resized.clone()).to_luma8()) } else { None };
    let mut output = vec![0.0; 3 * size * size];
    for y in 0..size {
        for x in 0..size {
            let channels = if let Some(luma) = &gray {
                let value = luma.get_pixel(x as u32, y as u32)[0] as f32 / 255.0;
                [value, value, value]
            } else {
                let p = resized.get_pixel(x as u32, y as u32);
                [p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0]
            };
            for c in 0..3 { output[c * size * size + y * size + x] = (channels[c] - manifest.normalize_mean[c]) / manifest.normalize_std[c]; }
        }
    }
    Ok(output)
}

fn run_logits(plan: &Arc<OnnxPlan>, values: Vec<f32>, manifest: &ModelManifest) -> Result<Vec<f32>, AppError> {
    let size = manifest.input_size;
    let input = Tensor::from_shape(&[1, 3, size, size], &values).map_err(|_| AppError::Internal("could not construct model input"))?;
    let outputs = plan.run(tvec!(input.into())).map_err(|_| AppError::Internal("model inference failed"))?;
    let view = outputs.first().ok_or(AppError::Internal("model returned no output"))?.to_plain_array_view::<f32>().map_err(|_| AppError::Internal("model output was not numeric"))?;
    Ok(view.iter().copied().collect())
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() { return vec![]; }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut values: Vec<f32> = logits.iter().map(|x| (*x - max).exp()).collect();
    let total: f32 = values.iter().sum();
    if total > 0.0 { for x in &mut values { *x /= total; } }
    values
}

fn prediction(name: &str, model: &LoadedModel, image: &DynamicImage) -> Result<Value, AppError> {
    let plan = model.plan.as_ref().ok_or(AppError::Unavailable("local model is unavailable"))?;
    let start = Instant::now();
    let values = preprocess(image, &model.manifest)?;
    let preprocess_ms = start.elapsed().as_secs_f64() * 1000.0;
    let infer_start = Instant::now();
    let probabilities = softmax(&run_logits(plan, values, &model.manifest)?);
    let inference_ms = infer_start.elapsed().as_secs_f64() * 1000.0;
    if probabilities.is_empty() || model.manifest.classes.is_empty() { return Err(AppError::Internal("model classes or output are unavailable")); }
    let original_index = probabilities.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|x| x.0).unwrap_or(0).min(model.manifest.classes.len() - 1);
    let mut index = original_index;
    let positive_probability = model.manifest.threshold.and_then(|_| probabilities.get(1).copied());
    if let (Some(threshold), Some(positive)) = (model.manifest.threshold, positive_probability) { index = if positive >= threshold { 1 } else { 0 }; }
    let confidence = probabilities.get(index).copied().unwrap_or(0.0) * 100.0;
    let original_confidence = probabilities.get(original_index).copied().unwrap_or(0.0) * 100.0;
    let mut result = json!({"prediction": model.manifest.classes.get(index).cloned().unwrap_or_default(), "confidence": (confidence * 100.0).round() / 100.0, "classes": model.manifest.classes, "original_prediction": model.manifest.classes.get(original_index).cloned().unwrap_or_default(), "original_confidence": (original_confidence * 100.0).round() / 100.0, "threshold_calibrated": model.manifest.threshold.is_some(), "provenance": provenance(name, model), "timings": {"preprocess_ms": (preprocess_ms * 10.0).round() / 10.0, "inference_ms": (inference_ms * 10.0).round() / 10.0, "total_ms": (start.elapsed().as_secs_f64() * 1000.0 * 10.0).round() / 10.0}});
    if let (Some(threshold), Some(positive)) = (model.manifest.threshold, positive_probability) {
        let label = if name == "heart" { "cardiomegaly".to_string() } else { model.manifest.positive_class.clone().unwrap_or_else(|| model.manifest.classes.get(1).cloned().unwrap_or_default()) };
        result["positive_probability"] = json!((positive * 10000.0).round() / 100.0);
        result["decision_threshold"] = json!(threshold);
        result["calibrated_label"] = json!(label.clone());
        if label == "pneumonia" { result["pneumonia_probability"] = result["positive_probability"].clone(); }
        if label == "cardiomegaly" { result["cardiomegaly_probability"] = result["positive_probability"].clone(); }
    }
    Ok(result)
}

async fn multipart_image(mut multipart: Multipart, field_name: &'static str, max: usize) -> Result<(Vec<u8>, String, HashMap<String, String>), AppError> {
    let mut image = None;
    let mut content_type = String::new();
    let mut fields = HashMap::new();
    while let Some(field) = multipart.next_field().await.map_err(|_| AppError::BadRequest("invalid multipart form"))? {
        let name = field.name().unwrap_or("").to_string();
        if name == field_name || (field_name == "image" && name == "file") {
            let ct = field.content_type().unwrap_or("application/octet-stream").to_string();
            let filename = field.file_name().unwrap_or("upload").to_string();
            let bytes = field.bytes().await.map_err(|_| AppError::BadRequest("could not read uploaded image"))?;
            if bytes.is_empty() { return Err(AppError::BadRequest("uploaded image is empty")); }
            if bytes.len() > max { return Err(AppError::TooLarge("uploaded image exceeds the size limit")); }
            image = Some(bytes.to_vec()); content_type = ct;
            fields.insert("__filename".into(), filename);
        } else {
            let bytes = field.bytes().await.map_err(|_| AppError::BadRequest("invalid multipart field"))?;
            if bytes.len() <= MAX_CONTEXT_CHARS { fields.insert(name, String::from_utf8_lossy(&bytes).to_string()); }
        }
    }
    image.map(|bytes| (bytes, content_type, fields)).ok_or(AppError::BadRequest("expected a multipart image field"))
}

async fn predict_route(State(state): State<AppState>, Path(modality): Path<String>, multipart: Multipart) -> Result<Json<Value>, AppError> {
    let model = state.models.get(&modality).ok_or(AppError::NotFound("unknown modality"))?;
    let (bytes, _, _) = multipart_image(multipart, "image", MAX_LOCAL_IMAGE_BYTES).await?;
    let image = decode_bounded(&bytes, 4096)?;
    let result = prediction(&modality, model, &image)?;
    Ok(Json(result))
}
async fn predict_kidney(state: State<AppState>, multipart: Multipart) -> Result<Json<Value>, AppError> { predict_route(state, Path("kidney".into()), multipart).await }

async fn predict_chest(state: State<AppState>, multipart: Multipart) -> Result<Json<Value>, AppError> { predict_route(state, Path("chest".into()), multipart).await }
async fn predict_brain(state: State<AppState>, multipart: Multipart) -> Result<Json<Value>, AppError> { predict_route(state, Path("brain".into()), multipart).await }
async fn predict_heart(state: State<AppState>, multipart: Multipart) -> Result<Json<Value>, AppError> { predict_route(state, Path("heart".into()), multipart).await }

async fn health(State(state): State<AppState>) -> Json<Value> {
    let models: Map<String, Value> = state.models.iter().map(|(name, model)| (name.clone(), json!({"loaded": model.plan.is_some(), "checkpoint": model.manifest.model, "classes": model.manifest.classes}))).collect();
    let all_loaded = state.models.values().all(|m| m.plan.is_some());
    Json(json!({"status":"online", "service":"NephroScan AI", "version":"rust-onnx-1", "device":"cpu", "models":models, "all_models_loaded":all_loaded, "startup_time":state.started_at, "ai":{"enabled":state.ai.api_key.is_some(), "vision_model":if state.ai.api_key.is_some(){Some(state.ai.vision_model.clone())}else{None::<String>}, "chat_model":if state.ai.api_key.is_some(){Some(state.ai.chat_model.clone())}else{None::<String>}}, "endpoints":["/api/health","/api/predict","/api/predict-chest","/api/predict-brain","/api/predict-heart","/api/explain","/api/lab/health","/api/lab/analyze","/api/ai/health","/api/ai/analyze-image","/api/ai/chat"]}))
}

async fn explain(State(state): State<AppState>, multipart: Multipart) -> Result<Json<Value>, AppError> {
    let (bytes, _, fields) = multipart_image(multipart, "image", MAX_LOCAL_IMAGE_BYTES).await?;
    let modality = fields.get("scan_type").map(|s| s.trim().to_lowercase()).unwrap_or_default();
    let model = state.models.get(&modality).ok_or(AppError::BadRequest("unsupported scan_type; expected kidney, chest, brain, or heart"))?;
    let plan = model.plan.as_ref().ok_or(AppError::Unavailable("attention visualization is unavailable because the local model is not loaded"))?;
    let image = decode_bounded(&bytes, 2048)?;
    let size = model.manifest.input_size;
    let resized = image::imageops::resize(&image.to_rgb8(), size as u32, size as u32, FilterType::Triangle);
    let base_values = preprocess(&image, &model.manifest)?;
    let base_probs = softmax(&run_logits(plan, base_values, &model.manifest)?);
    let target = base_probs.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|x| x.0).unwrap_or(0);
    let baseline = *base_probs.get(target).unwrap_or(&0.0);
    let grid = 8usize;
    let patch = (size / grid).max(1);
    let mut drops = vec![0.0f32; grid * grid];
    for gy in 0..grid { for gx in 0..grid {
        let mut occluded = resized.clone();
        for y in (gy * patch)..((gy + 1) * patch).min(size) { for x in (gx * patch)..((gx + 1) * patch).min(size) { occluded.put_pixel(x as u32, y as u32, Rgb([128,128,128])); } }
        let occluded_image = DynamicImage::ImageRgb8(occluded);
        let probs = softmax(&run_logits(plan, preprocess(&occluded_image, &model.manifest)?, &model.manifest)?);
        drops[gy * grid + gx] = (baseline - probs.get(target).copied().unwrap_or(baseline)).max(0.0);
    }}
    let max_drop = drops.iter().copied().fold(0.0f32, f32::max);
    let mut overlay = resized.clone();
    for y in 0..size { for x in 0..size {
        let cell = (y / patch).min(grid - 1) * grid + (x / patch).min(grid - 1);
        let intensity = if max_drop > 0.0 { drops[cell] / max_drop } else { 0.0 };
        let p = overlay.get_pixel_mut(x as u32, y as u32);
        let alpha = 0.15 + intensity * 0.55;
        p[0] = ((p[0] as f32) * (1.0 - alpha) + 255.0 * alpha) as u8;
        p[1] = ((p[1] as f32) * (1.0 - alpha) + 32.0 * alpha) as u8;
        p[2] = ((p[2] as f32) * (1.0 - alpha) + 32.0 * alpha) as u8;
    }}
    let mut png = Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(overlay).write_to(&mut png, ImageFormat::Png).map_err(|_| AppError::Internal("could not encode explanation"))?;
    let class = model.manifest.classes.get(target).cloned().unwrap_or_default();
    Ok(Json(json!({"status":"ok", "scan_type":modality, "model":model.manifest.model, "prediction":class, "heatmap_image":format!("data:image/png;base64,{}", BASE64.encode(png.into_inner())), "method":"occlusion_sensitivity", "disclaimer":"CPU occlusion-sensitivity heatmap, not Grad-CAM, lesion segmentation, or diagnosis. Highlighted regions show where occlusion changed the model score and do not confirm disease.", "provenance":provenance(&modality, model)})))
}

fn command_available(command: &str) -> bool {
    Command::new(command).arg("--version").output().map(|output| output.status.success() || !output.stdout.is_empty() || !output.stderr.is_empty()).unwrap_or(false)
}

fn run_ocr(bytes: &[u8], filename: &str) -> Result<String, AppError> {
    if !command_available("tesseract") { return Err(AppError::Unavailable("native OCR is not installed")); }
    let input = NamedTempFile::new().map_err(|_| AppError::Internal("could not allocate OCR temporary file"))?;
    fs::write(input.path(), bytes).map_err(|_| AppError::Internal("could not write OCR temporary file"))?;
    let extension = FsPath::new(filename).extension().and_then(|v| v.to_str()).unwrap_or_default().to_ascii_lowercase();
    let mut converted: Option<NamedTempFile> = None;
    let image_path = if extension == "pdf" {
        if !command_available("pdftoppm") { return Err(AppError::Unavailable("native PDF rendering is not installed")); }
        let output = NamedTempFile::new().map_err(|_| AppError::Internal("could not allocate PDF temporary file"))?;
        let prefix = output.path().with_extension("");
        let status = Command::new("pdftoppm").args(["-f", "1", "-l", "1", "-singlefile", "-png"]).arg(input.path()).arg(&prefix).status().map_err(|_| AppError::Unavailable("native PDF rendering is not installed"))?;
        if !status.success() { return Err(AppError::BadRequest("the PDF could not be rendered")); }
        let page = prefix.with_extension("png");
        converted = Some(output);
        page
    } else { input.path().to_path_buf() };
    let output = Command::new("tesseract").arg(&image_path).arg("stdout").arg("--dpi").arg("150").output().map_err(|_| AppError::Unavailable("native OCR is not installed"))?;
    drop(converted);
    if !output.status.success() { return Err(AppError::BadRequest("the report could not be read")); }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn lab_report(text: &str, filename: &str, context: &Value) -> Value {
    let pattern = Regex::new(r"(?m)^([A-Za-z][A-Za-z0-9 /(),.%-]{1,60})\s+(\d+\.?\d*)\s+([A-Za-z/%μµ IU.-]{0,20})\s*(?:(\d+\.?\d+)\s*[-–]\s*(\d+\.?\d+))?").expect("valid lab regex");
    let mut tests = Vec::new();
    for captures in pattern.captures_iter(text).take(100) {
        let name = captures.get(1).map(|v| v.as_str().trim()).unwrap_or_default();
        let value = captures.get(2).map(|v| v.as_str().trim()).unwrap_or_default();
        if name.len() < 2 || value.is_empty() || name.to_ascii_lowercase().contains("patient") { continue; }
        let unit = captures.get(3).map(|v| v.as_str().trim()).unwrap_or_default();
        let low = captures.get(4).map(|v| v.as_str()).unwrap_or_default();
        let high = captures.get(5).map(|v| v.as_str()).unwrap_or_default();
        let numeric = value.parse::<f32>().ok();
        let status = match (numeric, low.parse::<f32>().ok(), high.parse::<f32>().ok()) {
            (Some(v), Some(lo), Some(_hi)) if v < lo => "Below stated range",
            (Some(v), Some(_lo), Some(hi)) if v > hi => "Above stated range",
            (Some(_), Some(_), Some(_)) => "Within stated range",
            _ => "Needs review",
        };
        tests.push(json!({"name":name,"value":value,"unit":unit,"refLow":low,"refHigh":high,"status":status}));
    }
    let summary = if tests.is_empty() { format!("Analysis of {filename} could not extract structured test values. Please review the original report manually.") } else { format!("Analysis of {filename} found {} readable test value(s). Values outside stated ranges require clinician review.", tests.len()) };
    let mut uncertainty = vec![];
    if tests.is_empty() { uncertainty.push("No structured test values could be extracted from the uploaded report."); }
    if context.get("fasting").and_then(Value::as_str).unwrap_or_default().is_empty() { uncertainty.push("Fasting status is unknown; some tests may require fasting."); }
    if context.get("symptoms").and_then(Value::as_str).unwrap_or_default().is_empty() { uncertainty.push("No symptom information was provided; clinical correlation is essential."); }
    json!({"status":"ok","tests":tests,"findings":[],"overallSummary":summary,"possibleProblems":[],"whatCanBeDone":["Share this report with a qualified clinician for interpretation.","Bring the original report to your next appointment.","Ask whether any flagged values need repeat testing."],"dietGuidance":["General balanced nutrition supports overall health.","Stay hydrated unless a clinician advises otherwise."],"lifestyleGuidance":["Maintain regular sleep and activity as tolerated and advised."],"urgencyGuidance":"Routine follow-up with your clinician is recommended; seek urgent care for severe symptoms.","doctorDiscussionPoints":["What do these results mean with my symptoms and history?","Are follow-up tests needed?"],"uncertainty":uncertainty,"disclaimer":LAB_DISCLAIMER})
}

async fn lab_health() -> Json<Value> { Json(json!({"status":"ok","lab_endpoint":true,"version":"lab-route-rust-2","ocr_available":command_available("tesseract"),"pdf_support":command_available("pdftoppm"),"endpoint":"/api/lab/analyze"})) }
async fn lab_analyze(multipart: Multipart) -> Result<Json<Value>, AppError> {
    let (bytes, _, fields) = multipart_image(multipart, "lab_report", 16 * 1024 * 1024).await?;
    if bytes.is_empty() { return Err(AppError::BadRequest("lab report is empty")); }
    let context = fields.get("context").and_then(|value| serde_json::from_str::<Value>(value).ok()).unwrap_or_else(|| json!({}));
    match run_ocr(&bytes, fields.get("__filename").map(String::as_str).unwrap_or("report.png")) {
        Ok(text) => Ok(Json(lab_report(&text, "uploaded report", &context))),
        Err(AppError::Unavailable(_)) => Ok(Json(json!({"status":"needs_review","tests":[],"findings":[],"uncertainty":["Native OCR is not available on this host; no laboratory findings were inferred."],"overallSummary":"The report needs manual review. Enter readable values into the extraction table.","possibleProblems":[],"whatCanBeDone":["Check the original report and review it with a qualified clinician."],"dietGuidance":[],"lifestyleGuidance":[],"urgencyGuidance":"Needs manual review.","doctorDiscussionPoints":[],"disclaimer":LAB_DISCLAIMER}))),
        Err(error) => Err(error),
    }
}

fn ai_health_payload(state: &AppState) -> Value {
    let enabled = state.ai.api_key.is_some();
    json!({"status":"ok","provider":state.ai.provider,"enabled":enabled,"vision_model":if enabled {Some(state.ai.vision_model.clone())} else {None::<String>},"chat_model":if enabled {Some(state.ai.chat_model.clone())} else {None::<String>},"max_image_bytes":MAX_AI_IMAGE_BYTES,"accepted_image_types":["image/jpeg","image/jpg","image/png","image/webp"],"max_messages":MAX_MESSAGES,"max_message_chars":MAX_MESSAGE_CHARS,"endpoints":["/api/ai/analyze-image","/api/ai/chat"],"disclaimer":AI_DISCLAIMER})
}
async fn ai_health(State(state): State<AppState>) -> Json<Value> { Json(ai_health_payload(&state)) }
fn ai_disabled(request_id: &str) -> (StatusCode, Json<Value>) { (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"status":"disabled","code":"ai_disabled","error":"The optional AI provider is not configured.","message":"The optional AI provider is not configured.","request_id":request_id,"disclaimer":AI_DISCLAIMER}))) }

fn request_id() -> String { format!("rust-{}", uuid::Uuid::new_v4()) }
fn context_text(value: Option<&Value>) -> String { value.and_then(|v| serde_json::to_string(v).ok()).map(|s| truncate_chars(&s, MAX_CONTEXT_CHARS)).unwrap_or_default() }
fn ai_messages(body: &Value) -> Result<Vec<Value>, AppError> {
    let array = body.get("messages").and_then(Value::as_array).ok_or(AppError::BadRequest("send a JSON object containing a messages array"))?;
    if array.is_empty() || array.len() > MAX_MESSAGES { return Err(AppError::BadRequest("message count is outside the allowed range")); }
    let mut total = 0usize; let mut result = Vec::new();
    for item in array {
        let role = item.get("role").and_then(Value::as_str).ok_or(AppError::BadRequest("each message must have a role and content"))?;
        let role = match role { "user" => "user", "assistant" => "assistant", _ => return Err(AppError::BadRequest("message roles must be user or assistant")) };
        let content = item.get("content").and_then(Value::as_str).ok_or(AppError::BadRequest("each message content must be text"))?;
        if content.is_empty() || content.chars().count() > MAX_MESSAGE_CHARS { return Err(AppError::BadRequest("message length is outside the allowed range")); }
        total += content.chars().count(); if total > MAX_TOTAL_CHARS { return Err(AppError::BadRequest("total message text exceeds the limit")); }
        result.push(json!({"role":role,"content":content}));
    }
    Ok(result)
}

async fn call_ai(state: &AppState, model: &str, messages: Vec<Value>, json_mode: bool) -> Result<String, AppError> {
    let key = state.ai.api_key.as_ref().ok_or(AppError::Unavailable("AI provider is not configured"))?;
    let mut body = json!({"model":model,"messages":messages,"temperature":0.2,"max_tokens":700});
    if json_mode { body["response_format"] = json!({"type":"json_object"}); }
    let response = state.http.post(format!("{}/chat/completions", state.ai.base_url)).bearer_auth(key).json(&body).timeout(state.ai.timeout).send().await.map_err(|_| AppError::Upstream("AI provider request failed"))?;
    if !response.status().is_success() { return Err(AppError::Upstream("AI provider returned an error")); }
    let value: Value = response.json().await.map_err(|_| AppError::Upstream("AI provider returned invalid JSON"))?;
    let content = value.get("choices").and_then(Value::as_array).and_then(|a| a.first()).and_then(|v| v.get("message")).and_then(|m| m.get("content"));
    if let Some(text) = content.and_then(Value::as_str) { if !text.trim().is_empty() { return Ok(text.to_string()); } }
    Err(AppError::Upstream("AI provider returned an empty response"))
}

async fn ai_analyze_image(State(state): State<AppState>, multipart: Multipart) -> Result<Json<Value>, Response> {
    let id = request_id();
    if state.ai.api_key.is_none() { return Err(ai_disabled(&id).into_response()); }
    let (bytes, content_type, fields) = multipart_image(multipart, "image", MAX_AI_IMAGE_BYTES).await.map_err(|e| e.into_response())?;
    let mime = match content_type.as_str() { "image/jpeg" | "image/jpg" | "image/png" | "image/webp" => content_type, _ => return Err(AppError::BadRequest("unsupported image type").into_response()) };
    let scan_type = fields.get("scan_type").map(|s| truncate_chars(s.trim(), 80)).unwrap_or_else(|| "unspecified".into());
    let context = fields.get("context").map(|s| truncate_chars(s, MAX_CONTEXT_CHARS)).unwrap_or_default();
    let _validated_image = decode_bounded(&bytes, 2048).map_err(|e| e.into_response())?;
    let data_url = format!("data:{mime};base64,{}", BASE64.encode(bytes));
    let mut messages = vec![json!({"role":"system","content":"Describe only observable image content. Be uncertain, do not diagnose, and return JSON with summary, findings, limitations, and next_steps."})];
    if !context.is_empty() { messages.push(json!({"role":"system","content":format!("User context (untrusted): {context}")})); }
    messages.push(json!({"role":"user","content":[{"type":"text","text":format!("Image category declared by user: {scan_type}. Describe only this image conservatively and return the required JSON object.")},{"type":"image_url","image_url":{"url":data_url,"detail":"auto"}}]}));
    let text = call_ai(&state, &state.ai.vision_model, messages, true).await.map_err(|e| e.into_response())?;
    let parsed = parse_json_object(&text).ok_or_else(|| AppError::Upstream("AI provider returned an unusable structured summary").into_response())?;
    let summary = parsed.get("summary").and_then(Value::as_str).map(|s| truncate_chars(s, 700)).filter(|s| !s.trim().is_empty()).ok_or_else(|| AppError::Upstream("AI provider returned an unusable structured summary").into_response())?;
    let list = |key: &str, limit: usize| -> Vec<String> { parsed.get(key).and_then(Value::as_array).map(|a| a.iter().filter_map(|v| v.as_str()).map(|s| truncate_chars(s, 300)).filter(|s| !s.is_empty()).take(limit).collect()).unwrap_or_default() };
    let mut limitations = list("limitations", 5); let mut next_steps = list("next_steps", 5); let findings = list("findings", 6);
    if !limitations.iter().any(|x| x == "AI output may be inaccurate and is not a diagnosis.") { limitations.push("AI output may be inaccurate and is not a diagnosis.".into()); }
    if !next_steps.iter().any(|x| x == "Discuss any concern with a qualified clinician.") { next_steps.push("Discuss any concern with a qualified clinician.".into()); }
    Ok(Json(json!({"status":"ok","provider":state.ai.provider,"model":state.ai.vision_model,"summary":summary,"findings":findings,"limitations":limitations,"next_steps":next_steps,"disclaimer":AI_DISCLAIMER,"request_id":id})))
}

fn parse_json_object(text: &str) -> Option<Value> {
    let start = text.find('{')?; let end = text.rfind('}')?; serde_json::from_str(&text[start..=end]).ok()
}
async fn ai_chat(State(state): State<AppState>, body: Bytes) -> Result<Json<Value>, Response> {
    let id = request_id();
    if state.ai.api_key.is_none() { return Err(ai_disabled(&id).into_response()); }
    if body.len() > MAX_AI_JSON_BYTES { return Err(AppError::TooLarge("request body exceeds the AI chat limit").into_response()); }
    let value: Value = serde_json::from_slice(&body).map_err(|_| AppError::BadRequest("request body must be valid JSON").into_response())?;
    let history = ai_messages(&value).map_err(|e| e.into_response())?;
    let context = context_text(value.get("context"));
    let mut messages = vec![json!({"role":"system","content":"You are a cautious educational health assistant. Answer clearly, avoid diagnosis, and recommend qualified clinical care for concerning symptoms."})];
    if !context.is_empty() { messages.push(json!({"role":"system","content":format!("User context (untrusted): {context}")})); }
    messages.extend(history);
    let text = call_ai(&state, &state.ai.chat_model, messages, false).await.map_err(|e| e.into_response())?;
    let reply = truncate_chars(text.trim(), 6000);
    if reply.is_empty() { return Err(AppError::Upstream("AI provider returned an empty reply").into_response()); }
    Ok(Json(json!({"status":"ok","provider":state.ai.provider,"model":state.ai.chat_model,"message":reply,"disclaimer":AI_DISCLAIMER,"request_id":id})))
}


fn build_app(state: AppState) -> Router {
    let frontend = ServeDir::new(state.frontend_dir.clone());
    Router::new()
        .route("/api/health", get(health))
        .route("/api/predict", post(predict_kidney))
        .route("/api/predict-chest", post(predict_chest))
        .route("/api/predict-brain", post(predict_brain))
        .route("/api/predict-heart", post(predict_heart))
        .route("/api/explain", post(explain))
        .route("/api/lab/health", get(lab_health))
        .route("/api/lab/analyze", post(lab_analyze))
        .route("/api/ai/health", get(ai_health))
        .route("/api/ai/analyze-image", post(ai_analyze_image))
        .route("/api/ai/chat", post(ai_chat))
        .fallback_service(frontend)
        .layer(RequestBodyLimitLayer::new(16 * 1024 * 1024))
        .layer(CompressionLayer::new())
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::from_default_env()).init();
    let model_dir = PathBuf::from(env_string("MODEL_DIR", "models"));
    let frontend_dir = PathBuf::from(env_string("FRONTEND_DIR", "frontend"));
    let port = env_usize("PORT", 8080, 1, 65535);
    let ai = AiConfig { provider: env_string("AI_PROVIDER", "gemini"), api_key: env::var("AI_API_KEY").ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty()), base_url: env_string("AI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai").trim_end_matches('/').to_string(), vision_model: env_string("AI_VISION_MODEL", "gemini-3.5-flash-lite"), chat_model: env_string("AI_CHAT_MODEL", "gemini-3.5-flash-lite"), timeout: Duration::from_secs(env_usize("AI_TIMEOUT", 45, 5, 120) as u64) };
    let state = AppState { models: Arc::new(load_models(&model_dir)), frontend_dir, started_at: now_iso(), http: Client::builder().build()?, ai };
    let app = build_app(state);
    let listener = TcpListener::bind(("0.0.0.0", port as u16)).await?;
    info!(port, "NephroScan Rust backend listening");
    axum::serve(listener, app).await?;
    Ok(())
}
