# NephroScan AI — Demo Script

## Opening (30 seconds)

"NephroScan is an AI-assisted medical imaging screening prototype. It uses deep learning to classify medical images across four modalities: kidney stones, chest pneumonia, brain MRI, and heart cardiomegaly."

## Live Demo Flow

### 1. Dashboard Overview (1 minute)
- Show the dashboard with model status indicators
- Point out all 4 models loaded and ready
- Navigate through views to show the interface

### 2. Run an Analysis (2 minutes)
- Go to New Analysis
- Upload a demo image (kidney CT, chest X-ray, brain MRI, or heart X-ray)
- Select scan type and enter clinical indication
- Click Run Analysis
- Show the result with confidence score and provenance label
- Point out: "Every result shows the model used, timestamp, and inference type"

### 3. Grad-CAM Explainability (1 minute)
- Toggle Explainability Overlay on the result
- Show the attention heatmap
- Explain: "This shows where the model focused — it's not a diagnosis or segmentation"

### 4. Lab Test Analysis (2 minutes)
- Go to Lab Test Analysis
- Upload a lab report and add the values in the editable table
- Point out: "Extraction is manual in this build — the table is the source of truth"
- Generate the report and walk through the guidance sections

### 5. AI Assistant (1 minute)
- Open the assistant
- Ask "Explain my result"
- Show the AI-generated explanation with disclaimer
- Ask "What should I ask my doctor?"
- Show the suggested questions

## Safety Talking Points

- "This is an educational prototype, not a diagnostic device"
- "Every result includes provenance metadata for transparency"
- "Grad-CAM shows model attention, not a segmentation or a measurement"
- "All results require clinical validation by a qualified professional"
- "No patient data is stored — everything stays in the browser session"

## Closing (30 seconds)

"NephroScan demonstrates how AI can assist medical imaging screening while maintaining transparency through provenance labels and explainability. The system is designed for deployment on standard web infrastructure with no specialized hardware."

## Troubleshooting

- **Models not loading:** verify `MODEL_DIR` holds `kidney`, `chest`, `brain`, and `heart` `.onnx` graphs with their `.json` manifests and `models.json`
- **AI Assistant unavailable:** `AI_API_KEY` is unset, so `/api/ai/*` is disabled and the drawer shows the outage notice; local analysis still works
- **Slow inference:** First run may be slower as models warm up
