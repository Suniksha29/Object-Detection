# Object Detector & Counter

A production-ready web application that detects real-world objects in images, annotates them with labeled bounding boxes, and delivers both an aggregate count and an itemized per-class breakdown.

Powered by **FastAPI** and pretrained **Apache-2.0** models from **Hugging Face Transformers** (**RT-DETR** for real-time COCO 80-class detection and **OWLv2** for zero-shot open-vocabulary detection).

---

## Features

- **Dual Detection Engines**:
  - **Standard Mode (RT-DETR)**: Pretrained Real-Time DEtection TRansformer (`PekingU/rtdetr_r50vd`, Apache-2.0) detecting 80 standard COCO classes.
  - **Open-Vocabulary Mode (OWLv2)**: Vision-language zero-shot detector (`google/owlv2-base-patch16-ensemble`, Apache-2.0) that detects arbitrary custom text prompts (e.g., *"helmet, laptop, coffee mug, dog, backpack"*).
- **Intelligent Deduplication**: Pure PyTorch batched **Non-Maximum Suppression (NMS)** with configurable IoU and confidence thresholds so no object is double-counted.
- **Robust Validation**: Rejects files larger than 10 MB (HTTP 413), unsupported MIME types or extensions (HTTP 400), empty images (HTTP 400), and corrupt image streams (HTTP 400).
- **Modern Responsive Frontend**:
  - Drag-and-drop upload zone with live image thumbnail and metadata.
  - Live webcam / mobile camera capture modal with device switching.
  - Real-time confidence threshold (0.10 – 0.95) and NMS IoU threshold (0.10 – 0.90) sliders.
  - One-click sample test images for rapid demonstration.
  - Big **Total Objects: N** metric badge, per-class breakdown chips, annotated image viewer with download, and full detection coordinates table `[x1, y1, x2, y2]`.
- **Hardware Acceleration**: Automatic CUDA GPU acceleration when available, with automatic fallback to CPU.
- **Lifespan Model Loading**: Deep learning models are loaded into memory once during application startup, never on a per-request basis.

---

## Project Structure

```text
Object detection task/
├── app/
│   ├── __init__.py
│   ├── api.py             # Route handlers: POST /detect, GET /health
│   ├── config.py          # App settings (thresholds, max file size, MIME types)
│   ├── main.py            # FastAPI initialization, lifespan manager, static mount
│   ├── schemas.py         # Pydantic schemas (DetectionItem, DetectResponse, etc.)
│   └── utils.py           # Image validation, PIL drawing, Base64 encoding
├── models/
│   ├── __init__.py
│   ├── base.py            # Abstract BaseDetector interface & DetectionResult
│   ├── manager.py         # DetectorManager handling model lifecycles and dispatch
│   ├── nms.py             # Pure PyTorch vectorized Non-Maximum Suppression
│   ├── owlv2_detector.py  # OWLv2 Open-Vocabulary detector implementation
│   └── rtdetr_detector.py # RT-DETR ResNet-50 COCO detector implementation
├── static/
│   ├── app.js             # Client-side state, camera stream, API calls, rendering
│   ├── index.html         # Responsive single-page application UI
│   ├── styles.css         # Modern dark-mode styling system
│   └── sample_images/     # Photographic sample images for immediate testing
│       ├── living_room.jpg
│       ├── street.jpg
│       └── workspace.jpg
├── tests/
│   ├── __init__.py
│   ├── conftest.py        # Pytest fixtures and test client
│   └── test_api.py        # 12 automated unit and integration tests
├── Dockerfile             # Production-ready container definition
├── .dockerignore
├── requirements.txt       # Project dependencies
├── run.py                 # CLI server launcher
└── README.md              # Documentation and accuracy notes
```

---

## Installation & Setup

### Prerequisites

- Python 3.11 or 3.12
- `pip` package manager
- (Optional) NVIDIA GPU with CUDA drivers for acceleration

### 1. Clone & Set Up Virtual Environment

```bash
# Clone the repository
git clone <repository_url>
cd "Object detection task"

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Application

### Option A: Using the CLI Runner

```bash
python run.py --host 127.0.0.1 --port 8000 --reload
```

### Option B: Using Uvicorn Directly

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser at **`http://127.0.0.1:8000`** to access the web application. Interactive OpenAPI documentation is available at **`http://127.0.0.1:8000/docs`**.

---

## Running with Docker

Build and run the container:

```bash
# Build Docker image
docker build -t object-detector .

# Run container on port 8000
docker run -p 8000:8000 object-detector
```

Access the UI at `http://localhost:8000`.

---

## Cloud Deployment Guide

This project is pre-configured for one-click deployment across major cloud providers:

### 1. Deploy on Render
1. Go to [render.com](https://render.com) and create an account.
2. Click **New +** &rarr; **Web Service** and connect your GitHub repository (`Suniksha29/Object-Detection`).
3. Render automatically detects the included `render.yaml` or `Dockerfile`:
   - **Runtime**: `Docker` or `Python 3`
   - **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Click **Deploy Web Service**.

### 2. Deploy on Railway
1. Go to [railway.app](https://railway.app) and sign in with GitHub.
2. Click **New Project** &rarr; **Deploy from GitHub repo** &rarr; select `Suniksha29/Object-Detection`.
3. Railway automatically detects `Dockerfile` and the included `Procfile`, binds to `$PORT`, and provisions a live HTTPS URL.

### 3. Deploy on Hugging Face Spaces (Free CPU / GPU)
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Select **Docker** as the Space SDK and choose **Blank**.
3. In your Space settings, connect your GitHub repository or mirror the code to HF Spaces.
4. HF Spaces automatically builds the `Dockerfile` and hosts the interactive app.

### 4. Deploy on Google Cloud Run
```bash
# Build and submit to Google Artifact Registry
gcloud builds submit --tag gcr.io/<PROJECT_ID>/object-detector

# Deploy serverless container to Cloud Run
gcloud run deploy object-detector \
  --image gcr.io/<PROJECT_ID>/object-detector \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```

---

## API Reference

### 1. Health Check

**Endpoint:** `GET /health`  
Returns operational status, device (CPU or CUDA), and loaded model identifiers.

#### Response Example (`200 OK`):

```json
{
  "status": "ok",
  "device": "cpu",
  "models": {
    "standard_detector": "PekingU/rtdetr_r50vd",
    "open_vocabulary_detector": "google/owlv2-base-patch16-ensemble"
  },
  "version": "1.0.0"
}
```

---

### 2. Detect & Count Objects

**Endpoint:** `POST /detect`  
Accepts a multipart form submission containing an image and inference parameters.

#### Form Parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | Binary File | *Required* | JPG, PNG, or WEBP image file (Max 10 MB). |
| `threshold` | Float | `0.50` | Minimum confidence score threshold (`0.01` – `1.00`). |
| `nms_threshold` | Float | `0.50` | IoU threshold for Non-Maximum Suppression deduplication (`0.01` – `1.00`). |
| `mode` | String | `"standard"` | `"standard"` (COCO 80 classes) or `"open_vocab"` (custom text queries). |
| `custom_labels`| String | `null` | Comma-separated custom labels for open-vocabulary mode (e.g. `"helmet, laptop, bottle"`). |

#### cURL Example:

```bash
curl -X POST "http://127.0.0.1:8000/detect" \
  -F "file=@sample.jpg;type=image/jpeg" \
  -F "threshold=0.5" \
  -F "nms_threshold=0.5" \
  -F "mode=standard"
```

#### Open-Vocabulary cURL Example:

```bash
curl -X POST "http://127.0.0.1:8000/detect" \
  -F "file=@office.jpg;type=image/jpeg" \
  -F "threshold=0.2" \
  -F "mode=open_vocab" \
  -F "custom_labels=keyboard, coffee mug, monitor, notebook"
```

#### Response Example (`200 OK`):

```json
{
  "total": 5,
  "counts": {
    "person": 2,
    "chair": 2,
    "laptop": 1
  },
  "detections": [
    {
      "label": "person",
      "confidence": 0.9412,
      "box": [124.5, 88.2, 310.0, 480.0]
    },
    {
      "label": "person",
      "confidence": 0.9105,
      "box": [450.1, 102.4, 620.8, 510.3]
    },
    {
      "label": "laptop",
      "confidence": 0.8847,
      "box": [220.0, 310.5, 380.2, 440.0]
    },
    {
      "label": "chair",
      "confidence": 0.7632,
      "box": [100.0, 350.0, 240.0, 520.0]
    },
    {
      "label": "chair",
      "confidence": 0.7128,
      "box": [420.0, 360.0, 550.0, 530.0]
    }
  ],
  "annotated_image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ..."
}
```

---

## Running the Automated Test Suite

The test suite validates the endpoints against:
1. Valid JPEG uploads.
2. Valid PNG uploads.
3. Valid WEBP uploads.
4. Invalid file formats (`.txt`, `.pdf`).
5. Empty (0-byte) image uploads.
6. Oversized uploads exceeding 10 MB.
7. Corrupted image payloads.
8. Threshold bounds validation.
9. System `/health` endpoint.
10. Open-vocabulary detection with custom labels.
11. Real-world sample image object verification.

Run pytest:

```bash
pytest tests/ -v
```

Expected output:
```text
============================= test session starts =============================
collected 12 items

tests/test_api.py::test_health_endpoint PASSED                           [  8%]
tests/test_api.py::test_root_endpoint PASSED                             [ 16%]
tests/test_api.py::test_detect_valid_jpeg PASSED                         [ 25%]
tests/test_api.py::test_detect_valid_png PASSED                          [ 33%]
tests/test_api.py::test_detect_valid_webp PASSED                         [ 41%]
tests/test_api.py::test_detect_invalid_file_type PASSED                  [ 50%]
tests/test_api.py::test_detect_empty_file PASSED                         [ 58%]
tests/test_api.py::test_detect_oversized_file PASSED                     [ 66%]
tests/test_api.py::test_detect_corrupt_file PASSED                       [ 75%]
tests/test_api.py::test_detect_threshold_validation PASSED               [ 83%]
tests/test_api.py::test_detect_open_vocabulary_mode PASSED               [ 91%]
tests/test_api.py::test_detect_sample_image_detection PASSED             [100%]

======================= 12 passed in 21.45s =======================
```

---

## Model Documentation, Limitations & Accuracy Notes

### 1. Default Model: RT-DETR (COCO 80 Classes)

The default detector uses **RT-DETR** (`PekingU/rtdetr_r50vd`), an Apache-2.0 licensed end-to-end transformer-based object detector that outperforms earlier YOLO and DETR architectures in both speed and mAP.

#### Supported COCO Classes (80 Classes):
> **People & Vehicles:** `person`, `bicycle`, `car`, `motorcycle`, `airplane`, `bus`, `train`, `truck`, `boat`  
> **Outdoor & Street:** `traffic light`, `fire hydrant`, `stop sign`, `parking meter`, `bench`  
> **Animals:** `bird`, `cat`, `dog`, `horse`, `sheep`, `cow`, `elephant`, `bear`, `zebra`, `giraffe`  
> **Accessories:** `backpack`, `umbrella`, `handbag`, `tie`, `suitcase`, `frisbee`, `skis`, `snowboard`, `sports ball`, `kite`, `baseball bat`, `baseball glove`, `skateboard`, `surfboard`, `tennis racket`  
> **Kitchen & Dining:** `bottle`, `wine glass`, `cup`, `fork`, `knife`, `spoon`, `bowl`, `banana`, `apple`, `sandwich`, `orange`, `broccoli`, `carrot`, `hot dog`, `pizza`, `donut`, `cake`  
> **Furniture & Household:** `chair`, `couch`, `potted plant`, `bed`, `dining table`, `toilet`, `tv`, `laptop`, `mouse`, `remote`, `keyboard`, `cell phone`, `microwave`, `oven`, `toaster`, `sink`, `refrigerator`, `book`, `clock`, `vase`, `scissors`, `teddy bear`, `hair drier`, `toothbrush`

### 2. Known Model Limitations

1. **Fixed Class Vocabulary (Standard Mode)**:
   - In standard mode, the model can only recognize the 80 classes present in the COCO dataset. Real-world objects outside this taxonomy (e.g., hardhats, safety vests, microchips, medical instruments) will be either ignored or misclassified into the nearest visual surrogate (e.g. a helmet might be detected as a `sports ball` or ignored).
   - *Mitigation*: Switch to the app's built-in **Open-Vocabulary Mode (OWLv2)** to supply arbitrary class names.

2. **Small or Distant Objects**:
   - Objects occupying fewer than ~16×16 pixels often lose critical feature representations due to feature downsampling in convolutional backbones and patch tokenization.
   - Tiny objects in high-altitude drone shots or wide-angle surveillance cameras may be missed.

3. **Severe Occlusion & Crowded Scenes**:
   - When multiple objects heavily overlap (e.g., a person carrying another person, or a chair partially hidden behind a desk), the detector may predict overlapping bounding boxes that either get suppressed by NMS or fail to reach the confidence threshold.

4. **Extreme Lighting & Domain Shift**:
   - Models trained primarily on daytime internet photographs degrade in performance when exposed to severe domain shifts: night-time infrared, thermal cameras, aerial viewpoints, or specialized microscopy.

5. **Open-Vocabulary Prompt Sensitivity (OWLv2)**:
   - Vision-language models like OWLv2 match image patches to text embeddings produced by CLIP. Slight variations in prompt phrasing (e.g., `"safety helmet"` vs `"hardhat"` vs `"construction helmet"`) can noticeably influence detection confidence.

---

### 3. How to Improve Accuracy in Production

1. **Use Larger Model Checkpoints**:
   - The default model uses a ResNet-50 backbone (`rtdetr_r50vd`) for balanced speed and CPU friendliness. For maximum accuracy on GPU servers, upgrade to larger backbones such as **RT-DETR-X** (`PekingU/rtdetr_xvd`) or **OWLv2-Large** (`google/owlv2-large-patch14-ensemble`), which increase mAP by +4–6 points.

2. **High-Resolution Inference & Tiling (SAHI)**:
   - For images with small objects (e.g. 4K camera streams), pass higher resolution inputs or apply **Slicing Aided Hyper Inference (SAHI)**. SAHI crops high-resolution images into overlapping tiles, runs inference on each tile at full resolution, and merges bounding boxes using NMS.

3. **Calibrate Thresholds Per Domain**:
   - **Confidence Threshold**: Lower the threshold to `0.25 - 0.35` for dense environments to increase recall, or raise to `0.65+` when false positives are expensive.
   - **NMS Threshold**: Adjust the IoU threshold from `0.50` to `0.35` if overlapping duplicate boxes appear, or raise it to `0.65` if closely stacked objects (e.g. stacked chairs or dense crowds) are being suppressed.

4. **Prompt Engineering for Open-Vocabulary Mode**:
   - Instead of single-word queries, supply descriptive natural phrases such as:
     - `"a photo of a yellow safety helmet"`
     - `"a black wireless computer mouse"`
     - `"a ceramic coffee mug"`

5. **Fine-Tuning on Custom Domain Datasets**:
   - For specialized industrial or enterprise applications (e.g. factory QA, retail shelf scanning, autonomous driving), collect 500–2,000 domain-specific labeled images and fine-tune the pretrained RT-DETR model using the Hugging Face `Trainer` API. Transfer learning converges rapidly (often in 20–50 epochs) while retaining pretrained visual representations.

---

## License

This software is released under the **Apache-2.0 License**.  
Pretrained model weights (**PekingU/rtdetr_r50vd** and **google/owlv2-base-patch16-ensemble**) are distributed under the **Apache-2.0 License**.

