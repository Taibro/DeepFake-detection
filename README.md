#  Deepfake Scanner 

A real-time Deepfake detection system that analyzes video, webcam feeds, and screen captures. The project consists of a FastAPI-based backend that handles inference and a React-Vite frontend for the user interface.

## Models and Algorithms

The core of the deepfake detection engine relies on a **Deepfake Fusion Model**, which combines visual features and biometric signals using a spatial-temporal attention mechanism. 

The architecture consists of:
1. **Visual Feature Extractor**: Utilizes a **Swin Transformer** (`swin_tiny_patch4_window7_224`) to analyze the spatial artifacts and anomalies often found in synthesized faces.
2. **Biometric Feature Extractor**: Uses a custom **rPPG (remote photoplethysmography) Convolutional Neural Network (CNN)**. Deepfakes often lack consistent blood flow patterns (micro-color changes in the skin), which this network attempts to capture.
3. **Attention Fusion Module**: Concatenates visual and biometric embeddings, passing them through an attention layer to weight the importance of each feature type before the final classification.
4. **Face Detection**: Driven by **Google MediaPipe Face Detection**, ensuring fast and robust face localization in real-time streams.

## Getting Started

Follow the instructions below to set up both the backend and frontend environments.

### 1. Download the Pre-trained Model

Before running the backend, you need to download the trained model weights.

1. Go to this [Google Drive Link](https://drive.google.com/drive/folders/1dVTP9NouoMxArE4al2q4fLJ92mVq_6dx?hl=vi).
2. Download the latest model file, specifically: `fusion_model_ffpp_v2_epoch_20.pth`.
3. Place the downloaded `.pth` file directly into the `backend/` directory.

> **Note:** The backend will still run without the weights (initializing with random weights), but it will not accurately detect deepfakes.

---

### 2. Backend Setup

The backend uses [uv](https://github.com/astral-sh/uv), an extremely fast Python package and project manager written in Rust.

#### Installing `uv`
If you haven't installed `uv` yet, you can do so via `pip` or standard installation scripts:
- **Windows (PowerShell):**
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- **macOS/Linux:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- *Alternatively, via pip:* `pip install uv`

#### Running the Backend
Navigate to the `backend` folder, sync the dependencies, and start the FastAPI server:

```bash
cd backend

# Sync dependencies and create a virtual environment automatically
uv sync

# Run the backend server using uvicorn
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
The backend API will be available at `http://localhost:8000`.

---

### 3. Frontend Setup

The frontend is built with React and Vite. You'll need [Node.js](https://nodejs.org/) installed.

#### Running the Frontend
Navigate to the `frontend` folder, install the packages, and start the development server:

```bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```
The frontend application will be available at `http://localhost:5173` (or the port specified by Vite in your console).

## Testing the Application (Test Images/Videos)

Once both the backend and frontend are running:
1. Open your browser and go to the frontend URL.
2. **Video Upload**: You can upload standard `mp4` videos (real or deepfake) to test the frame-by-frame analysis.
3. **Webcam**: Click on the webcam option to test real-time inference on your face.
4. **Screen Capture / Window Capture**: Select a specific window (e.g., a playing YouTube video or Zoom call) to analyze a live stream.

*(If you need sample test images or videos, you can generally use any face video or look up standard datasets like FaceForensics++ or Celeb-DF to try it out).*
