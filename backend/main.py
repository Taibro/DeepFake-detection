"""
HUIT Deepfake Scanner - FastAPI Backend
Run: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import base64
import collections
import io
import os
import tempfile
import time
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms
import mediapipe as mp
import pygetwindow as gw
import mss

# ──────────────────────────────────────────────
# 1. APP INIT
# ──────────────────────────────────────────────
app = FastAPI(title="HUIT Deepfake Scanner API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# 2. MODEL ARCHITECTURE
# ──────────────────────────────────────────────
class DeepfakeFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224", pretrained=False, num_classes=0
        )
        self.rppg_net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.attention_weights = nn.Sequential(
            nn.Linear(768 + 64, 768 + 64), nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(768 + 64, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(128, 1), nn.Sigmoid(),
        )

    def forward(self, image_input, rppg_input):
        visual = self.swin(image_input)
        bio = torch.flatten(self.rppg_net(rppg_input), 1)
        combined = torch.cat((visual, bio), dim=1)
        att = self.attention_weights(combined)
        return self.classifier(combined * att)


# ──────────────────────────────────────────────
# 3. GLOBAL MODEL SINGLETON
# ──────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL: Optional[DeepfakeFusionModel] = None
MODEL_LOADED = False

SWIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
RPPG_TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

MP_FACE = mp.solutions.face_detection


def get_model() -> DeepfakeFusionModel:
    global MODEL, MODEL_LOADED
    if MODEL is None:
        MODEL = DeepfakeFusionModel().to(DEVICE)
        weights_path = "fusion_model_ffpp_v2_epoch_20.pth"
        if os.path.exists(weights_path):
            MODEL.load_state_dict(torch.load(weights_path, map_location=DEVICE))
            MODEL_LOADED = True
            print("✅ Loaded trained weights.")
        else:
            MODEL_LOADED = False
            print("⚠️  Running with random weights (no .pth found).")
        MODEL.eval()
    return MODEL


@app.on_event("startup")
async def startup():
    get_model()


# ──────────────────────────────────────────────
# 4. INFERENCE HELPER
# ──────────────────────────────────────────────
def infer_face(face_rgb: np.ndarray) -> float:
    """Return probability 0-1 of deepfake."""
    model = get_model()
    pil = Image.fromarray(face_rgb)
    vis = SWIN_TRANSFORM(pil).unsqueeze(0).to(DEVICE)
    rppg = RPPG_TRANSFORM(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return model(vis, rppg).item()


def frame_to_base64(frame_bgr: np.ndarray, quality: int = 75) -> str:
    """Encode a BGR frame as JPEG base64 string."""
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("utf-8")


def detect_and_analyze(frame_bgr: np.ndarray, face_detection, history: collections.deque,
                       frame_count: int, smoothed_prob: float):
    """
    Run face detection + AI on frame.
    Returns (annotated_frame, smoothed_prob, face_found, detections_list)
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)
    detections = []

    if results.detections:
        ih, iw = frame_bgr.shape[:2]
        for det in results.detections:
            bb = det.location_data.relative_bounding_box
            x, y = int(bb.xmin * iw), int(bb.ymin * ih)
            w, h = int(bb.width * iw), int(bb.height * ih)
            px, py = int(w * 0.15), int(h * 0.2)
            x1, y1 = max(0, x - px), max(0, y - py * 2)
            x2, y2 = min(iw, x + w + px), min(ih, y + h + py)

            face_crop = rgb[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            if frame_count % 3 == 0:
                raw = infer_face(face_crop)
                history.append(raw)
                smoothed_prob = sum(history) / len(history)

            is_fake = smoothed_prob > 0.5
            pct = smoothed_prob * 100 if is_fake else (1 - smoothed_prob) * 100
            label = f"{'DEEPFAKE' if is_fake else 'REAL'}: {pct:.1f}%"
            color = (0, 0, 220) if is_fake else (0, 200, 80)

            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(frame_bgr, (x1, y1 - 28), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame_bgr, label, (x1 + 2, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            detections.append({
                "is_fake": is_fake,
                "probability": round(smoothed_prob, 4),
                "confidence": round(pct, 1),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            })

    return frame_bgr, smoothed_prob, len(detections) > 0, detections


# ──────────────────────────────────────────────
# 5. REST ENDPOINTS
# ──────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL_LOADED,
        "device": str(DEVICE),
    }


@app.get("/api/windows")
def list_windows():
    """Return list of visible desktop windows suitable for capture."""
    # Thêm các từ khóa của các app hệ thống ẩn để loại trừ
    forbidden = {"HUIT Deepfake Scanner", "Program Manager", "Settings", "Taskbar", "Microsoft Text Input Application", ""}
    wins = []
    
    try:
        # Lấy tất cả cửa sổ
        for w in gw.getAllWindows():
            # Lọc điều kiện: Phải có tiêu đề (title), kích thước đủ lớn, và không nằm trong danh sách cấm
            if (
                w.title and w.title.strip() != ""
                and w.width > 100
                and w.height > 100
                and not any(f.lower() in w.title.lower() for f in forbidden if f)
            ):
                wins.append(w.title)
    except Exception as e:
        print(f"Lỗi khi quét cửa sổ: {e}")

    # Nếu vẫn không tìm thấy, thêm một lựa chọn fallback để test
    if not wins:
        wins = ["Không tìm thấy cửa sổ (Thử mở một thư mục hoặc Notepad)"]

    return {"windows": sorted(list(set(wins)))}


@app.post("/api/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    """
    Analyze an uploaded video file.
    Returns per-frame results + overall summary.
    """
    suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    history: collections.deque = collections.deque(maxlen=10)
    frame_results = []
    frame_count = 0
    smoothed_prob = 0.5

    face_det = MP_FACE.FaceDetection(model_selection=0, min_detection_confidence=0.65)

    # Sample at most 120 frames evenly
    sample_every = max(1, total_frames // 120)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % sample_every != 0:
            continue

        _, smoothed_prob, face_found, detections = detect_and_analyze(
            frame, face_det, history, frame_count, smoothed_prob
        )

        if face_found:
            frame_results.append({
                "frame": frame_count,
                "time_sec": round(frame_count / fps, 2),
                "detections": detections,
            })

    cap.release()
    os.unlink(tmp_path)
    face_det.close()

    if not frame_results:
        return JSONResponse({"error": "No faces detected in video."}, status_code=422)

    all_probs = [d["probability"] for f in frame_results for d in f["detections"]]
    avg_prob = sum(all_probs) / len(all_probs) if all_probs else 0.5
    verdict = "DEEPFAKE" if avg_prob > 0.5 else "REAL"

    return {
        "verdict": verdict,
        "average_probability": round(avg_prob, 4),
        "confidence": round((avg_prob if verdict == "DEEPFAKE" else 1 - avg_prob) * 100, 1),
        "total_frames_analyzed": len(frame_results),
        "frames": frame_results[:50],  # Return up to 50 sampled frames
    }


# ──────────────────────────────────────────────
# 6. WEBSOCKET – WEBCAM
# ──────────────────────────────────────────────
@app.websocket("/ws/webcam")
async def ws_webcam(ws: WebSocket):
    await ws.accept()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        await ws.send_json({"error": "Cannot open webcam"})
        await ws.close()
        return

    history: collections.deque = collections.deque(maxlen=10)
    frame_count = 0
    smoothed_prob = 0.5
    face_det = MP_FACE.FaceDetection(model_selection=0, min_detection_confidence=0.65)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            frame = cv2.flip(frame, 1)

            annotated, smoothed_prob, face_found, detections = detect_and_analyze(
                frame, face_det, history, frame_count, smoothed_prob
            )

            b64 = frame_to_base64(annotated, quality=60)
            await ws.send_json({
                "frame": b64,
                "face_found": face_found,
                "detections": detections,
                "smoothed_prob": round(smoothed_prob, 4),
            })
            await asyncio.sleep(0.03)  # ~30 fps cap

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Webcam WS error: {e}")
    finally:
        cap.release()
        face_det.close()


# ──────────────────────────────────────────────
# 7. WEBSOCKET – WINDOW CAPTURE
# ──────────────────────────────────────────────
@app.websocket("/ws/window")
async def ws_window(ws: WebSocket):
    await ws.accept()
    data = await ws.receive_json()
    window_title: str = data.get("window_title", "")

    sct = mss.mss()
    history: collections.deque = collections.deque(maxlen=10)
    frame_count = 0
    smoothed_prob = 0.5
    face_det = MP_FACE.FaceDetection(model_selection=0, min_detection_confidence=0.65)

    try:
        while True:
            try:
                wins = gw.getWindowsWithTitle(window_title)
                if not wins:
                    await ws.send_json({"error": f"Window '{window_title}' not found"})
                    break
                win = wins[0]
                if win.isMinimized:
                    await asyncio.sleep(0.5)
                    continue

                monitor = {
                    "top": win.top, "left": win.left,
                    "width": win.width, "height": win.height,
                }
                sct_img = sct.grab(monitor)
                frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
            except Exception as e:
                await ws.send_json({"error": str(e)})
                break

            frame_count += 1
            annotated, smoothed_prob, face_found, detections = detect_and_analyze(
                frame, face_det, history, frame_count, smoothed_prob
            )

            b64 = frame_to_base64(annotated, quality=55)
            await ws.send_json({
                "frame": b64,
                "face_found": face_found,
                "detections": detections,
                "smoothed_prob": round(smoothed_prob, 4),
            })
            await asyncio.sleep(0.05)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Window WS error: {e}")
    finally:
        face_det.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)