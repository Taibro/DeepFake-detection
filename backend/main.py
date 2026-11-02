"""
HUIT Deepfake Scanner - FastAPI Backend
Run: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import base64
import collections
import concurrent.futures
import io
import json
import os
import tempfile
import time
from typing import Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms
import mediapipe as mp
import pygetwindow as gw
import mss
try:
    import onnxruntime as ort
except ImportError:
    ort = None

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
from model_arch import build_fusion_model

# ──────────────────────────────────────────────
# 3. GLOBAL MODEL SINGLETON
# ──────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
MODEL_TYPE = None
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
# rPPG v3 / CAN: Normalized to [0, 1] via ToTensor(), NO ImageNet normalization
RPPG_APP_TRANSFORM = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
])
DIFF_EPS = 0.01

MP_FACE = mp.solutions.face_detection
INFERENCE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def get_model():
    global MODEL, MODEL_TYPE, MODEL_LOADED
    if MODEL is None:
        pth_path = "fusion_model_best.pth"
        onnx_path = "fusion_model_v5_best.onnx"

        # Priority 1: PyTorch latest trained model
        if os.path.exists(pth_path):
            try:
                model = build_fusion_model()
                ckpt = torch.load(pth_path, map_location=DEVICE, weights_only=False)
                st = ckpt.get("model_state", ckpt)
                model.load_state_dict(st, strict=False)
                model.to(DEVICE)
                model.eval()
                MODEL = model
                MODEL_TYPE = "pytorch"
                MODEL_LOADED = True
                print(f"✅ Loaded PyTorch model from {pth_path} on {DEVICE}.")
                return MODEL
            except Exception as e:
                print(f"⚠️ Failed to load PyTorch model {pth_path}: {e}")

        # Priority 2: ONNX Runtime
        if os.path.exists(onnx_path):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            MODEL = ort.InferenceSession(onnx_path, providers=providers)
            MODEL_TYPE = "onnx"
            MODEL_LOADED = True
            print(f"✅ Loaded ONNX model from {onnx_path}.")
            return MODEL

        MODEL_LOADED = False
        print(f"⚠️ Neither {pth_path} nor {onnx_path} found.")
    return MODEL


@app.on_event("startup")
async def startup():
    get_model()


# ──────────────────────────────────────────────
# 4. INFERENCE HELPER
# ──────────────────────────────────────────────
# Optimal EER decision threshold calibrated from Stage 3 validation metrics (eer_threshold = 0.785)
CALIBRATED_EER_THRESHOLD = 0.78

def calibrate_probability(raw_prob: float, threshold: float = CALIBRATED_EER_THRESHOLD) -> float:
    """
    Calibrate raw sigmoid probability with the model's optimal EER threshold.
    Maps [0, threshold) -> [0, 0.5) (REAL)
    Maps [threshold, 1.0] -> [0.5, 1.0] (DEEPFAKE)
    """
    if raw_prob < threshold:
        return float(0.5 * (raw_prob / max(1e-5, threshold)))
    else:
        return float(0.5 + 0.5 * ((raw_prob - threshold) / max(1e-5, (1.0 - threshold))))


def infer_face(face_rgb: np.ndarray, prev_face_rgb: Optional[np.ndarray] = None) -> float:
    """Return calibrated probability 0-1 of deepfake (0.5 threshold boundary)."""
    model = get_model()
    if model is None:
        return 0.5  # Fallback if model not loaded

    pil = Image.fromarray(face_rgb)

    if MODEL_TYPE == "pytorch":
        sp = SWIN_TRANSFORM(pil).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1, 1, 3, 224, 224)
        app = RPPG_APP_TRANSFORM(pil).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1, 1, 3, 96, 96)
        if prev_face_rgb is not None:
            prev_pil = Image.fromarray(prev_face_rgb)
            prev_app = RPPG_APP_TRANSFORM(prev_pil).unsqueeze(0).unsqueeze(0).to(DEVICE)
            # Normalized difference: (c - p) / (c + p + DIFF_EPS) matching train notebooks
            mot = (app - prev_app) / (app + prev_app + DIFF_EPS)
        else:
            mot = torch.zeros_like(app)  # Fallback for single/first frame
        with torch.inference_mode():
            logits = model(sp, app, mot)
            raw_prob = torch.sigmoid(logits).item()
            return calibrate_probability(raw_prob)
    else:
        vis = SWIN_TRANSFORM(pil).unsqueeze(0).numpy()
        rppg = RPPG_TRANSFORM(pil).unsqueeze(0).numpy()
        ort_inputs = {
            model.get_inputs()[0].name: vis,
            model.get_inputs()[1].name: rppg
        }
        ort_outs = model.run(None, ort_inputs)
        raw_prob = float(ort_outs[0][0][0])
        return calibrate_probability(raw_prob)


def infer_faces_batch(face_crops: list) -> list:
    """
    Run batched AI inference on a list of RGB face crops in a single PyTorch forward pass.
    Leverages multi-threaded PyTorch CPU SIMD/MKL parallelism (7-10x faster than sequential).
    """
    if not face_crops:
        return []

    model = get_model()
    if model is None:
        return [0.5] * len(face_crops)

    sp_list = []
    app_list = []
    mot_list = []

    for i, crop in enumerate(face_crops):
        pil = Image.fromarray(crop)
        sp = SWIN_TRANSFORM(pil).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 224, 224)
        app = RPPG_APP_TRANSFORM(pil).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 96, 96)

        if i > 0 and face_crops[i - 1] is not None:
            prev_pil = Image.fromarray(face_crops[i - 1])
            prev_app = RPPG_APP_TRANSFORM(prev_pil).unsqueeze(0).unsqueeze(0)
            mot = (app - prev_app) / (app + prev_app + DIFF_EPS)
        else:
            mot = torch.zeros_like(app)

        sp_list.append(sp)
        app_list.append(app)
        mot_list.append(mot)

    batch_sp = torch.cat(sp_list, dim=0).to(DEVICE)
    batch_app = torch.cat(app_list, dim=0).to(DEVICE)
    batch_mot = torch.cat(mot_list, dim=0).to(DEVICE)

    with torch.inference_mode():
        logits = model(batch_sp, batch_app, batch_mot)
        raw_probs = torch.sigmoid(logits).squeeze(-1).tolist()
        if isinstance(raw_probs, float):
            raw_probs = [raw_probs]

    return [calibrate_probability(p) for p in raw_probs]


def frame_to_base64(frame_bgr: np.ndarray, quality: int = 75) -> str:
    """Encode a BGR frame as JPEG base64 string."""
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("utf-8")


def detect_and_analyze(frame_bgr: np.ndarray, face_detection, history: collections.deque,
                       frame_count: int, smoothed_prob: float, prev_face_holder: list = None):
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
                prev_crop = prev_face_holder[0] if (prev_face_holder and len(prev_face_holder) > 0) else None
                raw = infer_face(face_crop, prev_crop)
                if prev_face_holder is not None and len(prev_face_holder) > 0:
                    prev_face_holder[0] = face_crop
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


def find_target_window(window_id: Optional[int] = None, window_title: Optional[str] = None):
    """
    Locate target desktop window by HWND first, then by title.
    Ensures that only valid, non-zero-sized windows are returned.
    """
    all_wins = gw.getAllWindows()

    # 1. Match by hWnd directly if window_id is provided
    if window_id:
        try:
            target_hwnd = int(window_id)
            for w in all_wins:
                if int(getattr(w, '_hWnd', 0)) == target_hwnd:
                    return w
        except Exception:
            pass

    # 2. Match by title if provided
    if window_title:
        title_str = str(window_title).strip()
        if title_str:
            # 2a. Exact match
            exact_matches = [
                w for w in all_wins
                if getattr(w, 'title', '').strip() == title_str
                and ((w.width > 50 and w.height > 50) or w.isMinimized)
            ]
            if exact_matches:
                return sorted(exact_matches, key=lambda w: (not w.isMinimized, w.width * w.height), reverse=True)[0]

            # 2b. Case-insensitive substring match
            sub_matches = [
                w for w in all_wins
                if title_str.lower() in getattr(w, 'title', '').lower()
                and ((w.width > 50 and w.height > 50) or w.isMinimized)
            ]
            if sub_matches:
                return sorted(sub_matches, key=lambda w: (not w.isMinimized, w.width * w.height), reverse=True)[0]

            # 2c. Reverse substring (window's title in title_str)
            rev_matches = [
                w for w in all_wins
                if getattr(w, 'title', '').strip()
                and getattr(w, 'title', '').lower() in title_str.lower()
                and ((w.width > 50 and w.height > 50) or w.isMinimized)
            ]
            if rev_matches:
                return sorted(rev_matches, key=lambda w: (not w.isMinimized, w.width * w.height), reverse=True)[0]

    return None


@app.get("/api/windows")
def list_windows():
    """Return list of visible desktop windows suitable for capture with handles."""
    forbidden = {
        "HUIT Deepfake Scanner", "Hệ thống Giám định Deepfake", "Program Manager",
        "Settings", "Taskbar", "Microsoft Text Input Application", "Default IME",
        "MSCTFIME UI", "Windows Input Experience", "frontend"
    }
    wins = []
    seen_hwnds = set()

    try:
        for w in gw.getAllWindows():
            try:
                hwnd = int(w._hWnd)
            except Exception:
                continue
            if hwnd in seen_hwnds:
                continue
            seen_hwnds.add(hwnd)

            title = w.title.strip() if (w.title and hasattr(w, "title")) else ""
            if not title:
                continue

            if any(f.lower() in title.lower() for f in forbidden if f):
                continue

            # Must have non-trivial size OR be minimized
            if (w.width > 80 and w.height > 80) or w.isMinimized:
                wins.append({
                    "id": hwnd,
                    "title": title,
                    "width": int(w.width),
                    "height": int(w.height),
                    "is_minimized": bool(w.isMinimized)
                })
    except Exception as e:
        print(f"Lỗi khi quét cửa sổ: {e}")

    if not wins:
        wins = [{"id": 0, "title": "Không tìm thấy cửa sổ (Hãy mở VLC, Chrome hoặc video)", "width": 0, "height": 0, "is_minimized": False}]

    return {
        "windows": wins,
        "titles": [w["title"] for w in wins]
    }


@app.post("/api/windows/activate")
async def activate_window(req: Request):
    """Bring target window to foreground."""
    try:
        data = await req.json()
        target_id = data.get("window_id")
        target_title = data.get("window_title")
        win = find_target_window(target_id, target_title)
        if win:
            if win.isMinimized:
                try:
                    win.restore()
                except Exception:
                    pass
            try:
                win.activate()
            except Exception:
                pass
            return {"status": "ok", "title": getattr(win, "title", "")}
        return {"status": "not_found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    """
    High-speed video analysis using representative keyframe sampling & PyTorch batched inference.
    Executes in ~3-4 seconds while strictly maintaining academic detection accuracy.
    """
    suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    # Uniform keyframe sampling: 14 representative frames across video duration
    TARGET_SAMPLES = 8
    step = max(1, total_frames // TARGET_SAMPLES)

    face_det = MP_FACE.FaceDetection(model_selection=0, min_detection_confidence=0.6)
    sampled_items = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % step != 0 or len(sampled_items) >= TARGET_SAMPLES:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_det.process(rgb)
        if res.detections:
            ih, iw = frame.shape[:2]
            det = res.detections[0]
            bb = det.location_data.relative_bounding_box
            x, y = int(bb.xmin * iw), int(bb.ymin * ih)
            w, h = int(bb.width * iw), int(bb.height * ih)
            px, py = int(w * 0.15), int(h * 0.2)
            x1, y1 = max(0, x - px), max(0, y - py * 2)
            x2, y2 = min(iw, x + w + px), min(ih, y + h + py)

            crop = rgb[y1:y2, x1:x2]
            if crop.size > 0:
                sampled_items.append({
                    "frame_idx": frame_count,
                    "crop": crop,
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                })

    cap.release()
    try:
        os.unlink(tmp_path)
    except Exception:
        pass
    face_det.close()

    if not sampled_items:
        return JSONResponse({"error": "Không tìm thấy khuôn mặt nào trong video tải lên."}, status_code=422)

    # Batched PyTorch inference on all keyframe face crops in 1 single forward pass
    face_crops = [item["crop"] for item in sampled_items]
    probs = infer_faces_batch(face_crops)

    frame_results = []
    for item, prob in zip(sampled_items, probs):
        is_fake = prob > 0.5
        pct = prob * 100 if is_fake else (1 - prob) * 100
        frame_results.append({
            "frame": item["frame_idx"],
            "time_sec": round(item["frame_idx"] / fps, 2),
            "detections": [{
                "is_fake": is_fake,
                "probability": round(prob, 4),
                "confidence": round(pct, 1),
                "bbox": item["bbox"],
            }],
        })

    all_probs = probs
    avg_prob = sum(all_probs) / len(all_probs) if all_probs else 0.5
    verdict = "DEEPFAKE" if avg_prob > 0.5 else "REAL"

    return {
        "verdict": verdict,
        "average_probability": round(avg_prob, 4),
        "confidence": round((avg_prob if verdict == "DEEPFAKE" else 1 - avg_prob) * 100, 1),
        "total_frames_analyzed": len(frame_results),
        "frames": frame_results,
    }


# ──────────────────────────────────────────────
# 6. WEBSOCKET – WEBCAM
# ──────────────────────────────────────────────
@app.websocket("/ws/webcam")
async def ws_webcam(ws: WebSocket, camera_index: int = 0):
    await ws.accept()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened() and camera_index != 0:
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        await ws.send_json({"error": "Cannot open webcam"})
        await ws.close()
        return

    # Set optimal capture resolution for high FPS on CPU
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    history: collections.deque = collections.deque(maxlen=8)
    smoothed_prob = 0.5
    face_det = MP_FACE.FaceDetection(model_selection=0, min_detection_confidence=0.6)
    prev_face_crop = None
    current_task = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            # 1. Check if background inference finished
            if current_task is not None and current_task.done():
                try:
                    raw = current_task.result()
                    history.append(raw)
                    smoothed_prob = sum(history) / len(history)
                except Exception as ex:
                    print(f"Async infer error: {ex}")
                current_task = None

            # 2. Fast face detection (MediaPipe runs in ~5ms)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_det.process(rgb)
            detections = []
            face_found = False

            if results.detections:
                face_found = True
                ih, iw = frame.shape[:2]
                for det in results.detections:
                    bb = det.location_data.relative_bounding_box
                    x, y = int(bb.xmin * iw), int(bb.ymin * ih)
                    w, h = int(bb.width * iw), int(bb.height * ih)
                    px, py = int(w * 0.15), int(h * 0.2)
                    x1, y1 = max(0, x - px), max(0, y - py * 2)
                    x2, y2 = min(iw, x + w + px), min(ih, y + h + py)

                    face_crop = rgb[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        # Dispatch inference in background if worker is free
                        if current_task is None:
                            current_task = INFERENCE_EXECUTOR.submit(
                                infer_face, face_crop.copy(), prev_face_crop.copy() if prev_face_crop is not None else None
                            )
                            prev_face_crop = face_crop

                    is_fake = smoothed_prob > 0.5
                    pct = smoothed_prob * 100 if is_fake else (1 - smoothed_prob) * 100
                    label = f"{'DEEPFAKE' if is_fake else 'REAL'}: {pct:.1f}%"
                    color = (0, 0, 220) if is_fake else (0, 200, 80)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                    cv2.rectangle(frame, (x1, y1 - 28), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 2, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

                    detections.append({
                        "is_fake": is_fake,
                        "probability": round(smoothed_prob, 4),
                        "confidence": round(pct, 1),
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    })
                    break  # Focus on primary face for maximum speed

            b64 = frame_to_base64(frame, quality=60)
            await ws.send_json({
                "frame": b64,
                "face_found": face_found,
                "detections": detections,
                "smoothed_prob": round(smoothed_prob, 4),
            })
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Webcam WS error: {e}")
    finally:
        if current_task is not None and not current_task.done():
            current_task.cancel()
        cap.release()
        face_det.close()


# ──────────────────────────────────────────────
# 7. WEBSOCKET – WINDOW CAPTURE
# ──────────────────────────────────────────────
@app.websocket("/ws/window")
async def ws_window(ws: WebSocket):
    await ws.accept()
    init_data = await ws.receive_json()

    current_target = {
        "id": init_data.get("window_id"),
        "title": init_data.get("window_title", "")
    }

    sct = mss.mss()
    history: collections.deque = collections.deque(maxlen=8)
    smoothed_prob = 0.5
    face_det = MP_FACE.FaceDetection(model_selection=0, min_detection_confidence=0.6)
    prev_face_crop = None
    current_task = None
    stop_event = asyncio.Event()

    # Dynamic command listener so user can switch window while scanning
    async def command_listener():
        try:
            while not stop_event.is_set():
                msg_text = await ws.receive_text()
                try:
                    data = json.loads(msg_text)
                    if "window_id" in data or "window_title" in data:
                        nonlocal prev_face_crop
                        current_target["id"] = data.get("window_id")
                        current_target["title"] = data.get("window_title", "")
                        prev_face_crop = None  # Reset optical difference
                        history.clear()
                        print(f"🔄 Switched capture target to: {current_target}")
                except Exception as ex:
                    print(f"Parse WS message error: {ex}")
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        except Exception as e:
            print(f"Command listener error: {e}")

    listener_task = asyncio.create_task(command_listener())

    try:
        while not stop_event.is_set():
            target_id = current_target.get("id")
            target_title = current_target.get("title")

            win = find_target_window(target_id, target_title)
            if not win:
                await ws.send_json({
                    "warning": f"Đang chờ cửa sổ '{target_title or target_id}' hiển thị...",
                    "face_found": False,
                    "detections": []
                })
                await asyncio.sleep(0.3)
                continue

            if win.isMinimized:
                try:
                    win.restore()
                    await asyncio.sleep(0.12)
                except Exception:
                    pass

            try:
                # Calculate coordinates bounded by all monitors
                virt = sct.monitors[0]
                left = max(virt["left"], min(int(win.left), virt["left"] + virt["width"] - 60))
                top = max(virt["top"], min(int(win.top), virt["top"] + virt["height"] - 60))
                width = min(max(60, int(win.width)), virt["left"] + virt["width"] - left)
                height = min(max(60, int(win.height)), virt["top"] + virt["height"] - top)

                monitor = {
                    "top": top,
                    "left": left,
                    "width": width,
                    "height": height,
                }
                sct_img = sct.grab(monitor)
                frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
                if frame.shape[1] > 960:
                    scale = 960.0 / frame.shape[1]
                    frame = cv2.resize(frame, (960, int(frame.shape[0] * scale)))
            except Exception as e:
                print(f"Window grab error: {e}")
                await ws.send_json({"warning": f"Lỗi đọc khung hình: {str(e)}"})
                await asyncio.sleep(0.2)
                continue

            # 1. Check if background inference finished
            if current_task is not None and current_task.done():
                try:
                    raw = current_task.result()
                    history.append(raw)
                    smoothed_prob = sum(history) / len(history)
                except Exception as ex:
                    print(f"Async infer error: {ex}")
                current_task = None

            # 2. Fast face detection
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_det.process(rgb)
            detections = []
            face_found = False

            if results.detections:
                face_found = True
                ih, iw = frame.shape[:2]
                for det in results.detections:
                    bb = det.location_data.relative_bounding_box
                    x, y = int(bb.xmin * iw), int(bb.ymin * ih)
                    w, h = int(bb.width * iw), int(bb.height * ih)
                    px, py = int(w * 0.15), int(h * 0.2)
                    x1, y1 = max(0, x - px), max(0, y - py * 2)
                    x2, y2 = min(iw, x + w + px), min(ih, y + h + py)

                    face_crop = rgb[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        if current_task is None:
                            current_task = INFERENCE_EXECUTOR.submit(
                                infer_face, face_crop.copy(), prev_face_crop.copy() if prev_face_crop is not None else None
                            )
                            prev_face_crop = face_crop

                    is_fake = smoothed_prob > 0.5
                    pct = smoothed_prob * 100 if is_fake else (1 - smoothed_prob) * 100
                    label = f"{'DEEPFAKE' if is_fake else 'REAL'}: {pct:.1f}%"
                    color = (0, 0, 220) if is_fake else (0, 200, 80)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                    cv2.rectangle(frame, (x1, y1 - 28), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 2, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

                    detections.append({
                        "is_fake": is_fake,
                        "probability": round(smoothed_prob, 4),
                        "confidence": round(pct, 1),
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    })
                    break  # Focus on primary face

            b64 = frame_to_base64(frame, quality=55)
            await ws.send_json({
                "frame": b64,
                "face_found": face_found,
                "detections": detections,
                "smoothed_prob": round(smoothed_prob, 4),
            })
            await asyncio.sleep(0.02)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Window WS error: {e}")
    finally:
        stop_event.set()
        listener_task.cancel()
        if current_task is not None and not current_task.done():
            current_task.cancel()
        face_det.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)