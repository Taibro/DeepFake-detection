import cv2
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import numpy as np
import mediapipe as mp
import mss 
import tkinter as tk
from tkinter import ttk
import collections
import pygetwindow as gw

# ==========================================
# 1. ĐỊNH NGHĨA KIẾN TRÚC MÔ HÌNH (Swin + rPPG)
# ==========================================
class DeepfakeFusionModel(nn.Module):
    def __init__(self):
        super(DeepfakeFusionModel, self).__init__()
        # Nhánh Visual (Swin Transformer)
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=0)
        
        # Nhánh rPPG (Sinh lý học)
        self.rppg_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Cơ chế Cross-Attention Fusion
        self.attention_weights = nn.Sequential(
            nn.Linear(768 + 64, 768 + 64), nn.Sigmoid() 
        )
        
        # Bộ phân loại cuối cùng
        self.classifier = nn.Sequential(
            nn.Linear(768 + 64, 128), nn.BatchNorm1d(128), nn.ReLU(), 
            nn.Dropout(0.5), nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, image_input, rppg_input):
        visual_features = self.swin(image_input)                       
        bio_features = torch.flatten(self.rppg_net(rppg_input), 1)     
        combined_features = torch.cat((visual_features, bio_features), dim=1) 
        attention_scores = self.attention_weights(combined_features)
        attended_features = combined_features * attention_scores       
        return self.classifier(attended_features)

# ==========================================
# 2. HÀM CHẠY SCANNER (Core Logic)
# ==========================================
def run_scanner(mode="webcam", target_window_title=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepfakeFusionModel().to(device)

    try:
        # Load trọng số đã huấn luyện
        model.load_state_dict(torch.load("fusion_model_ffpp_v2_epoch_20.pth", map_location=device))
        print("✅ Đã load AI thành công!")
    except Exception:
        print("⚠️ CẢNH BÁO: Đang chạy bằng trọng số ngẫu nhiên!")
    
    model.eval()

    swin_transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    rppg_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7)

    if mode == "webcam":
        cap = cv2.VideoCapture(0)
    elif mode == "window":
        sct = mss.mss()

    history_probs = collections.deque(maxlen=10) # Lưu 10 kết quả gần nhất để làm mượt
    frame_count = 0
    smoothed_prob = 0.5 

    # Cửa sổ hiển thị AI: Luôn nổi và cố định ở góc
    cv2.namedWindow("HUIT Deepfake Scanner", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("HUIT Deepfake Scanner", cv2.WND_PROP_TOPMOST, 1)
    cv2.resizeWindow("HUIT Deepfake Scanner", 350, 300)
    cv2.moveWindow("HUIT Deepfake Scanner", 20, 20)
    
    while True:
        frame_count += 1
        
        if mode == "webcam":
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            
        elif mode == "window":
            try:
                win = gw.getWindowsWithTitle(target_window_title)[0]
                if win.isMinimized: continue
                
                # Lấy tọa độ vùng hiển thị của cửa sổ được chọn
                monitor = {"top": win.top, "left": win.left, "width": win.width, "height": win.height}
                sct_img = sct.grab(monitor)
                frame = np.array(sct_img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            except Exception:
                print("❌ Lỗi: Cửa sổ bị đóng hoặc không tìm thấy!")
                break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
                w, h = int(bboxC.width * iw), int(bboxC.height * ih)
                
                pad_x, pad_y = int(w * 0.15), int(h * 0.2)
                x1, y1 = max(0, x - pad_x), max(0, y - pad_y * 2) 
                x2, y2 = min(iw, x + w + pad_x), min(ih, y + h + pad_y)
                
                face_crop = rgb_frame[y1:y2, x1:x2]
                if face_crop.size == 0: continue
                
                # Chạy AI mỗi 3 khung hình để tối ưu FPS
                if frame_count % 3 == 0:
                    img_pil = Image.fromarray(face_crop)
                    visual_input = swin_transform(img_pil).unsqueeze(0).to(device)
                    rppg_input = rppg_transform(img_pil).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        raw_prob = model(visual_input, rppg_input).item()
                        history_probs.append(raw_prob)
                        smoothed_prob = sum(history_probs) / len(history_probs)
                
                # Hiển thị kết quả
                if smoothed_prob > 0.5:
                    label, color = f"DEEPFAKE: {smoothed_prob*100:.1f}%", (0, 0, 255)
                else:
                    label, color = f"REAL: {(1-smoothed_prob)*100:.1f}%", (0, 255, 0)
                    
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + tw, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

        cv2.imshow("HUIT Deepfake Scanner", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    if mode == "webcam": cap.release()
    cv2.destroyAllWindows()

# ==========================================
# 3. GIAO DIỆN APP (UI HIỆN ĐẠI)
# ==========================================
def get_active_windows():
    """Lấy danh sách các ứng dụng đang mở, loại bỏ chính App này để tránh lặp"""
    windows = gw.getAllWindows()
    valid_windows = []
    forbidden = ["HUIT Deepfake Scanner", "Program Manager", "Settings"]
    
    for w in windows:
        if w.title and w.width > 100 and w.height > 100 and w.visible:
            if not any(f in w.title for f in forbidden):
                valid_windows.append(w.title)
    return sorted(list(set(valid_windows)))

def start_webcam():
    root.destroy() 
    run_scanner(mode="webcam")

def start_window_record():
    selected = combo_windows.get()
    if selected:
        root.destroy() 
        run_scanner(mode="window", target_window_title=selected)

def auto_refresh_dropdown():
    """Tự động cập nhật danh sách cửa sổ mỗi khi click vào Menu"""
    fresh_list = get_active_windows()
    combo_windows['values'] = fresh_list
    if not combo_windows.get() and fresh_list:
        combo_windows.current(0)

# Cửa sổ Menu chính
root = tk.Tk()
root.title("HUIT Deepfake Scanner - Nhóm 4")
root.geometry("450x380")
root.configure(bg="#ffffff")

style = ttk.Style()
style.theme_use('clam')
style.configure("TButton", font=("Segoe UI", 11, "bold"), padding=8)
style.configure("TCombobox", font=("Segoe UI", 10), padding=5)

header = tk.Frame(root, bg="#f8f9fa", pady=15)
header.pack(fill="x")
tk.Label(header, text="DEEPFAKE SCANNER", font=("Segoe UI Black", 18), bg="#f8f9fa", fg="#2c3e50").pack()
tk.Label(header, text="Phát hiện giả mạo thời gian thực", font=("Segoe UI", 10, "italic"), bg="#f8f9fa", fg="#7f8c8d").pack()

# Khu vực Webcam
tk.Frame(root, bg="#ffffff", pady=10).pack()
btn_webcam = tk.Button(root, text="📷 Quét Webcam (Tự kiểm tra)", font=("Segoe UI", 11, "bold"), 
                       bg="#27ae60", fg="white", relief="flat", cursor="hand2", width=35, command=start_webcam)
btn_webcam.pack(pady=10)

tk.Frame(root, height=1, bg="#e0e0e0").pack(fill="x", padx=40, pady=5)

# Khu vực Window Capture
frame_app = tk.Frame(root, bg="#ffffff", pady=10)
frame_app.pack(fill="x", padx=20)
tk.Label(frame_app, text="Chọn ứng dụng cần quét (Zalo, Meet, Chrome):", font=("Segoe UI", 10), bg="#ffffff", fg="#34495e").pack(anchor="w")

# Dropdown với tính năng TỰ ĐỘNG CẬP NHẬT (postcommand)
combo_windows = ttk.Combobox(frame_app, postcommand=auto_refresh_dropdown, state="readonly")
combo_windows.pack(fill="x", pady=10)
# Quét lần đầu khi mở app
combo_windows['values'] = get_active_windows()
if combo_windows['values']: combo_windows.current(0)

btn_screen = tk.Button(frame_app, text="💻 Bắt đầu Quét Ứng Dụng", font=("Segoe UI", 11, "bold"), 
                       bg="#2980b9", fg="white", relief="flat", cursor="hand2", width=35, command=start_window_record)
btn_screen.pack(pady=5)

tk.Label(root, text="*Nhấn phím 'Q' trong màn hình AI để dừng quét", font=("Segoe UI", 9), bg="#ffffff", fg="#95a5a6").pack(side="bottom", pady=10)

root.mainloop()