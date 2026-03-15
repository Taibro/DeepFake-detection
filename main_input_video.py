import cv2
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog, messagebox

# ==========================================
# 1. ĐỊNH NGHĨA KIẾN TRÚC MÔ HÌNH
# ==========================================
class DeepfakeFusionModel(nn.Module):
    def __init__(self):
        super(DeepfakeFusionModel, self).__init__()
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=0)
        self.rppg_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(768 + 64, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, image_input, rppg_input):
        visual_features = self.swin(image_input)
        bio_features = torch.flatten(self.rppg_net(rppg_input), 1)
        combined_features = torch.cat((visual_features, bio_features), dim=1)
        return self.classifier(combined_features)

# ==========================================
# 2. KHỞI TẠO AI VÀ MEDIAPIPE (Chạy ngầm khi mở app)
# ==========================================
print("⏳ Đang khởi động hệ thống AI. Vui lòng đợi...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepfakeFusionModel().to(device)

try:
    model.load_state_dict(torch.load("fusion_model_ffpp_epoch_20.pth", map_location=device))
    print("✅ Đã load AI thành công!")
except Exception as e:
    print(f"❌ Lỗi load mô hình: {e}")
    exit()

model.eval()

swin_transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
rppg_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

mp_face_detection = mp.solutions.face_detection
# Dùng model_selection=1 cho video (khoảng cách xa/chuyển động), thay vì 0 (khoảng cách gần như webcam)
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)

# ==========================================
# 3. HÀM XỬ LÝ VIDEO (TỐI ƯU SIÊU TỐC)
# ==========================================
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        messagebox.showerror("Lỗi", "Không thể mở được file video này!")
        return

    print(f"▶️ Đang phát video: {video_path}")
    print("Nhấn 'q' trên bàn phím để dừng phát video sớm.")

    frame_counter = 0     # Bộ đếm khung hình
    FRAME_SKIP = 5        # Cứ 5 frame mới chạy AI 1 lần
    
    # Bộ nhớ tạm để lưu kết quả AI của frame trước đó
    last_label = "Scanning..." 
    last_color = (255, 255, 0) # Màu vàng lơ lửng lúc chưa quét xong
    last_box = None            # Tọa độ khuôn mặt

    while True:
        ret, frame = cap.read()
        if not ret: 
            print("⏹️ Đã quét xong video.")
            break
            
        frame_counter += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # BƯỚC 1: MEDIAPIPE TÌM KHUÔN MẶT (Chạy mọi frame vì nó rất nhẹ, giúp khung vẽ bám mặt mượt mà)
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
                
                last_box = (x1, y1, x2, y2) # Cập nhật tọa độ mới nhất
                
                # BƯỚC 2: CHẠY AI (CHỈ CHẠY MỖI 5 FRAME 1 LẦN)
                if frame_counter % FRAME_SKIP == 0 or frame_counter == 1:
                    face_crop = rgb_frame[y1:y2, x1:x2]
                    
                    if face_crop.size > 0:
                        img_pil = Image.fromarray(face_crop)
                        visual_input = swin_transform(img_pil).unsqueeze(0).to(device)
                        rppg_input = rppg_transform(img_pil).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            fake_prob = model(visual_input, rppg_input).item()
                        
                        # Cập nhật kết quả vào bộ nhớ tạm
                        if fake_prob > 0.5:
                            last_label = f"DEEPFAKE: {fake_prob*100:.1f}%"
                            last_color = (0, 0, 255) # Đỏ
                        else:
                            last_label = f"REAL: {(1-fake_prob)*100:.1f}%"
                            last_color = (0, 255, 0) # Xanh lá

        # BƯỚC 3: VẼ KẾT QUẢ LÊN MÀN HÌNH (Dùng kết quả từ bộ nhớ tạm)
        if last_box is not None:
            x1, y1, x2, y2 = last_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), last_color, 2)
            cv2.putText(frame, last_label, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, last_color, 1)

        cv2.imshow("Deepfake Scanner - Nhom 4 (Optimized)", frame)
        
        # ĐÃ SỬA LỖI WAITKEY TẠI ĐÂY: Dùng 1 để video chạy nhanh nhất có thể
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()
    
    
# ==========================================
# 4. THIẾT KẾ GIAO DIỆN TKINTER CƠ BẢN
# ==========================================
def open_file_dialog():
    """Hàm mở hộp thoại chọn file"""
    file_path = filedialog.askopenfilename(
        title="Chọn video để kiểm tra",
        filetypes=[
            ("Video Files", "*.mp4 *.avi *.mov *.mkv"),
            ("All Files", "*.*")
        ]
    )
    
    if file_path:
        process_video(file_path)

# Khởi tạo cửa sổ chính
root = tk.Tk()
root.title("Ứng dụng Nhận diện Deepfake - Nhóm 4")
root.geometry("400x200") # Kích thước cửa sổ 400x200 pixel
root.configure(bg="#f0f0f0")

# Thêm Tiêu đề
lbl_title = tk.Label(root, text="HỆ THỐNG QUÉT DEEPFAKE", font=("Arial", 16, "bold"), bg="#f0f0f0")
lbl_title.pack(pady=20)

# Thêm Nút bấm
btn_select = tk.Button(
    root, 
    text="📂 Chọn File Video", 
    font=("Arial", 12), 
    bg="#4CAF50", 
    fg="white", 
    padx=20, 
    pady=10, 
    command=open_file_dialog
)
btn_select.pack()

# Thêm dòng chữ hướng dẫn
lbl_info = tk.Label(root, text="(Hỗ trợ: .mp4, .avi, .mov)", font=("Arial", 9, "italic"), bg="#f0f0f0", fg="gray")
lbl_info.pack(pady=10)

# Bắt đầu chạy vòng lặp GUI
root.mainloop()