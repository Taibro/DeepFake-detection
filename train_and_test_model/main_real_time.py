import cv2
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import numpy as np
import mediapipe as mp # Import thư viện dò tìm khuôn mặt của Google

# ==========================================
# 1. ĐỊNH NGHĨA LẠI KIẾN TRÚC MÔ HÌNH (Giữ nguyên)
# ==========================================
# ==========================================
# 1. ĐỊNH NGHĨA KIẾN TRÚC MÔ HÌNH (BẢN V2 NÂNG CẤP)
# ==========================================
class DeepfakeFusionModel(nn.Module):
    def __init__(self):
        super(DeepfakeFusionModel, self).__init__()
        # 1. Nhánh Visual (Swin)
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=0)
        
        # 2. Nhánh rPPG (Sinh tồn)
        self.rppg_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 3. VŨ KHÍ MỚI: CƠ CHẾ CROSS-ATTENTION FUSION
        self.attention_weights = nn.Sequential(
            nn.Linear(768 + 64, 768 + 64),
            nn.Sigmoid() 
        )
        
        # 4. Bộ phân loại cuối cùng (Đã thêm BatchNorm1d)
        self.classifier = nn.Sequential(
            nn.Linear(768 + 64, 128), 
            nn.BatchNorm1d(128), 
            nn.ReLU(), 
            nn.Dropout(0.5), 
            nn.Linear(128, 1), 
            nn.Sigmoid()
        )

    def forward(self, image_input, rppg_input):
        visual_features = self.swin(image_input)                       
        bio_features = torch.flatten(self.rppg_net(rppg_input), 1)     
        
        combined_features = torch.cat((visual_features, bio_features), dim=1) 
        attention_scores = self.attention_weights(combined_features)
        
        # Nhân đặc trưng với điểm Attention (Lọc nhiễu)
        attended_features = combined_features * attention_scores       
        
        return self.classifier(attended_features)
    
# ==========================================
# 2. KHỞI TẠO AI VÀ MEDIAPIPE
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepfakeFusionModel().to(device)

try:
    # Thay đường dẫn tới file .pth của bạn nếu cần
    model.load_state_dict(torch.load("fusion_model_ffpp_v2_epoch_20.pth", map_location=device))
    print("✅ Đã load AI thành công!")
except Exception as e:
    print(f"❌ Lỗi load mô hình: {e}")
    exit()

model.eval()

# Transform cho AI
swin_transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
rppg_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

# --- KHỞI TẠO MEDIAPIPE FACE DETECTION ---
mp_face_detection = mp.solutions.face_detection
# model_selection=0 (khoảng cách gần < 2m), min_detection_confidence=0.7 (độ tự tin 70%)
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7)

# ==========================================
# 3. VÒNG LẶP WEBCAM (Tích hợp Cắt khuôn mặt)
# ==========================================
cap = cv2.VideoCapture(0) 

print("🎥 Đang mở Webcam... Nhấn phím 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Lật ngược khung hình (như gương) để dễ nhìn
    frame = cv2.flip(frame, 1)
    
    # MediaPipe yêu cầu ảnh RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # BƯỚC 1: MEDIAPIPE TÌM KHUÔN MẶT
    results = face_detection.process(rgb_frame)
    
    # Nếu tìm thấy ít nhất 1 khuôn mặt trong Camera
    if results.detections:
        for detection in results.detections:
            # Lấy tọa độ hộp sọ (Bounding Box)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)
            
            # --- CỘNG THÊM PADDING (MỞ RỘNG KHUNG CẮT) ---
            # Tránh việc cắt cụt mất trán và cằm (rất quan trọng cho nhịp tim rPPG)
            pad_x = int(w * 0.15)
            pad_y = int(h * 0.2)
            
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y * 2) # Mở rộng lên trên nhiều hơn để lấy trán
            x2 = min(iw, x + w + pad_x)
            y2 = min(ih, y + h + pad_y)
            
            # BƯỚC 2: CẮT KHUÔN MẶT RA KHỎI KHUNG HÌNH
            face_crop = rgb_frame[y1:y2, x1:x2]
            
            # Kiểm tra an toàn: Tránh lỗi mảng rỗng nếu mặt bị khuất mép camera
            if face_crop.size == 0: continue
            
            # BƯỚC 3: ĐƯA KHUÔN MẶT ĐÃ CẮT VÀO AI
            img_pil = Image.fromarray(face_crop)
            visual_input = swin_transform(img_pil).unsqueeze(0).to(device)
            rppg_input = rppg_transform(img_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                fake_prob = model(visual_input, rppg_input).item()
            
            # BƯỚC 4: VẼ KẾT QUẢ LÊN MÀN HÌNH CHÍNH
            if fake_prob > 0.5:
                label = f"DEEPFAKE: {fake_prob*100:.1f}%"
                color = (0, 0, 255) # Đỏ
            else:
                label = f"REAL: {(1-fake_prob)*100:.1f}%"
                color = (0, 255, 0) # Xanh lá
                
            # Vẽ hình chữ nhật ôm sát khuôn mặt
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # In chữ phía trên khuôn mặt
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1)

    # Hiển thị toàn cảnh
    cv2.imshow("Deepfake Scanner - Nhom 4 (MediaPipe Integration)", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()