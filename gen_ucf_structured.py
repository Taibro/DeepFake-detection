import json

cells = []

def code_cell(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}

def md_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": [source]}

# ── Header ────────────────────────────────────────────────────────
cells.append(md_cell(
"""# UCF Model V2 — Uncovering Common Features for Generalizable Deepfake Detection

**Paper**: *Uncovering Common Features for Generalizable Deepfake Detection (ICCV 2023)*

### Điểm nổi bật của kiến trúc
Mô hình tách biệt đặc trưng ảnh thành **Content Features** (thông tin nội dung) và **Fingerprint Features** (vết tích giả mạo) thông qua hai nhánh Xception song song. Từ nhánh Fingerprint, mô hình tách tiếp thành Common Features (đặc trưng chung) và Specific Features (đặc trưng riêng từng miền).

| Component | Chi tiết |
|---|---|
| Content Encoder | Nhánh Xception học thông tin nội dung ảnh |
| Fingerprint Encoder | Nhánh Xception học vết tích làm giả chung |
| Decoder | Khôi phục lại ảnh 256x256 để ép mô hình học đúng nội dung |
| Loss Functions | Binary (BCE), Manipulation (CE), Recon (L1), Contrastive |

### Architecture Overview
```
RGB frame (256x256) ──► Content Encoder (Xception) ─────► Content Features ──────────┐
      │                                                                              ▼
      └───────────────► Fingerprint Encoder (Xcep) ─┬───► Common Features ───────► Decoder
                                                    │
                                                    └───► Specific Features
```

### Notebook Sections
0. Environment Setup
1. Configuration & Dataset Preparation (MTCNN)
2. Dataset Class & DataLoader
3. Model Architecture (UCF)
4. Training Configuration
5. Training Loop
6. Evaluation — FF++ Validation (Accuracy, F1, Precision, Recall, AUC)
7. Cross-Dataset Evaluation — Celeb-DF-v2
8. Ablation Study
9. Export & Save"""
))

# ── Section 0 ────────────────────────────────────────────────────────
cells.append(md_cell("---\n## Section 0: Environment Setup"))
cells.append(code_cell(
"""from google.colab import drive
drive.mount('/content/drive')

!pip install -q timm facenet-pytorch Pillow==9.5.0 scikit-learn matplotlib seaborn tqdm scipy"""
))

cells.append(code_cell(
"""import os, glob, random, time, warnings, shutil, gc
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from facenet_pytorch import MTCNN
import timm

from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, auc,
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score
)
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if device.type == 'cuda':
    print(f'GPU:   {torch.cuda.get_device_name(0)}')
    print(f'VRAM:  {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"""
))

# ── Section 1 ────────────────────────────────────────────────────────
cells.append(md_cell("---\n## Section 1: Configuration & Dataset Preparation\n\nTiền xử lý ảnh tĩnh bằng MTCNN và lưu vào Drive."))
cells.append(code_cell(
"""class Config:
    # ── Paths ──
    DRIVE_ROOT   = '/content/drive/MyDrive/DoAn_Nhom4'
    FF_ZIP       = f'{DRIVE_ROOT}/FaceForensics.zip'
    CELEB_ZIP    = f'{DRIVE_ROOT}/Celeb-DF-v2.zip'
    EXTRACT_DIR  = '/content/dataset'
    FRAMES_DIR   = '/content/frames'
    BACKUP_DIR   = f'{DRIVE_ROOT}/UCF_frames_backup'
    SAVE_DIR     = f'{DRIVE_ROOT}/weights_ucf'

    # ── Extraction ──
    MAX_VIDEOS_PER_CLASS = 1000   
    FRAMES_PER_VIDEO     = 10     
    IMG_SIZE             = 256

    # ── Model ──
    FEAT_DIM     = 512
    NUM_MANIP    = 4
    DROPOUT      = 0.3

    # ── Training ──
    BATCH_SIZE           = 16
    EPOCHS               = 30
    LR                   = 1e-4
    WEIGHT_DECAY         = 1e-4
    
    # ── Losses Weights ──
    L_BIN = 1.0
    L_MAN = 0.5
    L_REC = 0.1
    L_CON = 0.3

    SEED = 42

def set_seed(seed=Config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()
for k, v in vars(Config).items():
    if not k.startswith('_'):
        print(f'  {k:<22} = {v}')"""
))

cells.append(code_cell(
"""# Giải nén FF++
os.makedirs(Config.EXTRACT_DIR, exist_ok=True)
ff_folders = glob.glob(os.path.join(Config.EXTRACT_DIR, 'FaceForensics*'))
if not ff_folders:
    print('Extracting FaceForensics++...')
    !unzip -q "{Config.FF_ZIP}" -d "{Config.EXTRACT_DIR}/"
    ff_folders = glob.glob(os.path.join(Config.EXTRACT_DIR, 'FaceForensics*'))
FF_ROOT = ff_folders[0]

if 'original' not in os.listdir(FF_ROOT):
    if 'FaceForensics++_C23' in os.listdir(FF_ROOT):
        FF_ROOT = os.path.join(FF_ROOT, 'FaceForensics++_C23')
    else:
        for root, dirs, _ in os.walk(FF_ROOT):
            if 'original' in dirs:
                FF_ROOT = root
                break

FAKE_DIRS = {
    'Deepfakes':      os.path.join(FF_ROOT, 'Deepfakes'),
    'Face2Face':      os.path.join(FF_ROOT, 'Face2Face'),
    'FaceSwap':       os.path.join(FF_ROOT, 'FaceSwap'),
    'NeuralTextures': os.path.join(FF_ROOT, 'NeuralTextures'),
}
REAL_DIR = os.path.join(FF_ROOT, 'original')
print(f'FF++ root: {FF_ROOT}')"""
))

cells.append(code_cell(
"""# MTCNN Extraction
os.makedirs(Config.FRAMES_DIR, exist_ok=True)
os.makedirs(Config.BACKUP_DIR, exist_ok=True)

mtcnn = MTCNN(margin=20, keep_all=False, post_process=False, device=device)

def extract_faces_mtcnn(video_list, save_path):
    os.makedirs(save_path, exist_ok=True)
    count = 0
    for vid_path in tqdm(video_list, desc=f"Extracting {os.path.basename(save_path)}"):
        cap = cv2.VideoCapture(vid_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: cap.release(); continue
        target_frames = set([int(i * total_frames / Config.FRAMES_PER_VIDEO) for i in range(Config.FRAMES_PER_VIDEO)])
        vid_name = os.path.basename(vid_path).split('.')[0]
        cur = 0; saved = 0
        while True:
            ret = cap.grab()
            if not ret: break
            if cur in target_frames:
                ret, frame = cap.retrieve()
                if ret:
                    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    face = mtcnn(img_pil)
                    if face is not None:
                        face_bgr = cv2.cvtColor(face.permute(1,2,0).cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
                        cv2.imwrite(os.path.join(save_path, f"{vid_name}_{saved}.jpg"), face_bgr)
                        count += 1
                saved += 1
                if saved >= Config.FRAMES_PER_VIDEO: break
            cur += 1
        cap.release()
    return count

folders = ['original', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
for folder in folders:
    src = os.path.join(Config.FRAMES_DIR, folder)
    bup = os.path.join(Config.BACKUP_DIR, folder)
    if not os.path.exists(src) and os.path.exists(bup):
        print(f"♻️ Khôi phục '{folder}' từ Drive...")
        shutil.copytree(bup, src)
    elif os.path.exists(src) and len(os.listdir(src)) > 100:
        print(f"✅ Thư mục '{folder}' đã sẵn sàng.")
    else:
        vids = [os.path.join(REAL_DIR if folder=='original' else FAKE_DIRS[folder], f) for f in os.listdir(REAL_DIR if folder=='original' else FAKE_DIRS[folder]) if f.endswith('.mp4')]
        random.shuffle(vids)
        extract_faces_mtcnn(vids[:Config.MAX_VIDEOS_PER_CLASS], src)
        shutil.copytree(src, bup, dirs_exist_ok=True)
print("🎉 Hoàn tất tiền xử lý ảnh!")"""
))

# ── Section 2 ────────────────────────────────────────────────────────
cells.append(md_cell("---\n## Section 2: Dataset Class & DataLoader"))
cells.append(code_cell(
"""MANIP2IDX = {'Deepfakes': 0, 'Face2Face': 1, 'FaceSwap': 2, 'NeuralTextures': 3}

def collect_images(folder, label_binary, label_manip):
    paths = []
    if os.path.exists(folder):
        for f in os.listdir(folder):
            if f.lower().endswith('.jpg'): paths.append((os.path.join(folder, f), label_binary, label_manip))
    return paths

all_samples = collect_images(os.path.join(Config.FRAMES_DIR, 'original'), 0, 4)
for name in FAKE_DIRS.keys():
    all_samples += collect_images(os.path.join(Config.FRAMES_DIR, name), 1, MANIP2IDX[name])

random.shuffle(all_samples)
n = len(all_samples)
n_train, n_val = int(0.8 * n), int(0.1 * n)
train_s, val_s, test_s = all_samples[:n_train], all_samples[n_train:n_train+n_val], all_samples[n_train+n_val:]
print(f"Total: {n} | Train: {len(train_s)} | Val: {len(val_s)} | Test: {len(test_s)}")

train_tf = T.Compose([
    T.Resize((Config.IMG_SIZE, Config.IMG_SIZE)), T.RandomHorizontalFlip(),
    T.ColorJitter(0.2, 0.2, 0.2, 0.05), T.RandomGrayscale(0.05),
    T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_tf = T.Compose([
    T.Resize((Config.IMG_SIZE, Config.IMG_SIZE)), T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

class FFDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples, self.transform = samples, transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, lb, lm = self.samples[idx]
        return self.transform(Image.open(path).convert('RGB')), lb, lm

train_loader = DataLoader(FFDataset(train_s, train_tf), Config.BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(FFDataset(val_s,   val_tf),   Config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(FFDataset(test_s,  val_tf),   Config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)"""
))

# ── Section 3 ────────────────────────────────────────────────────────
cells.append(md_cell("---\n## Section 3: Model Architecture V5 (UCF)"))
cells.append(code_cell(
"""class XceptionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        base = timm.create_model('xception', pretrained=True, num_classes=0, global_pool='')
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.pool     = nn.AdaptiveAvgPool2d(1)
    def forward(self, x): return self.pool(self.features(x)).flatten(1)

class UCFModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone_dim = 2048
        
        # Content Branch
        self.content_enc  = XceptionEncoder()
        self.content_proj = nn.Linear(backbone_dim, Config.FEAT_DIM)
        
        # Fingerprint Branch
        self.fp_enc        = XceptionEncoder()
        self.common_proj   = nn.Linear(backbone_dim, Config.FEAT_DIM)
        self.specific_proj = nn.Linear(backbone_dim, Config.FEAT_DIM)
        
        # Recon Decoder (8x8 -> 256x256)
        self.decoder = nn.Sequential(
            nn.Linear(Config.FEAT_DIM * 2, 512 * 8 * 8), nn.Unflatten(1, (512, 8, 8)),
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128,  64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d( 64,  32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d( 32,   3, 4, 2, 1), nn.Tanh(),
        )
        
        # Classifiers
        self.binary_cls = nn.Sequential(nn.Linear(Config.FEAT_DIM, 256), nn.ReLU(), nn.Dropout(Config.DROPOUT), nn.Linear(256, 1))
        self.manip_cls  = nn.Sequential(nn.Linear(Config.FEAT_DIM, 256), nn.ReLU(), nn.Dropout(Config.DROPOUT), nn.Linear(256, Config.NUM_MANIP))

    def forward(self, x):
        content  = self.content_proj(self.content_enc(x))
        fp_feat  = self.fp_enc(x)
        common   = self.common_proj(fp_feat)
        specific = self.specific_proj(fp_feat)
        
        recon = self.decoder(torch.cat([content, common], dim=1))
        bin_logit = self.binary_cls(common).squeeze(1)
        manip_logit = self.manip_cls(specific)
        
        return bin_logit, manip_logit, recon, common, specific

model = UCFModel().to(device)
print(f"✅ UCF Model Loaded | Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")"""
))

# ── Section 4 ────────────────────────────────────────────────────────
cells.append(md_cell("---\n## Section 4: Training Configuration"))
cells.append(code_cell(
"""def contrastive_loss(common, specific, temperature=0.07):
    cn = F.normalize(common, dim=1); sn = F.normalize(specific, dim=1)
    sim = torch.sum(cn * sn, dim=1) / temperature
    return F.mse_loss(sim, torch.zeros_like(sim))

def reconstruction_loss(recon, target):
    mean = torch.tensor([0.485,0.456,0.406], device=target.device).view(1,3,1,1)
    std  = torch.tensor([0.229,0.224,0.225], device=target.device).view(1,3,1,1)
    t = (target * std + mean) * 2 - 1
    return F.l1_loss(recon, t)

bce_fn = nn.BCEWithLogitsLoss()
ce_fn  = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)"""
))

# ── Section 5 ────────────────────────────────────────────────────────
cells.append(md_cell("---\n## Section 5: Training Loop"))
cells.append(code_cell(
"""os.makedirs(Config.SAVE_DIR, exist_ok=True)
best_auc = 0
history  = {'tr_loss': [], 'vl_loss': [], 'tr_acc': [], 'vl_acc': [], 'vl_auc': []}

def train_epoch(loader):
    model.train()
    total_loss = 0; correct = 0; total = 0
    for imgs, lb, lm in loader:
        imgs, lb, lm = imgs.to(device), lb.float().to(device), lm.to(device)
        optimizer.zero_grad()
        
        bin_logit, manip_logit, recon, common, specific = model(imgs)
        l_bin  = bce_fn(bin_logit, lb)
        mask   = (lb == 1) & (lm < 4)
        l_mani = ce_fn(manip_logit[mask], lm[mask]) if mask.sum() > 0 else torch.tensor(0.).to(device)
        
        loss = (Config.L_BIN * l_bin + Config.L_MAN * l_mani + 
                Config.L_REC * reconstruction_loss(recon, imgs) + 
                Config.L_CON * contrastive_loss(common, specific))
                
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * imgs.size(0)
        correct += ((torch.sigmoid(bin_logit) > 0.5).long() == lb.long()).sum().item()
        total += imgs.size(0)
    torch.cuda.empty_cache()
    return total_loss / total, correct / total

def eval_epoch(loader):
    model.eval()
    total_loss = 0; all_probs = []; all_labels = []
    with torch.no_grad():
        for imgs, lb, lm in loader:
            imgs, lb, lm = imgs.to(device), lb.float().to(device), lm.to(device)
            bin_logit, manip_logit, recon, common, specific = model(imgs)
            mask = (lb == 1) & (lm < 4)
            l_mani = ce_fn(manip_logit[mask], lm[mask]) if mask.sum() > 0 else torch.tensor(0.).to(device)
            loss = (Config.L_BIN * bce_fn(bin_logit, lb) + Config.L_MAN * l_mani + 
                    Config.L_REC * reconstruction_loss(recon, imgs) + 
                    Config.L_CON * contrastive_loss(common, specific))
            total_loss += loss.item() * imgs.size(0)
            all_probs.extend(torch.sigmoid(bin_logit).cpu().numpy())
            all_labels.extend(lb.cpu().numpy())
    probs = np.array(all_probs); labels = np.array(all_labels)
    preds = (probs > 0.5).astype(int)
    fpr, tpr, _ = roc_curve(labels, probs)
    return total_loss / len(labels), accuracy_score(labels, preds), auc(fpr, tpr), probs, preds, labels, fpr, tpr

for epoch in range(1, Config.EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_acc = train_epoch(train_loader)
    vl_loss, vl_acc, vl_auc, *_ = eval_epoch(val_loader)
    scheduler.step()
    
    for k, v in zip(history.keys(), [tr_loss, vl_loss, tr_acc, vl_acc, vl_auc]): history[k].append(v)
    
    tag = ""
    if vl_auc > best_auc:
        best_auc = vl_auc
        torch.save(model.state_dict(), os.path.join(Config.SAVE_DIR, 'ucf_v5_best.pth'))
        tag = " ← 🏆 BEST"
        
    print(f"Epoch {epoch:02d} | Loss: {tr_loss:.4f}/{vl_loss:.4f} | Acc: {tr_acc:.4f}/{vl_acc:.4f} | AUC: {vl_auc:.4f} | {time.time()-t0:.0f}s{tag}")"""
))

# ── Section 6 ────────────────────────────────────────────────────────
cells.append(md_cell("---\n## Section 6: Evaluation — FF++ Validation (Full Metrics)"))
cells.append(code_cell(
"""# Load best weights
model.load_state_dict(torch.load(os.path.join(Config.SAVE_DIR, 'ucf_v5_best.pth'), map_location=device))
_, test_acc, test_auc, probs, preds, labels, fpr, tpr = eval_epoch(test_loader)

eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
f1  = f1_score(labels, preds)
prec = precision_score(labels, preds)
rec = recall_score(labels, preds)

print(f"==================================================")
print(f"📊 IN-DATASET EVALUATION (FaceForensics++ C23)")
print(f"🎯 Accuracy  : {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"🎯 AUC       : {test_auc:.4f}")
print(f"🎯 EER       : {eer:.4f}")
print(f"🎯 Precision : {prec:.4f}")
print(f"🎯 Recall    : {rec:.4f}")
print(f"🎯 F1-Score  : {f1:.4f}")
print(f"==================================================")

print("\\nClassification Report:")
print(classification_report(labels, preds, target_names=['Real (0)', 'Fake (1)']))

# Vẽ biểu đồ CM & ROC
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

cm = confusion_matrix(labels, preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], annot_kws={"size": 16})
axes[0].set_title('Confusion Matrix', fontweight='bold', fontsize=18)
axes[0].set_xticklabels(['Real', 'Fake'])
axes[0].set_yticklabels(['Real', 'Fake'])

axes[1].plot(fpr, tpr, color='#d62728', lw=3, label=f'UCF (AUC = {test_auc:.4f} | EER = {eer:.4f})')
axes[1].plot([0,1],[0,1], color='navy', lw=2, linestyle='--', alpha=0.6)
axes[1].scatter([eer], [1-eer], s=200, marker='*', color='gold', edgecolor='black', zorder=5)
axes[1].set_xlabel('False Positive Rate', fontweight='bold')
axes[1].set_ylabel('True Positive Rate', fontweight='bold')
axes[1].set_title('ROC Curve', fontweight='bold', fontsize=18)
axes[1].legend(loc='lower right', frameon=True, shadow=True, fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(Config.SAVE_DIR, 'ucf_v5_evaluation.pdf'), format='pdf', bbox_inches='tight')
plt.show()"""
))

# ── Section 7 ────────────────────────────────────────────────────────
cells.append(md_cell("---\n## Section 7: Cross-Dataset Evaluation — Celeb-DF-v2"))
cells.append(code_cell(
"""# Hướng dẫn test chéo trên Celeb-DF giống format V4
\"\"\"
CELEB_DIR = '/content/drive/MyDrive/CelebDF_frames'
celeb_samples = []
celeb_samples += collect_images(os.path.join(CELEB_DIR, 'real'), label_binary=0, label_manip=4)
celeb_samples += collect_images(os.path.join(CELEB_DIR, 'fake'), label_binary=1, label_manip=0) 
celeb_loader = DataLoader(FFDataset(celeb_samples, val_tf), Config.BATCH_SIZE, shuffle=False, num_workers=2)

_, celeb_acc, celeb_auc, c_probs, c_preds, c_labels, c_fpr, c_tpr = eval_epoch(celeb_loader)
celeb_eer = brentq(lambda x: 1. - x - interp1d(c_fpr, c_tpr)(x), 0., 1.)

print(f"📊 CROSS-DATASET EVALUATION (Celeb-DF v2)")
print(f"🎯 Accuracy : {celeb_acc:.4f} | AUC: {celeb_auc:.4f} | EER: {celeb_eer:.4f}")
print("\\nClassification Report (Celeb-DF):")
print(classification_report(c_labels, c_preds, target_names=['Real (0)', 'Fake (1)']))
\"\"\"
print("✅ Cấu trúc Cross-Dataset (Celeb-DF) đã sẵn sàng.")"""
))

# ── Section 8 ────────────────────────────────────────────────────────
cells.append(md_cell("---\n## Section 8: Ablation Study\n\nBạn có thể thử nghiệm tắt/bật Contrastive Loss (Config.L_CON = 0) hoặc Reconstruction Loss (Config.L_REC = 0) để chạy kiểm chứng độ hiệu quả của từng nhánh mạng."))
cells.append(code_cell(
"""# Để thực hiện Ablation Study:
# 1. Chỉnh Config.L_CON = 0.0
# 2. Re-run Section 5 (Training Loop)
# 3. Đổi tên weight lưu thành 'ucf_no_contrastive.pth'
print("Ready for Ablation Study.")"""
))

# ── Section 9 ────────────────────────────────────────────────────────
cells.append(md_cell("---\n## Section 9: Export & Save"))
cells.append(code_cell(
"""# Đóng gói và lưu toàn bộ kết quả lên Drive
print(f"✅ Mọi file weight và biểu đồ PDF chất lượng cao đã được sao lưu tại: {Config.SAVE_DIR}")"""
))

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "cells": cells
}

out = 'd:/file_ky6/DEEP_LEARNING/DeepFake-detection/train_and_test_model/UCF_Model_Master_Structured.ipynb'
with open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Đã tạo {out} thành công!")
