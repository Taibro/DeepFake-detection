import torch
import torch.nn as nn
import timm

class SpatialTransformerDetector(nn.Module):
    def __init__(self, backbone_name="swin_base_patch4_window7_224", pretrained=False, drop_path_rate=0.2):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, drop_path_rate=drop_path_rate)
        feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.classifier(self.backbone(x)).squeeze(-1)

class SqueezeExcitation(nn.Module):
    def __init__(self, ch, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, ch // reduction), nn.ReLU(),
            nn.Linear(ch // reduction, ch), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)

class ResConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch)
        )
        self.se = SqueezeExcitation(out_ch) if use_se else nn.Identity()
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.se(self.conv(x)) + self.shortcut(x))

class AttentionMask(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, 1, 1), nn.Sigmoid())
    def forward(self, x):
        return self.conv(x)

class TemporalTransformer(nn.Module):
    def __init__(self, dim, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pos_drop  = nn.Dropout(dropout)
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 2,
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm    = nn.LayerNorm(dim)
    def forward(self, x):
        B = x.shape[0]
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        return self.norm(self.encoder(self.pos_drop(x))[:, 0])

class DeepPhysCAN_v2(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.app_block1 = ResConvBlock(3, 48);   self.att_mask1 = AttentionMask(48)
        self.app_block2 = ResConvBlock(48, 96);  self.att_mask2 = AttentionMask(96)
        self.app_block3 = ResConvBlock(96, 192, use_se=True);  self.att_mask3 = AttentionMask(192)
        self.app_block4 = ResConvBlock(192, 256, use_se=True); self.att_mask4 = AttentionMask(256)
        self.mot_block1 = ResConvBlock(3, 48)
        self.mot_block2 = ResConvBlock(48, 96)
        self.mot_block3 = ResConvBlock(96, 192, use_se=True)
        self.mot_block4 = ResConvBlock(192, 256, use_se=True)
        self.pool = nn.AvgPool2d(2, 2)
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.frame_feat = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Dropout(dropout))
        self.temporal   = TemporalTransformer(dim=256, n_heads=4, n_layers=2, dropout=dropout)
        self.classifier = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1))
        self.embed_dim  = 256

    def _forward_pair(self, app, mot):
        a1 = self.app_block1(app); m1 = self.att_mask1(a1); a1 = self.pool(a1)
        a2 = self.app_block2(a1);  m2 = self.att_mask2(a2); a2 = self.pool(a2)
        a3 = self.app_block3(a2);  m3 = self.att_mask3(a3); a3 = self.pool(a3)
        a4 = self.app_block4(a3);  m4 = self.att_mask4(a4)
        v1 = self.mot_block1(mot) * m1; v1 = self.pool(v1)
        v2 = self.mot_block2(v1) * m2;  v2 = self.pool(v2)
        v3 = self.mot_block3(v2) * m3;  v3 = self.pool(v3)
        v4 = self.mot_block4(v3) * m4
        return self.frame_feat(self.gap(v4).flatten(1))

    def forward_features(self, app_clip, mot_clip):
        B, T = app_clip.shape[:2]
        app_flat = app_clip.reshape(B * T, *app_clip.shape[2:])
        mot_flat = mot_clip.reshape(B * T, *mot_clip.shape[2:])
        CHUNK = 64
        if app_flat.shape[0] <= CHUNK:
            feats = self._forward_pair(app_flat, mot_flat)
        else:
            parts = []
            for i in range(0, app_flat.shape[0], CHUNK):
                parts.append(self._forward_pair(app_flat[i:i+CHUNK], mot_flat[i:i+CHUNK]))
            feats = torch.cat(parts, dim=0)
        return self.temporal(feats.reshape(B, T, -1))

class CrossAttentionFusion_v2(nn.Module):
    def __init__(self, spatial_dim, rppg_dim, proj_dim=384, n_heads=4, dropout=0.25):
        super().__init__()
        self.spatial_proj = nn.Sequential(nn.Linear(spatial_dim, proj_dim), nn.LayerNorm(proj_dim))
        self.rppg_proj    = nn.Sequential(nn.Linear(rppg_dim,   proj_dim), nn.LayerNorm(proj_dim))

        self.attn_s2r = nn.MultiheadAttention(proj_dim, n_heads, dropout=dropout, batch_first=True)
        self.attn_r2s = nn.MultiheadAttention(proj_dim, n_heads, dropout=dropout, batch_first=True)

        self.ffn_s = nn.Sequential(nn.Linear(proj_dim, proj_dim), nn.GELU(), nn.Dropout(dropout))
        self.ffn_r = nn.Sequential(nn.Linear(proj_dim, proj_dim), nn.GELU(), nn.Dropout(dropout))
        self.norm_s = nn.LayerNorm(proj_dim)
        self.norm_r = nn.LayerNorm(proj_dim)

        self.reliability_gate = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(proj_dim // 2, 1), nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(proj_dim * 2),
            nn.Linear(proj_dim * 2, proj_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(proj_dim // 2, 64), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1)
        )

    def forward(self, spatial_embed, rppg_embed, return_gate=False):
        s = self.spatial_proj(spatial_embed).unsqueeze(1)
        r = self.rppg_proj(rppg_embed).unsqueeze(1)

        s_att, _ = self.attn_s2r(s, r, r)
        r_att, _ = self.attn_r2s(r, s, s)

        s_out = self.norm_s(s.squeeze(1) + self.ffn_s(s_att.squeeze(1)))
        r_out = self.norm_r(r.squeeze(1) + self.ffn_r(r_att.squeeze(1)))

        gate = self.reliability_gate(torch.cat([s_out, r_out], dim=1))
        fused = torch.cat([s_out, gate * r_out], dim=1)

        logits = self.classifier(fused).squeeze(-1)
        if return_gate:
            return logits, gate.squeeze(-1)
        return logits

class DeepfakeFusionModel(nn.Module):
    def __init__(self, sp_backbone, rppg_backbone, sp_dim=1024, rppg_dim=256):
        super().__init__()
        self.spatial_backbone = sp_backbone
        self.rppg_backbone    = rppg_backbone
        self.fusion           = CrossAttentionFusion_v2(sp_dim, rppg_dim)

    def encode_spatial(self, frames):
        B, N = frames.shape[:2]
        flat = frames.reshape(B * N, *frames.shape[2:])
        feats = self.spatial_backbone(flat)
        return feats.reshape(B, N, -1).mean(1)

    def encode_rppg(self, app, mot):
        return self.rppg_backbone.forward_features(app, mot)

    def forward(self, sp_frames, app, mot, return_gate=False):
        s = self.encode_spatial(sp_frames)
        r = self.encode_rppg(app, mot)
        return self.fusion(s, r, return_gate=return_gate)

def build_fusion_model():
    sp_full = SpatialTransformerDetector(pretrained=False)
    sp_backbone = sp_full.backbone
    rppg_backbone = DeepPhysCAN_v2()
    model = DeepfakeFusionModel(sp_backbone, rppg_backbone, 1024, 256)
    return model
