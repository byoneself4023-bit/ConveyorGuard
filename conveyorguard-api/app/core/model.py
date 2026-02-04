"""
ConveyorGuard - Multimodal Transformer Model
기존 ml-service/app/models의 컴포넌트들을 조합한 통합 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    """모델 설정 (학습 시 사용한 값)"""
    seq_len: int = 30
    sensor_dim: int = 8
    external_dim: int = 3
    image_height: int = 60
    image_width: int = 80
    num_classes: int = 4
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1


class SensorEncoder(nn.Module):
    """센서 인코더: (B, T, 8) → (B, T, embed_dim)"""
    
    def __init__(self, input_dim: int, embed_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImageEncoder(nn.Module):
    """이미지 인코더 (CNN): (B, T, H, W) → (B, T, embed_dim)"""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W = x.shape
        x = x.view(B * T, 1, H, W)
        x = self.cnn(x).view(B * T, -1)
        x = self.fc(x)
        return x.view(B, T, -1)


class TemporalEncoder(nn.Module):
    """시계열 인코더 (Transformer): (B, T, D) → (B, T, D)"""
    
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int, 
                 dropout: float, max_len: int = 30):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed[:, :x.size(1), :]
        return self.transformer(x)


class MultimodalTransformer(nn.Module):
    """
    멀티모달 Transformer 모델
    
    입력:
        - sensor: (B, 30, 8) - 센서 시퀀스
        - image: (B, 30, 60, 80) - 열화상 시퀀스  
        - external: (B, 30, 3) - 외부환경 시퀀스
    
    출력:
        - logits: (B, 4) - 4-class 분류 로짓
    """
    
    def __init__(self, config: Config = None):
        super().__init__()
        
        if config is None:
            config = Config()
        
        c = config
        self.config = c
        
        # Encoders
        self.sensor_encoder = SensorEncoder(c.sensor_dim, c.embed_dim, c.dropout)
        self.image_encoder = ImageEncoder(c.embed_dim)
        self.external_encoder = nn.Sequential(
            nn.Linear(c.external_dim, c.embed_dim),
            nn.LayerNorm(c.embed_dim),
            nn.GELU()
        )
        
        # Temporal Encoders
        self.temporal = TemporalEncoder(
            c.embed_dim, c.num_heads, c.num_layers, c.dropout
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(c.embed_dim * 3, c.embed_dim),
            nn.LayerNorm(c.embed_dim),
            nn.GELU(),
            nn.Dropout(c.dropout)
        )
        
        # Attention Pooling
        self.attn_pool = nn.Sequential(
            nn.Linear(c.embed_dim, c.embed_dim // 4),
            nn.Tanh(),
            nn.Linear(c.embed_dim // 4, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(c.embed_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(64, c.num_classes)
        )
    
    def forward(self, sensor: torch.Tensor, image: torch.Tensor, 
                external: torch.Tensor) -> torch.Tensor:
        # Encode
        s = self.sensor_encoder(sensor)
        i = self.image_encoder(image)
        e = self.external_encoder(external)
        
        # Temporal encoding
        s = self.temporal(s)
        i = self.temporal(i)
        
        # Fusion
        x = torch.cat([s, i, e], dim=-1)
        x = self.fusion(x)
        
        # Attention Pooling
        attn = F.softmax(self.attn_pool(x), dim=1)
        x = (x * attn).sum(dim=1)
        
        # Classify
        logits = self.classifier(x)
        
        return logits
    
    def predict(self, sensor: torch.Tensor, image: torch.Tensor,
                external: torch.Tensor) -> dict:
        """예측 + 확률 반환"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(sensor, image, external)
            probs = F.softmax(logits, dim=-1)
            pred = logits.argmax(dim=-1)
            confidence = probs.max(dim=-1).values
        
        return {
            'prediction': pred,
            'confidence': confidence,
            'probabilities': probs
        }
