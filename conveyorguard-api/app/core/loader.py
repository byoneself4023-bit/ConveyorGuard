"""
ConveyorGuard - Model Loader
학습된 모델 로드 및 관리
"""

import torch
from pathlib import Path
from typing import Optional
import logging

from app.core.model import MultimodalTransformer, Config

logger = logging.getLogger(__name__)


class ModelLoader:
    """모델 로더 싱글톤"""
    
    _instance: Optional['ModelLoader'] = None
    _model: Optional[MultimodalTransformer] = None
    _device: torch.device = None
    _model_info: dict = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load(self, model_path: str) -> MultimodalTransformer:
        """모델 로드"""
        if self._model is not None:
            return self._model
        
        # Device 설정
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self._device}")
        
        # 모델 경로 확인
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Checkpoint 로드
        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self._device, weights_only=False)
        
        # Config 로드
        config = Config()
        if 'config' in checkpoint:
            for key, value in checkpoint['config'].items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # 모델 생성 및 가중치 로드
        self._model = MultimodalTransformer(config)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model.to(self._device)
        self._model.eval()
        
        # 모델 정보 저장
        self._model_info = {
            'path': str(path.absolute()),
            'device': str(self._device),
            'best_accuracy': checkpoint.get('best_acc', 'N/A'),
            'config': checkpoint.get('config', {}),
            'parameters': sum(p.numel() for p in self._model.parameters()),
        }
        
        logger.info(f"Model loaded successfully. Best accuracy: {self._model_info['best_accuracy']}")
        
        return self._model
    
    @property
    def model(self) -> MultimodalTransformer:
        """현재 로드된 모델 반환"""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model
    
    @property
    def device(self) -> torch.device:
        """현재 디바이스 반환"""
        return self._device
    
    @property
    def info(self) -> dict:
        """모델 정보 반환"""
        return self._model_info
    
    def is_loaded(self) -> bool:
        """모델 로드 여부"""
        return self._model is not None


# 전역 모델 로더 인스턴스
model_loader = ModelLoader()
