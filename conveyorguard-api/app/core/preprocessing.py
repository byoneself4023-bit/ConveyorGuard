"""
ConveyorGuard - Preprocessing Utilities
입력 데이터 전처리 (정규화)
"""

import numpy as np
from typing import Dict, Union


# 센서 정규화 범위
SENSOR_RANGES = {
    0: (0, 200),    # NTC (온도)
    1: (0, 1000),   # PM1.0
    2: (0, 1000),   # PM2.5
    3: (0, 1000),   # PM10
    4: (0, 200),    # CT1 (전류)
    5: (0, 200),    # CT2
    6: (0, 200),    # CT3
    7: (0, 200),    # CT4
}

# 외부환경 정규화 범위
EXTERNAL_RANGES = np.array([
    [0, 50],    # 온도 (℃)
    [0, 100],   # 습도 (%)
    [0, 1000],  # 조도 (lux)
], dtype=np.float32)

# 라벨 매핑
LABEL_MAP = {
    0: "정상",
    1: "경미한 열화",
    2: "중간 열화",
    3: "심각한 열화"
}


def normalize_sensors(data: np.ndarray) -> np.ndarray:
    """센서 데이터 Min-Max 정규화"""
    result = data.copy().astype(np.float32)
    for i, (min_v, max_v) in SENSOR_RANGES.items():
        result[..., i] = (data[..., i] - min_v) / (max_v - min_v + 1e-8)
    return np.clip(result, 0, 1)


def normalize_images(data: np.ndarray) -> np.ndarray:
    """이미지 데이터 Min-Max 정규화 (이미지별)"""
    result = np.zeros_like(data, dtype=np.float32)
    if data.ndim == 3:
        for i in range(len(data)):
            min_v, max_v = data[i].min(), data[i].max()
            if max_v - min_v > 0:
                result[i] = (data[i] - min_v) / (max_v - min_v)
    elif data.ndim == 4:
        for b in range(len(data)):
            for t in range(data.shape[1]):
                min_v, max_v = data[b, t].min(), data[b, t].max()
                if max_v - min_v > 0:
                    result[b, t] = (data[b, t] - min_v) / (max_v - min_v)
    return result


def normalize_external(data: np.ndarray) -> np.ndarray:
    """외부환경 데이터 Min-Max 정규화"""
    result = (data - EXTERNAL_RANGES[:, 0]) / (EXTERNAL_RANGES[:, 1] - EXTERNAL_RANGES[:, 0] + 1e-8)
    return np.clip(result, 0, 1).astype(np.float32)


def preprocess_input(
    sensors: Union[np.ndarray, list],
    images: Union[np.ndarray, list],
    external: Union[np.ndarray, list],
    normalize: bool = True
) -> Dict[str, np.ndarray]:
    """전체 입력 전처리 파이프라인"""
    sensors = np.array(sensors, dtype=np.float32)
    images = np.array(images, dtype=np.float32)
    external = np.array(external, dtype=np.float32)
    
    if normalize:
        sensors = normalize_sensors(sensors)
        images = normalize_images(images)
        external = normalize_external(external)
    
    return {
        'sensors': sensors,
        'images': images,
        'external': external
    }
