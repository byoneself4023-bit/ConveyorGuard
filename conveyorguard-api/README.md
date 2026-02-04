# 🏭 ConveyorGuard ML API

이송장치 열화 예측 API 서버

## 📁 폴더 구조

```
ConveyorGuard/                    ← 프로젝트 루트
├── data/
│   └── models/
│       └── best_model.pt        ← 모델 파일 (여기!)
│
├── conveyorguard-api/           ← 이 폴더 (여기서 실행)
│   ├── app/
│   │   ├── api/
│   │   │   ├── router.py
│   │   │   └── schemas.py
│   │   ├── core/
│   │   │   ├── loader.py
│   │   │   ├── model.py
│   │   │   └── preprocessing.py
│   │   └── main.py
│   ├── requirements.txt
│   └── README.md
│
└── ml-service/                  ← 기존 코드 (학습용)
```

## 🚀 실행 방법

```bash
# 1. 폴더 이동
cd ConveyorGuard/conveyorguard-api

# 2. 설치
pip install -r requirements.txt

# 3. 서버 실행
MODEL_PATH=../data/models/best_model.pt uvicorn app.main:app --reload --port 8000
```

## 📡 API 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/` | 서비스 정보 |
| GET | `/api/v1/health` | 헬스 체크 |
| GET | `/api/v1/model/info` | 모델 정보 |
| POST | `/api/v1/predict` | 단일 예측 |
| POST | `/api/v1/predict/batch` | 배치 예측 |

## 📊 API 문서

http://localhost:8000/docs

## 🔧 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| MODEL_PATH | ../data/models/best_model.pt | 모델 경로 |
