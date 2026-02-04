"""
ConveyorGuard - Streamlit Demo
이송장치 열화 예측 AI 시스템 데모 (연구 포트폴리오)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# --- 상수 ---
STATE_LABELS = {0: "정상", 1: "경미", 2: "중간", 3: "심각"}
STATE_COLORS = {0: "#22C55E", 1: "#FACC15", 2: "#F97316", 3: "#EF4444"}
STATE_ICONS = {0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴"}
PROJECT_ROOT = Path(__file__).parent.parent

# --- 페이지 설정 ---
st.set_page_config(
    page_title="ConveyorGuard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label { font-size: 0.85rem; }
    .sensor-gauge { margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)

st.title("🏭 ConveyorGuard")
st.caption("이송장치 열화 예측 AI 시스템")

# ============================================================
# 2탭 구성 (연구 포트폴리오 중심)
# ============================================================
tab1, tab2 = st.tabs([
    "🎯 프로젝트 개요",
    "🔬 실험 여정",
])


# ============================================================
# 탭 1: 프로젝트 개요
# ============================================================
with tab1:
    st.subheader("반도체 이송장치 열화 예측 AI 시스템")

    st.markdown("""
반도체 제조라인의 컨베이어 이송장치에서 수집되는 **센서 + 열화상 이미지**를 분석하여,
열화 상태를 **4단계**(정상 / 경미 / 중간 / 심각)로 **사전 예측**합니다.
비계획 정지를 방지하는 **예지보전(Predictive Maintenance)** 시스템입니다.
""")

    # 핵심 수치 메트릭
    ds_col1, ds_col2, ds_col3, ds_col4 = st.columns(4)
    ds_col1.metric("총 프레임", "111,870")
    ds_col2.metric("세션", "341개")
    ds_col3.metric("센서", "8채널")
    ds_col4.metric("불균형 비율", "6.34 : 1")

    st.divider()

    # 클래스 분포 + 데이터 특징
    chart_col, info_col = st.columns([1, 1])

    with chart_col:
        st.markdown("#### 클래스 분포")
        class_df = pd.DataFrame({
            "클래스": ["정상(0)", "경미(1)", "중간(2)", "심각(3)"],
            "프레임 수": [54928, 24081, 24191, 8670],
            "비율": [49.1, 21.5, 21.6, 7.8],
        })
        fig = px.pie(class_df, names="클래스", values="프레임 수",
                     color_discrete_sequence=["#22C55E", "#FACC15", "#F97316", "#EF4444"])
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0),
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with info_col:
        st.markdown("#### 멀티모달 입력")
        st.markdown("""
| 모달리티 | 내용 |
|----------|------|
| **센서** | NTC(온도), PM1.0/2.5/10(미세먼지), CT1~4(전류) |
| **열화상** | 60x80 해상도, 30프레임 시계열 |
| **외부환경** | 온도, 습도, 조도 |
""")
        st.markdown("""
- 30프레임 슬라이딩 윈도우 구조
- 세션 기반 Train/Val/Test 분할 (데이터 누출 방지)
""")

    st.divider()

    # 연구 흐름도 — 핵심 서사 중심
    st.subheader("연구 흐름")

    st.markdown("""
```
데이터 탐색 → 전처리 → DL Baseline → ML 비교 → DL 튜닝 → LLM 비교 → 앙상블 → 최종 결론
                         (93.24%)     (96.89%)                           (96.89%)
```
""")

    st.success("""💡 **핵심 서사**:
DL로 시작 → **ML이 더 좋음(반전)** → DL 튜닝해도 역전 불가 → LLM으로 해석력 보완 → 앙상블도 소용없음
→ **최종: LightGBM(정확도) + CNN(멀티모달) + Gemini(해석)**""")


# ============================================================
# 탭 2: 실험 여정
# ============================================================
with tab2:
    st.subheader("연구 스토리")

    steps = [
        "00 데이터 탐색",
        "01 전처리",
        "02 DL Baseline",
        "03 ML 비교",
        "04 DL 튜닝",
        "05 LLM 비교",
        "06 앙상블",
        "07 최종 결론",
    ]
    step = st.radio("실험 단계", steps, horizontal=True, key="exp_step")
    step_idx = steps.index(step)

    st.divider()

    # --- Step 00: 데이터 탐색 ---
    if step_idx == 0:
        st.markdown("### 00. 데이터 탐색 (EDA)")
        st.markdown("111,870 프레임의 멀티모달 센서 데이터 구조와 이상 패턴을 파악합니다.")

        c1, c2 = st.columns(2)
        with c1:
            # 클래스별 주요 센서 평균값 비교 (노트북 00에서 추출)
            import plotly.graph_objects as go
            sensor_labels = ["NTC", "CT1", "CT2", "PM2.5"]
            class_labels = ["정상", "경미", "중간", "심각"]
            # 노트북 00 클래스별 센서 평균값
            means = {
                "NTC":   [32.5, 38.2, 48.1, 72.6],
                "CT1":   [25.3, 35.8, 62.4, 138.5],
                "CT2":   [22.1, 30.5, 55.2, 98.3],
                "PM2.5": [18.0, 42.5, 128.0, 285.0],
            }
            colors = ["#22C55E", "#FACC15", "#F97316", "#EF4444"]
            fig = go.Figure()
            for i, cls in enumerate(class_labels):
                fig.add_trace(go.Bar(
                    name=cls, x=sensor_labels,
                    y=[means[s][i] for s in sensor_labels],
                    marker_color=colors[i],
                ))
            fig.update_layout(
                barmode="group", title="클래스별 센서 평균값",
                height=300, margin=dict(l=0, r=0, t=30, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            # 센서 상관 히트맵
            sensor_names = ["NTC", "PM1.0", "PM2.5", "PM10", "CT1", "CT2", "CT3", "CT4"]
            corr_matrix = np.array([
                [1.00, 0.12, 0.15, 0.13, 0.45, 0.42, 0.38, 0.40],
                [0.12, 1.00, 0.95, 0.92, 0.08, 0.07, 0.06, 0.07],
                [0.15, 0.95, 1.00, 0.97, 0.10, 0.09, 0.08, 0.09],
                [0.13, 0.92, 0.97, 1.00, 0.09, 0.08, 0.07, 0.08],
                [0.45, 0.08, 0.10, 0.09, 1.00, 0.88, 0.85, 0.87],
                [0.42, 0.07, 0.09, 0.08, 0.88, 1.00, 0.90, 0.91],
                [0.38, 0.06, 0.08, 0.07, 0.85, 0.90, 1.00, 0.93],
                [0.40, 0.07, 0.09, 0.08, 0.87, 0.91, 0.93, 1.00],
            ])
            fig = px.imshow(corr_matrix, x=sensor_names, y=sensor_names,
                            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                            title="센서 간 상관관계")
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0),
                              paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        st.info("💡 **발견**: 심각 클래스에서 NTC, CT1, PM2.5 평균이 급등. PM 센서끼리(0.95), CT 센서끼리(0.90) 높은 상관 → 다중공선성 주의 필요.")

        # 심화: 센서 중요도 순위
        st.markdown("#### 센서별 열화 상태 상관도")
        st.markdown("""
| 순위 | 센서 | 상관도 | 의미 |
|------|------|--------|------|
| 1 | **NTC (온도)** | **0.79** | 열화 상태와 가장 강한 상관 |
| 2 | CT2 (전류) | 0.38 | 모터 부하 반영 |
| 3 | CT1 (전류) | 0.34 | 전류 이상 패턴 |
| 4 | PM2.5 | 0.15 | 미세먼지 증가 |

> **심각(3) 클래스 특성**: 세션 전체가 심각인 경우 0개, 하지만 심각 포함 세션은 289개(85%)
> → 세션 단위가 아닌 **윈도우 단위 분류**가 필수
""")

    # --- Step 01: 전처리 ---
    elif step_idx == 1:
        st.markdown("### 01. 전처리")
        st.markdown("시계열 센서 데이터를 모델이 학습할 수 있는 구조로 변환합니다.")

        st.markdown("""
#### 30프레임 슬라이딩 윈도우 구조
```
세션 내 프레임:  [f1] [f2] [f3] ... [f30] [f31] [f32] ...
                 |__________________________|
                          윈도우 1
                      |__________________________|
                               윈도우 2
```
""")

        w_col1, w_col2, w_col3 = st.columns(3)
        w_col1.metric("윈도우 크기", "30 프레임")
        w_col2.metric("세션 수", "341개")
        w_col3.metric("분할 방식", "세션 기반")

        st.markdown("""
| 항목 | 설명 |
|------|------|
| 윈도우 | 30프레임 슬라이딩, stride=1 |
| 분할 | 세션 단위 Train(70%) / Val(15%) / Test(15%) |
| 입력 모달리티 | 센서(8ch) + 열화상(224x224) + 외부환경(온도/습도) |
""")

        st.info("💡 **발견**: 세션 기반 분할로 데이터 누출(data leakage) 방지. 같은 세션의 프레임이 Train과 Test에 동시에 포함되지 않도록 보장.")

    # --- Step 02: DL Baseline ---
    elif step_idx == 2:
        st.markdown("### 02. DL Baseline (CNN + Transformer)")
        st.markdown("3-modal 딥러닝 모델을 baseline으로 구축합니다.")

        st.markdown("""
#### 모델 아키텍처
```
센서 (8ch x 30)  ─→ [1D-CNN + Transformer] ─┐
열화상 (224x224) ─→ [ResNet-18 backbone]     ├→ [Fusion + MLP] → 4-class
외부환경 (2ch)   ─→ [Linear]                 ┘
```
""")

        perf_col1, perf_col2, perf_col3 = st.columns(3)
        perf_col1.metric("Test Accuracy", "93.24%")
        perf_col2.metric("Test F1 Score", "93.09%")
        perf_col3.metric("모델 크기", "13.5 MB")

        # 심화: 학습 곡선 + Confusion Matrix
        detail_col1, detail_col2 = st.columns(2)
        with detail_col1:
            st.markdown("#### 학습 곡선")
            epoch_data = pd.DataFrame({
                "Epoch": [1, 2, 8, 14, 16, 19, 22, 23],
                "Val Acc (%)": [83.1, 88.5, 89.6, 90.5, 92.0, 92.7, 92.7, 93.2],
            })
            fig = px.line(epoch_data, x="Epoch", y="Val Acc (%)",
                          title="Validation Accuracy", markers=True)
            fig.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        with detail_col2:
            # Confusion Matrix (노트북 02 결과)
            st.markdown("#### Confusion Matrix (Test)")
            cm_labels = ["정상", "경미", "중간", "심각"]
            cm_data = np.array([
                [666, 122, 0, 0],
                [20, 336, 15, 0],
                [3, 21, 324, 13],
                [1, 0, 2, 85],
            ])
            fig = px.imshow(cm_data, x=cm_labels, y=cm_labels,
                            color_continuous_scale="Blues", text_auto=True,
                            labels=dict(x="예측", y="실제", color="건수"))
            fig.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0),
                              paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
| 클래스 | Precision | Recall | F1 | 주요 혼동 |
|--------|-----------|--------|----|----------|
| 정상(0) | 0.96 | 0.97 | 0.96 | |
| 경미(1) | 0.88 | 0.87 | 0.88 | 정상→경미 122건 (과탐지) |
| 중간(2) | 0.89 | 0.90 | 0.89 | |
| **심각(3)** | **0.96** | **0.93** | **0.95** | 심각→중간 단 2건 |
""")

        st.info("💡 **발견**: 3-modal 딥러닝으로 93.24% 달성. 괜찮은 성능이지만, 다음 단계에서 ML과 비교합니다.")

        with st.expander("핵심 코드: ConveyorGuardModel"):
            st.code("""class ConveyorGuardModel(nn.Module):
    \"\"\"3-modal fusion: Sensor + Thermal Image + External Environment\"\"\"
    def __init__(self, embed_dim=128, num_classes=4):
        super().__init__()
        self.sensor_encoder = SensorEncoder(input_dim=8, embed_dim=embed_dim)
        self.image_encoder = ImageEncoder(embed_dim=embed_dim)
        self.external_encoder = nn.Sequential(
            nn.Linear(3, embed_dim), nn.LayerNorm(embed_dim), nn.GELU()
        )
        self.sensor_temporal = TemporalEncoder(embed_dim=embed_dim)
        self.image_temporal = TemporalEncoder(embed_dim=embed_dim)
        self.fusion = CrossAttentionFusion(embed_dim=embed_dim)

        # FiLM: Feature-wise Linear Modulation
        self.film_gamma = nn.Linear(embed_dim, embed_dim)
        self.film_beta = nn.Linear(embed_dim, embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim),
            nn.GELU(), nn.Dropout(0.2), nn.Linear(embed_dim, num_classes)
        )

    def forward(self, sensors, images, externals=None):
        sensor_feat = self.sensor_temporal(self.sensor_encoder(sensors))
        image_feat = self.image_temporal(self.image_encoder(images))
        fused = self.fusion(sensor_feat, image_feat)
        pooled = fused.mean(dim=1)

        if externals is not None:  # FiLM conditioning
            ext_feat = self.external_encoder(externals).mean(dim=1)
            pooled = self.film_gamma(ext_feat) * pooled + self.film_beta(ext_feat)

        return self.classifier(pooled)""", language="python")

    # --- Step 03: ML 비교 ---
    elif step_idx == 3:
        st.markdown("### 03. ML 8종 비교")

        st.warning("⚡ **반전!** 전통 ML 모델 LightGBM이 96.89%로 딥러닝(93.24%)을 뛰어넘었습니다!")

        ml_data = pd.DataFrame({
            "모델": ["LightGBM", "XGBoost", "CatBoost", "RandomForest",
                    "DecisionTree", "KNN", "SVM", "Logistic"],
            "정확도": [96.89, 96.70, 96.46, 95.58, 92.97, 89.12, 87.75, 87.31],
        }).sort_values("정확도")

        fig = px.bar(ml_data, x="정확도", y="모델", orientation="h",
                     title="ML 8종 Test Accuracy (%)",
                     text=ml_data["정확도"].apply(lambda x: f"{x:.2f}%"))
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          xaxis_range=[85, 98])
        fig.update_traces(textposition="outside", marker_color="#3B82F6")
        fig.add_vline(x=93.24, line_dash="dash", line_color="#8B5CF6",
                      annotation_text="DL Baseline (93.24%)")
        st.plotly_chart(fig, use_container_width=True)

        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            st.markdown("""
| | DL (CNN+Transformer) | ML (LightGBM) |
|---|---|---|
| **정확도** | 93.24% | **96.89%** |
| **학습 시간** | ~30분 | ~2.7초 |
| **입력** | 센서+열화상+외부환경 | 센서 피처 64개 |
""")
        with comp_col2:
            st.markdown("""
**왜 ML이 이겼을까?**
- 정형 센서 데이터에서는 트리 기반 모델이 유리
- DL은 이미지 모달리티의 기여도가 낮아 오버헤드만 추가
- LightGBM은 피처 엔지니어링된 센서 데이터를 효율적으로 학습
""")

        st.info("💡 **발견**: 멀티모달 DL보다 센서 피처만 쓰는 LightGBM이 3.65%p 더 높다. 이미지가 오히려 노이즈?")

        # 심화: Feature Importance + Confusion Matrix
        st.divider()
        deep_col1, deep_col2 = st.columns(2)

        with deep_col1:
            st.markdown("#### Feature Importance (Top 10)")
            fi_data = pd.DataFrame({
                "피처": ["sensor_NTC_last", "sensor_CT2_diff", "sensor_CT2_std",
                         "sensor_NTC_max", "sensor_CT2_max", "sensor_CT2_mean",
                         "sensor_CT1_max", "sensor_CT1_diff", "sensor_PM10_min",
                         "sensor_PM10_std"],
                "Importance": [0.230, 0.085, 0.067, 0.057, 0.045, 0.042,
                               0.041, 0.035, 0.035, 0.030],
            })
            fig = px.bar(fi_data, x="Importance", y="피처", orientation="h",
                         title="XGBoost Feature Importance")
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              yaxis=dict(autorange="reversed"))
            fig.update_traces(marker_color="#F97316")
            st.plotly_chart(fig, use_container_width=True)

        with deep_col2:
            st.markdown("#### Confusion Matrix (LightGBM)")
            cm = np.array([
                [781,  7,  0,  0],
                [  8,349, 14,  0],
                [  0, 17,342,  2],
                [  0,  0,  2, 86],
            ])
            fig = px.imshow(cm,
                            x=["정상", "경미", "중간", "심각"],
                            y=["정상", "경미", "중간", "심각"],
                            text_auto=True, color_continuous_scale="Blues",
                            title="Predicted vs Actual")
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0),
                              paper_bgcolor="rgba(0,0,0,0)",
                              xaxis_title="Predicted", yaxis_title="Actual")
            st.plotly_chart(fig, use_container_width=True)

        # EDA 검증
        st.markdown("#### EDA 인사이트 → 모델 검증")
        st.markdown("""
| EDA 상관도 | Feature Importance | 일치 여부 |
|------------|-------------------|-----------|
| NTC 0.79 (1위) | NTC_last 0.23 (1위) | ✅ |
| CT2 0.38 (2위) | CT2_diff 0.09 (2위) | ✅ |
| CT1 0.34 (3위) | CT1_max, CT1_diff (상위) | ✅ |

> EDA에서 발견한 센서 중요도가 모델의 Feature Importance에서도 그대로 검증됨
""")

        with st.expander("핵심 코드: 멀티모달 → 64개 피처 변환"):
            st.code("""def extract_features(data: dict) -> np.ndarray:
    \"\"\"멀티모달 시계열 → 64차원 피처 벡터 변환
    Sensor (N,30,8) + Image (N,30,60,80) + External (N,30,3) → (N,64)\"\"\"

    sensors = data['sensors']
    # 센서: 6종 통계 x 8채널 = 48개 피처
    sensor_mean = sensors.mean(axis=1)
    sensor_std  = sensors.std(axis=1)
    sensor_max  = sensors.max(axis=1)
    sensor_min  = sensors.min(axis=1)
    sensor_last = sensors[:, -1, :]
    sensor_diff = sensors[:, -1, :] - sensors[:, 0, :]  # 시간 변화량

    # 열화상: 공간+시간 통계 = 7개 피처
    images = data['images']
    img_frame_mean = images.mean(axis=(2, 3))
    img_frame_max  = images.max(axis=(2, 3))
    img_mean  = img_frame_mean.mean(axis=1, keepdims=True)
    img_max   = img_frame_max.max(axis=1, keepdims=True)
    img_trend = img_frame_mean[:, -1:] - img_frame_mean[:, 0:1]  # 열 변화 추세

    # 외부환경: 3종 통계 x 3채널 = 9개 피처
    external = data['external']
    ext_mean = external.mean(axis=1)
    ext_std  = external.std(axis=1)
    ext_last = external[:, -1, :]

    return np.concatenate([
        sensor_mean, sensor_std, sensor_max, sensor_min, sensor_last, sensor_diff,
        img_mean, img_max, img_trend,
        ext_mean, ext_std, ext_last
    ], axis=1)  # (N, 64)""", language="python")

    # --- Step 04: DL 튜닝 ---
    elif step_idx == 4:
        st.markdown("### 04. DL 튜닝 (Optuna + Ablation Study)")
        st.markdown("DL 성능을 끌어올려 ML을 역전할 수 있을까?")

        # Ablation Study
        ablation_df = pd.DataFrame({
            "구성": ["Sensor Only", "Image Only", "Sensor+Image", "Full+FiLM"],
            "정확도": [89.12, 69.56, 89.64, 90.35],
        })
        fig = px.bar(ablation_df, x="구성", y="정확도",
                     title="Ablation Study: 모달리티별 기여도",
                     text=ablation_df["정확도"].apply(lambda x: f"{x:.2f}%"),
                     color="정확도",
                     color_continuous_scale=["#EF4444", "#FACC15", "#22C55E"])
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          yaxis_range=[60, 98], showlegend=False)
        fig.update_traces(textposition="outside")
        fig.add_hline(y=96.89, line_dash="dash", line_color="#3B82F6",
                      annotation_text="LightGBM (96.89%)")
        st.plotly_chart(fig, use_container_width=True)

        t_col1, t_col2 = st.columns(2)
        with t_col1:
            st.markdown("""
#### Optuna 탐색 공간 & 최적값

| 파라미터 | 범위 | 최적값 |
|----------|------|--------|
| embed_dim | [128, 256] | **256** |
| num_heads | [4, 8] | **4** |
| num_layers | [1, 2] | **2** |
| dropout | [0.1, 0.3] | **0.1** |
| lr | 1e-4 ~ 1e-3 | **1.96e-4** |
| weight_decay | 1e-5 ~ 1e-3 | **5.4e-5** |
""")
        with t_col2:
            st.markdown("""
#### Trial 결과

| Trial | Val Acc | 상태 |
|-------|---------|------|
| 0 | 88.74% | 완료 |
| 1 | 89.83% | 완료 |
| **2** | **90.48%** | **Best** |
| 3 | 90.09% | 완료 |
| 4-7 | - | Pruned |

**Pruning 효과**: MedianPruner로 비효율 Trial 조기 종료 → GPU 시간 절약
""")

        st.warning("💡 **발견**: 센서가 지배적(89%), 이미지 기여 미미(+0.5%p). Optuna 튜닝으로도 LightGBM 96.89%를 역전 불가.")

        with st.expander("핵심 코드: Optuna objective (Multi-GPU + AMP)"):
            st.code("""def objective(trial):
    # 하이퍼파라미터 탐색
    embed_dim = trial.suggest_categorical('embed_dim', [128, 256])
    num_heads = trial.suggest_categorical('num_heads', [4, 8])
    num_layers = trial.suggest_int('num_layers', 1, 2)
    dropout = trial.suggest_float('dropout', 0.1, 0.3, step=0.1)
    lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)

    # DataParallel (T4 x2) + AMP (Mixed Precision)
    model = ConveyorGuardModel(embed_dim=embed_dim, num_heads=num_heads,
                                num_layers=num_layers, dropout=dropout)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    scaler = GradScaler()

    best_acc = 0
    for epoch in range(FIXED_EPOCHS):
        train_epoch_optimized(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = evaluate_optimized(model, val_loader, criterion, device)

        trial.report(val_acc, epoch)  # Optuna pruning
        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_acc > best_acc:
            best_acc = val_acc

    del model; gc.collect(); torch.cuda.empty_cache()
    return best_acc""", language="python")

    # --- Step 05: LLM 비교 ---
    elif step_idx == 5:
        st.markdown("### 05. LLM 3종 비교 + LangGraph 멀티 에이전트")
        st.markdown("정확도에서 ML에 밀린 한계를 **해석력**으로 보완합니다.")

        llm_df = pd.DataFrame({
            "모델": ["Gemini 2.5 Flash", "Gemma-3-4B", "Qwen2.5-3B"],
            "응답시간(s)": [7.9, 14.6, 8.1],
            "JSON 출력": ["우수", "불안정", "보통"],
            "진단 정확도": ["정확하고 간결", "장황하고 부정확", "간결하지만 부정확"],
            "결과": ["✅ 채택", "❌", "❌"],
        })
        st.dataframe(llm_df, use_container_width=True, hide_index=True)

        l_col1, l_col2 = st.columns(2)
        with l_col1:
            fig = px.bar(
                pd.DataFrame({
                    "모델": ["Gemini 2.5 Flash", "Gemma-3-4B", "Qwen2.5-3B"],
                    "응답시간(s)": [7.9, 14.6, 8.1],
                }),
                x="모델", y="응답시간(s)", title="LLM 응답 시간 비교",
                text="응답시간(s)",
                color="모델",
                color_discrete_sequence=["#22C55E", "#EF4444", "#F97316"],
            )
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              showlegend=False)
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        with l_col2:
            st.markdown("#### T4 GPU 제약 → 모델 선택")
            st.markdown("""
| 모델 | VRAM | T4 16GB | 선택 |
|------|------|---------|------|
| Gemma-3-12B | ~24GB | ❌ OOM | - |
| Qwen2.5-7B | ~14GB | 빠듯 | - |
| **Gemma-3-4B** | ~8GB | ✅ | GPU 0 |
| **Qwen2.5-3B** | ~6GB | ✅ | GPU 1 |

4B + 3B 조합 = T4 x2에 각각 배치 가능한 최대 크기
""")

        # 심각 케이스 실제 응답 비교
        st.divider()
        st.markdown("#### 실제 진단 응답 비교 (심각 케이스)")
        st.markdown("입력: `NTC 0.3°C, CT1 0.0A, 열화상 max 1.0°C` → 설비 미가동 추정")

        resp_col1, resp_col2, resp_col3 = st.columns(3)
        with resp_col1:
            st.markdown("**Gemini 2.5 Flash** ✅")
            st.caption("장비 미가동 또는 전원 이상 추정. CT 전류 0.0A 및 낮은 온도가 지표. 전원 및 장비 작동 상태를 점검하고 필요 시 전원 공급 조치.")
        with resp_col2:
            st.markdown("**Gemma-3-4B** ❌")
            st.caption("응답 실패 (N/A)")
        with resp_col3:
            st.markdown("**Qwen2.5-3B** △")
            st.caption("장비 온도 제어 불량, CT1 과부하... (반복 문구, 품질 낮음)")

        st.info("💡 **발견**: Gemini 2.5 Flash가 속도/품질/안정성 모두 1위. LLM은 정확도가 아닌 **해석력** 보완 역할.")

        # LangGraph 멀티 에이전트
        st.divider()
        st.markdown("#### LangGraph 멀티 에이전트 워크플로우")
        st.markdown("단순 LLM 호출이 아닌, 실제 유지보수 프로세스를 반영한 **에이전트 오케스트레이션**.")

        st.markdown("""
```
         ┌─────────┐
         │  START  │
         └────┬────┘
              v
       ┌──────────────┐
       │   Analyzer   │  ← 센서 데이터 정상/이상 분석
       └──────┬───────┘
              v
       ┌──────────────┐
       │  Diagnoser   │  ← 이상 원인 추정 (3가지 이내)
       └──────┬───────┘
              v
       ┌──────────────┐
       │   Advisor    │  ← 유지보수 조치 추천
       └──────┬───────┘
              v
       ┌──────────────┐
       │   Reviewer   │  ← 진단 품질 검증
       └──────┬───────┘
              │
     ┌────────┴────────┐
     │                 │
   REVISE           APPROVE
     │                 │
     v                 v
  Diagnoser        Finalize
  (재진단)         (리포트 생성)
```
""")

        st.markdown("""
| 에이전트 | 역할 | 출력 |
|----------|------|------|
| **Analyzer** | 센서 데이터 정상/이상 판정 | 각 센서별 분석 |
| **Diagnoser** | 이상 원인 추정 (3가지 이내) | 원인 목록 |
| **Advisor** | 즉시/예방/점검 조치 추천 | 조치 사항 |
| **Reviewer** | 진단 품질 검증 → APPROVE 또는 REVISE | 품질 게이트 |

> **실행 결과**: Reviewer가 1회 REVISE → 자동 재진단 → 최종 APPROVE (총 91.5초)
""")

        with st.expander("핵심 코드: LangGraph StateGraph 구성"):
            st.code("""from langgraph.graph import StateGraph, END

class DiagnosisState(TypedDict):
    sensor_data: dict; thermal_data: dict
    analysis: str; diagnosis: str; advice: str
    review: str; review_count: int; final_report: str

def should_continue(state: DiagnosisState) -> str:
    \"\"\"Conditional routing: max 2 revisions\"\"\"
    if state['review_count'] >= 2:
        return 'end'
    if 'APPROVE' in state['review'].upper():
        return 'end'
    return 'revise'

# Build workflow graph
workflow = StateGraph(DiagnosisState)
workflow.add_node('analyzer', analyzer_node)
workflow.add_node('diagnoser', diagnoser_node)
workflow.add_node('advisor', advisor_node)
workflow.add_node('reviewer', reviewer_node)
workflow.add_node('finalize', finalize_node)

workflow.set_entry_point('analyzer')
workflow.add_edge('analyzer', 'diagnoser')
workflow.add_edge('diagnoser', 'advisor')
workflow.add_edge('advisor', 'reviewer')
workflow.add_conditional_edges('reviewer', should_continue,
    {'revise': 'diagnoser', 'end': 'finalize'})
workflow.add_edge('finalize', END)

app = workflow.compile()""", language="python")

    # --- Step 06: 앙상블 ---
    elif step_idx == 6:
        st.markdown("### 06. 앙상블 (Stacking)")
        st.markdown("여러 모델을 결합하면 LightGBM을 뛰어넘을 수 있을까?")

        # 전체 9종 순위 테이블
        ens_all = pd.DataFrame({
            "모델": ["LightGBM", "Stacking", "XGBoost", "Weighted Voting",
                     "Soft Voting", "CatBoost", "RandomForest", "Baseline CNN", "Tuned CNN"],
            "유형": ["ML", "Ensemble", "ML", "Ensemble",
                     "Ensemble", "ML", "ML", "DL", "DL"],
            "Test Acc (%)": [96.89, 96.89, 96.70, 96.70,
                            96.64, 96.46, 95.58, 92.72, 87.75],
        })
        fig = px.bar(ens_all.sort_values("Test Acc (%)"),
                     x="Test Acc (%)", y="모델", color="유형", orientation="h",
                     title="앙상블 포함 전체 모델 순위",
                     color_discrete_map={"ML": "#3B82F6", "DL": "#8B5CF6", "Ensemble": "#10B981"},
                     text=ens_all.sort_values("Test Acc (%)")["Test Acc (%)"].apply(lambda x: f"{x:.2f}%"))
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=0),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          xaxis_range=[85, 98])
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
| | LightGBM (단일) | Stacking Ensemble |
|---|---|---|
| **정확도** | 96.89% | 96.89% (동률) |
| **복잡도** | 낮음 (단일 모델) | 높음 (6개 모델 + 메타러너) |
| **추론 시간** | 빠름 | 느림 |
| **결론** | **✅ 단일 모델이 최적** | 성능 이득 없이 복잡도만 증가 |
""")

        # Weighted Voting 가중치 분포
        st.markdown("#### Weighted Voting 가중치 분포")
        st.markdown("""
```
LightGBM:     0.1706  ─────────────────
XGBoost:      0.1707  ─────────────────
CatBoost:     0.1699  ─────────────────
RandomForest: 0.1698  ─────────────────
Baseline CNN: 0.1640  ────────────────
Tuned CNN:    0.1550  ───────────────
```
> ML 4종이 거의 균등한 가중치 → DL의 기여도가 낮아 앙상블 효과 제한적
""")

        st.warning("💡 **발견**: Stacking 앙상블도 96.89%로 LightGBM과 동률. 복잡도만 증가하고 성능 이득 없음 → 단일 LightGBM이 최적.")

    # --- Step 07: 최종 결론 ---
    elif step_idx == 7:
        st.markdown("### 07. 최종 결론: 13개 모델 종합 비교")

        csv_path = PROJECT_ROOT / "data" / "results" / "final" / "final_comparison.csv"

        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df = df.sort_values("Test_Acc", ascending=True).reset_index(drop=True)

            fig = px.bar(
                df,
                x="Test_Acc",
                y="Model",
                color="Type",
                orientation="h",
                title="13개 모델 Test Accuracy (%)",
                color_discrete_map={"ML": "#3B82F6", "DL": "#8B5CF6", "Ensemble": "#10B981"},
                text=df["Test_Acc"].apply(lambda x: f"{x:.2f}%"),
            )
            fig.update_layout(
                height=500,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis_title="",
                xaxis_title="Test Accuracy (%)",
                xaxis_range=[85, 98],
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

            display_df = df.sort_values("Test_Acc", ascending=False).reset_index(drop=True)
            display_df["Rank"] = range(1, len(display_df) + 1)
            table_df = display_df[["Rank", "Model", "Type", "Test_Acc", "Test_F1", "Train_Time"]].copy()
            table_df.columns = ["순위", "모델", "유형", "정확도 (%)", "F1 (%)", "학습시간 (s)"]
            table_df["정확도 (%)"] = table_df["정확도 (%)"].apply(lambda x: f"{x:.2f}")
            table_df["F1 (%)"] = table_df["F1 (%)"].apply(lambda x: f"{x:.2f}")
            st.dataframe(table_df, use_container_width=True, hide_index=True)
        else:
            st.warning(f"파일을 찾을 수 없습니다: {csv_path}")

        st.divider()

        # 성능 향상 스토리 라인 차트
        st.markdown("#### 성능 향상 스토리")
        import plotly.graph_objects as go

        story_data = pd.DataFrame({
            "단계": ["02: DL Baseline", "04: DL Tuned", "03: XGBoost", "06: Stacking"],
            "Acc": [93.24, 87.75, 96.70, 96.89],
            "색상": ["#F59E0B", "#F59E0B", "#3B82F6", "#EF4444"],
        })
        fig = go.Figure(go.Scatter(
            x=list(range(4)), y=story_data["Acc"],
            mode="lines+markers+text",
            text=[f"{m}<br>{a:.1f}%" for m, a in zip(story_data["단계"], story_data["Acc"])],
            textposition="top center",
            marker=dict(size=14, color=story_data["색상"].tolist()),
            line=dict(width=2, color="#6B7280"),
        ))
        fig.update_layout(
            height=350, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(tickvals=list(range(4)), ticktext=story_data["단계"].tolist()),
            yaxis_title="Test Accuracy (%)",
            yaxis=dict(range=[83, 100]),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Accuracy vs Training Time 산점도 (노트북 07)
        st.markdown("#### 정확도 vs 학습 시간")
        type_colors = {"ML": "#3B82F6", "DL": "#F59E0B", "Ensemble": "#10B981"}
        scatter_data = pd.DataFrame([
            {"Model": "LightGBM", "Type": "ML", "Acc": 96.89, "Time": 2.7},
            {"Model": "XGBoost", "Type": "ML", "Acc": 96.70, "Time": 3.8},
            {"Model": "CatBoost", "Type": "ML", "Acc": 96.46, "Time": 21.5},
            {"Model": "RandomForest", "Type": "ML", "Acc": 95.58, "Time": 3.7},
            {"Model": "DL Baseline", "Type": "DL", "Acc": 93.24, "Time": 2178},
            {"Model": "DL Tuned", "Type": "DL", "Acc": 87.75, "Time": 600},
            {"Model": "Stacking", "Type": "Ensemble", "Acc": 96.89, "Time": 10},
        ])
        fig = px.scatter(scatter_data, x="Time", y="Acc", color="Type",
                         text="Model", log_x=True, size_max=14,
                         color_discrete_map=type_colors,
                         labels={"Time": "학습 시간 (초, log)", "Acc": "Test Accuracy (%)"})
        fig.update_traces(textposition="top center", marker=dict(size=12))
        fig.update_layout(
            height=350, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(range=[85, 98]),
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig, use_container_width=True)

        # 레이더 차트 — Top 3 모델 다차원 비교
        st.markdown("#### 다차원 비교 (레이더 차트)")
        categories = ["정확도", "학습 속도", "해석 가능성", "모델 경량성", "멀티모달"]
        radar_models = [
            {"name": "LightGBM (ML)", "vals": [3.96, 5.0, 5.0, 3.0, 2.0], "color": "#3B82F6"},
            {"name": "CNN+Transformer (DL)", "vals": [2.75, 1.5, 1.5, 1.5, 5.0], "color": "#F59E0B"},
            {"name": "Stacking (Ensemble)", "vals": [3.96, 2.0, 3.0, 5.0, 4.0], "color": "#EF4444"},
        ]
        fig = go.Figure()
        for rm in radar_models:
            fig.add_trace(go.Scatterpolar(
                r=rm["vals"] + [rm["vals"][0]],
                theta=categories + [categories[0]],
                fill="toself", name=rm["name"], opacity=0.5,
                line=dict(color=rm["color"], width=2),
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            height=450, margin=dict(l=40, r=40, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.1),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.markdown("#### 최종 배포 전략")
        st.success("""**최종 결론: 3-모델 협업 구조**

| 역할 | 모델 | 이유 |
|------|------|------|
| **정확도 (예측)** | LightGBM (96.89%) | 센서 피처 기반 최고 정확도 |
| **멀티모달 (배포)** | CNN+Transformer (93.24%) | 열화상 이미지 직접 처리 가능 |
| **해석력 (진단)** | Gemini 2.5 Flash | 센서 데이터 해석 + 조치 추천 |
""")

        st.markdown("""
**배포 모델 선택 근거:**

| 항목 | CNN+Transformer (배포) | LightGBM (최고 정확도) |
|------|------------------------|----------------------|
| **정확도** | 93.24% | 96.89% |
| **멀티모달** | 센서 + 열화상 + 외부환경 | 센서만 (피처 추출 필요) |
| **열화상 처리** | CNN으로 직접 처리 | 불가능 |
| **선택 이유** | 열화상 이미지를 직접 활용할 수 있어 실제 환경에서 더 유리 | - |
""")


# --- Footer ---
st.divider()
st.caption("ConveyorGuard v1.0 | LightGBM 96.89% + CNN+Transformer 93.24% + Gemini 2.5 Flash")
