# 🏓 탁구 AI 물리 엔진 연구 노트 (MLP Physics Decoder)

> 기존 FAISS 기반의 초경량 매핑(Mapping) 디코더 방식은 139개라는 실측 데이터셋(TT3D)의 절대적인 수량 부족으로 한계에 부딪혔습니다.
> **지도교수님과의 미팅 후, 연구 방향을 "수학적 물리 엔진으로 방대한 합성 데이터를 생성하고, 이를 모사하는 MLP 딥러닝 기반의 AI 물리 디코더 구현"으로 전면 수정하였습니다.**
> 
> **마지막 업데이트: 2026-05-25**

---

## 🎯 최종 목표 (연구의 본질)
복잡하고 무거운 수학적 물리 연산(SciPy `solve_ivp`, RK45)을 매 런타임마다 수행하는 대신, 물리 법칙의 결과를 학습한 **인공지능 모델(MLP)**이 단 한 번의 추론(Inference)으로 500스텝의 3D 궤적을 즉시 렌더링하도록 만듭니다.

**핵심 철학: Physics as a Neural Network**
- **데이터 무한 생성:** 공기저항, 마그누스 효과, 중력, 탁구대 경계, 말랑한 네트 충돌(Soft-wall), 바닥 추락 등의 법칙이 적용된 정밀한 수학적 시뮬레이터로 200만 개의 합성 데이터를 무한정 찍어냅니다.
- **다차원 학습:** 생성된 방대한 8차원 공간의 파라미터(입력)와 500스텝 궤적(출력)의 관계를 AI가 근사(Approximation)하도록 학습시킵니다.
- **초고속 렌더링:** 런타임에는 미분방정식 없이 행렬 곱셈만으로 궤적을 뱉어내어, 물리 엔진 대비 속도 향상을 이뤄냅니다. (압도적인 속도는 이 연구의 본질입니다.)

---

## 🏗️ 파이프라인 설계

### Step 1. 시뮬레이터 설계 및 데이터 생성 (Data Generation)
- **파일:** `generate_dataset.py`, `app.py` (내부 `traditional_physics_solver`)
- **수학적 물리 엔진의 진화 (World Coordinate Physics):**
  - 기존의 Z=0 무한 평면 반사를 폐기하고, **절대 월드 좌표계**를 도입했습니다.
  - **탁구대 경계:** `|X| <= 0.76m`, `|Y| <= 1.37m` 내에서만 Z=0 충돌 감지. 아웃되면 바닥(`Z=-0.76m`)까지 자유 낙하.
  - **말랑한 네트(Soft Wall):** 공이 중앙(`Y=0`)을 지날 때 네트 높이(`Z <= 0.1525m`) 이하면 네트에 걸리며, 네트 장력을 받아 `Vy`가 반대로 꺾이면서 감쇠되는 현실적인 텐션 로직 구현.
- **데이터 추출:** 랜덤하게 샘플링된 200만 개의 8D 입력을 물리 엔진에 넣어 정답지(1500D) 세트를 추출. (`dataset_random.npz`로 저장)

### Step 2. AI 물리 모델 아키텍처 진화 (Evolution of Physics Decoder)
- **파일:** `train_baseline.py`
- **현재 구조 (Expanding MLP):**
  - 병목(Bottleneck) 현상을 제거하고 차원이 점진적으로 넓어지는 순수 확장형 구조 (`8 -> 256 -> 512 -> 1024 -> 1500`) 도입.
  - 기교(ResNet 등) 없이 가장 단순한 형태의 다층 퍼셉트론으로 회귀.
- **손실 함수 (Loss):** 기본 L2 Loss (MSE Loss) 적용
- **데이터셋 처리:** 200만 개 데이터 Z-score 정규화 (`mlp_norm_standard_random.npy`).

### Step 3. 비교 검증 UI (Gradio Dashboard)
- **파일:** `app.py`
- 런타임에 사용자가 8차원 슬라이더를 조절하면, **[수학적 물리 엔진]**과 **[AI 물리 디코더]**가 동시에 궤적을 그리고 그 L2 오차(Error)와 속도 차이(Speedup)를 화면에 3D로 비교 렌더링합니다.

---

## 📐 AI 입출력 파라미터 (8D -> 1500D)

### 입력: 8D Input Features (World Coordinates)
| 인덱스 | 파라미터 | 설명 | 범위 |
|---|---|---|---|
| 0 | speed | 초기 타격 속도 (m/s) | 1 ~ 100 |
| 1 | θ_v | 수직 발사각 (°) | -85 ~ 85 |
| 2 | ω_top | 탑스핀 (rad/s, + 탑 / - 백) | -800 ~ 800 |
| 3 | ω_side | 사이드스핀 (rad/s) | -800 ~ 800 |
| 4 | hit_x | 타격 좌우 위치 (m) | -3 ~ 3 |
| 5 | hit_y | 타격 앞뒤 위치 (m) | -4 ~ 0 |
| 6 | hit_z | 타격 높이 (m) | -0.7 ~ 2.0 |
| 7 | θ_h | 수평 타격 방향각 (°) | -80 ~ 80 |

### 출력: 1500D Output Trajectory
- 500 스텝 × 3차원 좌표 (X, Y, Z) = 1500개의 Float 값.
- 시간 간격 `dt = 0.002초`, 총 1.0초 길이의 궤적.

---

## 🧠 시행착오 및 실험 기록 (Trial & Error Log)

스무딩 현상(U자 궤적)과 궤도 불안정 문제를 해결하기 위해 진행한 실험들을 이곳에 기록합니다.

1. **시도 1 (Reboot - Back to Basics):**
   - **날짜:** 2026-05-26
   - **내용:** 순수 Expanding MLP (`1024 -> 1500`) + 기본 MSE Loss
   - **결과:** 엄청난 파도타기(심각한 궤도 구불거림, Wavy artifacts) 현상이 발생함. 병목(Bottleneck)을 제거하고 차원을 크게 넓혔음에도 궤도가 마구잡이로 요동침.
   ![시도 1 진짜 궤적: 심각한 파도타기 현상](/root/myresearch/attempt1_protoMLP.png)

2. **시도 2 (Kinematic Loss 도입):**
   - **날짜:** 2026-05-27
   - **내용:** Expanding MLP (`1024 -> 1500`) + MSE Loss + **Kinematic Loss (L1)**
   - **결과:** 구불거림(파도타기)은 완전히 사라지고 궤적이 엄청나게 부드러워졌으나, **가장 중요한 바운드(탁구대 충돌) 현상 자체가 아예 실종됨!** 궤적이 바닥을 찍지 않고 공중에서 부드럽게 둥둥 떠가는 궤적이 생성됨.
   ![시도 2 진짜 궤적: 바운드 실종 및 과도한 스무딩](/root/myresearch/attempt2_kinematic.png)

3. **시도 3 (Chamfer Distance 도입):**
   - **날짜:** 2026-05-28
   - **내용:** Expanding MLP (`1024 -> 1500`) + MSE Loss + **Chamfer Distance**
   - **결과:** 형태(V자)만 억지로 맞추려다 보니 **시간(시계열) 개념이 완전히 붕괴됨**. 점들이 궤적을 따라 순서대로 배치되지 않고 바운드 지점 근처에 무질서하게 뭉치면서(Point Cloud 형태로 산산조각 남) **형태도, 애니메이션 속도도 모두 파탄 나는 대실패**를 겪음.
   ![시도 3 궤적 파탄](/root/myresearch/attempt3_chamfer_fail.png)

4. **시도 4 (ResNet 구조 도입 + L1 Loss):**
   - **날짜:** 2026-05-28
   - **내용:** Expanding MLP 뼈대 골목 사이에 **ResBlock(Skip Connection, Pre-Activation)** 삽입 + 순수 **L1 Loss(MAE)** 교체
   - **목적:** 손실 함수 조작으로는 본질적인 불확실성(평균화)을 피할 수 없다는 결론 하에, 뼈대 자체가 깊은 층에서도 날카로운 정보(High-frequency)를 소실 없이 끝까지 전달할 수 있도록 아키텍처를 진화시킴. 추가로 L1 Loss로 중앙값을 강제 추적함.
   - **결과:** **울퉁불퉁한 지렁이 궤적 발생 (Wobbling/Jittering)**. ResNet 덕분에 '날카로운(High-frequency)' 정보를 표현할 근육은 생겼고 L1 Loss 덕분에 둥글게 퉁치려는 현상은 막았으나, 데이터 밀도가 너무 낮아(차원의 저주) 바운드 타이밍을 정확히 확신하지 못함. 결과적으로 이리저리 찍어 맞추려다 바운드 지점 근처에서 심하게 요동치는(출렁거리는) 기괴한 궤적이 만들어짐.
   ![시도 4 지렁이 궤적](/root/myresearch/attempt4_resnet_fail.png)

ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ데이터 증강 및 파라미터 범위 축소ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
이전 파라미터 :
speed = np.random.uniform(1.0, 100.0)
        theta_v = np.random.uniform(-85.0, 85.0)
        omega_top = np.random.uniform(-800.0, 800.0)
        omega_side = np.random.uniform(-800.0, 800.0)
        hit_x = np.random.uniform(-3.0, 3.0)
        hit_y = np.random.uniform(-4.0, 0.0)
        hit_z = np.random.uniform(-0.7, 2.0)
        theta_h = np.random.uniform(-80.0, 80.0)

이후 파라미터 :
speed = np.random.uniform(1.0, 80.0)
        theta_v = np.random.uniform(-50.0, 50.0)
        omega_top = np.random.uniform(-200.0, 200.0)
        omega_side = np.random.uniform(-200.0, 200.0)
        hit_x = np.random.uniform(-3.0, 3.0)
        hit_y = np.random.uniform(-4.0, 0.0)
        hit_z = np.random.uniform(-0.76, 1.5)
        theta_h = np.random.uniform(-65.0, 65.0)

데이터셋 갯수 : 200만개 -> 300만개
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
1. **시도 1 (Reboot - Back to Basics):**
   - **날짜:** 2026-05-26
   - **내용:** 순수 Expanding MLP (`1024 -> 1500`) + 기본 MSE Loss
   - **결과:** 엄청난 파도타기(심각한 궤도 구불거림, Wavy artifacts) 현상이 발생함. 병목(Bottleneck)을 제거하고 차원을 크게 넓혔음에도 궤도가 마구잡이로 요동침.
   ![시도 1 진짜 궤적: 심각한 파도타기 현상](/root/myresearch/attempt1_protoMLP.png)
---

## 📁 디렉토리 구조 (최신화 완료)

```
/root/myresearch/
├── app.py                    # 물리엔진 vs AI 비교 3D 대시보드 (Gradio)
├── generate_dataset.py       # 물리엔진으로 50만개 데이터셋 생성 스크립트
├── train_baseline.py         # AI(Standard MLP) 딥러닝 모델 훈련 스크립트
├── visualize.py              # 궤적 예측 시각화 분석 스크립트
├── RESEARCH_NOTES.md         # 이 연구 노트
└── database/
    ├── dataset_random.npz    # 생성된 50만개 통합 정답지 파일
    ├── norm_params.npy       # 데이터 정규화(Z-score) 파라미터 저장
    ├── mlp_norm_standard_random.npy # 모델 입력 전처리 스케일러 저장
    └── mlp_standard_random.pth # 훈련 완료된 모델 (Standard MLP)
```

---

## ✅ 완료된 작업
- [x] 방향성 변경 승인: 매핑 방식 폐기, 합성 데이터 기반 AI 모델 방식 채택
- [x] 50만 개 데이터셋 생성 및 입력 파라미터 Z-score 정규화 적용
- [x] 스무딩 현상 해결을 위해 진행했던 ResNet/Kinematic Loss 롤백 완료 (기록 체계화 목적)

## 🔜 진행 중인 작업 (현재 상황)
- [ ] Standard MLP 모델의 정상 작동(훈련 및 추론) 확인
- [ ] 베이스라인 모델에서 발생하는 U자 바운스(스무딩 현상) 재확인

## 🔜 다음 할 일
- [ ] 스무딩 현상 해결을 위한 새로운 구조, 피처 엔지니어링, 혹은 새로운 손실 함수 적용 실험
- [ ] 실험 결과를 '시행착오 및 실험 기록'에 꼼꼼히 기록
