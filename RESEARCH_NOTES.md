# 🏓 탁구 AI 물리 엔진 연구 노트 (MLP Physics Decoder)
> **마지막 업데이트: 2026-06-11**

---

## 🎯 최종 목표 (연구의 본질)
복잡하고 무거운 수학적 물리 연산(SciPy `solve_ivp`, RK45)을 매 런타임마다 수행하는 대신, 물리 법칙의 결과를 학습한 **인공지능 모델**이 빠른 추론(Inference)을 통해 750스텝(1.5초)의 3D 궤적을 즉시 렌더링하도록 구현함.

**핵심 철학: Physics as a Neural Network**
- **데이터 생성:** 공기저항, 마그누스 효과, 중력, 탁구대 경계, 네트 충돌(Soft-wall), 바닥 추락 등의 물리 법칙이 적용된 정밀한 수학적 시뮬레이터로 500만 개의 합성 데이터를 대량 생성함.
- **다차원 학습:** 생성된 방대한 파라미터(입력)와 궤적(출력)의 관계를 AI가 근사(Approximation)하도록 학습시킴.
- **초고속 렌더링:** 런타임에는 미분방정식 연산 없이 행렬 곱셈만으로 궤적을 출력하여, 기존 물리 엔진 대비 비약적인 속도 향상을 달성함. (압도적인 추론 속도 확보가 본 연구의 핵심임)

---

## 🏗️ 파이프라인 설계

### Step 1. 시뮬레이터 설계 및 데이터 생성 (Data Generation)
- **파일:** `generate_dataset.py`, `app.py` (내부 `traditional_physics_solver`)
- **수학적 물리 엔진의 고도화 (World Coordinate Physics):**
  - 기존의 Z=0 무한 평면 반사 가정을 폐기하고, **절대 월드 좌표계**를 도입함.
  - **탁구대 경계:** `|X| <= 0.76m`, `|Y| <= 1.37m` 내에서만 Z=0 충돌을 감지함. 해당 영역을 벗어나면 바닥(`Z=-0.76m`)까지 자유 낙하하도록 구현함.
  - **네트 충돌(Soft Wall):** 공이 중앙(`Y=0`)을 지날 때 네트 높이(`Z <= 0.1525m`) 이하일 경우 네트에 걸리며, 네트 장력을 받아 `Vy`가 반전 및 감쇠되는 현실적인 텐션 로직을 구현함.
- **데이터 추출:** 무작위로 샘플링된 500만 개의 입력을 물리 엔진에 인가하여 정답지 세트를 추출함. (`dataset_gaussian_mixed.npz`로 저장)

### Step 2. AI 물리 모델 아키텍처 진화 (Evolution of Physics Decoder)
- **파일:** `train_baseline.py`
- **최종 구조 (Time-Conditioned MLP):**
  - 시간 특징을 분리하여 입력받는 Time-Conditioned MLP 구조 (29차원 입력 -> 256 -> ResBlock×3 -> 3차원 출력) 도입.
  - 깊은 층에서도 고주파 정보를 유지하기 위해 3개의 Residual Block을 적용함.
- **손실 함수 (Loss):** 순수 L1 Loss (MAE Loss) 적용.
- **데이터셋 처리:** 500만 개 데이터에 대해 Z-score 정규화 수행.

### Step 3. 비교 검증 UI (Gradio Dashboard)
- **파일:** `app.py`
- 런타임에 사용자가 8차원 슬라이더를 조절하면, **[수학적 물리 엔진]**과 **[AI 물리 디코더]**가 동시에 궤적을 추론하고, 그 L2 오차(Error)와 속도 향상률(Speedup)을 화면에 3D로 비교 렌더링함.

---

## 📐 AI 입출력 파라미터 (29D -> 3D)

### 입력: 29D Input Features
**1. 8개 물리 조건 (World Coordinates)**
| 인덱스 | 파라미터 | 설명 | 범위 |
|---|---|---|---|
| 0 | speed | 초기 타격 속도 (m/s) | 1 ~ 80 |
| 1 | θ_v | 수직 발사각 (°) | -50 ~ 50|
| 2 | ω_top | 탑스핀 (rad/s, + 탑 / - 백) | -200 ~ 200 |
| 3 | ω_side | 사이드스핀 (rad/s) | -200 ~ 200 |
| 4 | hit_x | 타격 좌우 위치 (m) | -3 ~ 3 |
| 5 | hit_y | 타격 앞뒤 위치 (m) | -4 ~ 0 |
| 6 | hit_z | 타격 높이 (m) | -0.76 ~ 1.5 |
| 7 | θ_h | 수평 타격 방향각 (°) | -65 ~ 65 |

**2. 21개 시간 특징**
- 시간 $t$에 대한 Fourier Positional Encoding 적용.

### 출력: 3D Output Trajectory
- 해당 시점($t$)의 3차원 좌표 (X, Y, Z) = 3개의 Float 값.
- 시간 간격 `dt = 0.002초`, 총 1.5초(750스텝) 길이의 궤적을 연속적으로 추론함.

---

## 💻 하드웨어 스펙 및 학습 최적화 가이드라인

이 프로젝트는 현재 로컬 딥러닝 워크스테이션 환경에 맞춰 최적화되어 있음.

### 시스템 스펙
- **GPU:** NVIDIA GeForce RTX 3070 (VRAM 8GB)
- **System RAM:** 약 128GB (가용 121GiB)

### 최적화 가이드 (필수 준수 사항)
1. **Batch Size 설정:** GPU 병렬 처리를 위해 `BATCH_SIZE = 1024`를 권장함.
2. **데이터 로딩 전략:** 시스템 RAM(128GB)의 여유를 활용하여 500만 개 이상의 데이터를 디스크 I/O 병목 없이 메모리에 한 번에 적재(`TensorDataset`)하여 사용함.
3. **DataLoader `num_workers = 0` 고정:** 데이터가 이미 시스템 RAM에 적재되어 있으므로, `num_workers`를 늘릴 경우 서브 프로세스 간 공유 메모리(`/dev/shm`) 복사 과정에서 **Bus Error (SIGBUS)** 및 메모리 오버플로우(프로세스당 35GB 코어 덤프)가 발생할 수 있음. 반드시 `0`으로 고정하여 메인 프로세스가 직접 GPU로 데이터를 전달하도록 설정함.
4. **백그라운드 무중단 실행 원칙:** 데이터 생성(`generate_dataset.py`) 및 모델 학습(`train_baseline.py`) 등 장시간 소요되는 작업은 SSH 세션 연결이 끊겨도 중단되지 않도록 `nohup [명령어] > [로그파일.log] 2>&1 &` 형태로 백그라운드 실행을 원칙으로 함.

---

## 🧠 시행착오 및 실험 기록 (Trial & Error Log)

스무딩 현상(U자 궤적)과 궤도 불안정 문제를 해결하기 위해 진행한 실험 기록.

1. **시도 1 :**
   - **날짜:** 2026-05-26
   - **내용:** 순수 Expanding MLP (`1024 -> 1500`) + 기본 MSE Loss
   - **결과:** 심각한 진동(Oscillation, Wavy artifacts) 현상 발생.
   ![시도 1 진짜 궤적: 심각한 파도타기 현상](results/attempt1_protoMLP.png)

2. **시도 2 (Kinematic Loss 도입):**
   - **날짜:** 2026-05-27
   - **내용:** Expanding MLP (`1024 -> 1500`) + MSE Loss + **Kinematic Loss (L1)**
   - **결과:** 진동 현상은 완화되어 궤적이 부드러워졌으나, **탁구대 바운드(충돌) 현상이 소실됨**. 궤적이 바닥에 닿지 않고 공중에 부유하는 형태로 생성됨.
   ![시도 2 진짜 궤적: 바운드 실종 및 과도한 스무딩](results/attempt2_kinematic.png)

3. **시도 3 (Chamfer Distance 도입):**
   - **날짜:** 2026-05-28
   - **내용:** Expanding MLP (`1024 -> 1500`) + MSE Loss + **Chamfer Distance**
   - **결과:** V자 형태를 강제하려다 **시계열(Time-series) 특성이 붕괴됨**. 점들이 궤적을 따라 순차적으로 배치되지 않고 바운드 지점 근처에 무질서하게 군집화(Point Cloud 형태)되어 형태와 애니메이션 속도 모두 정상적인 추론에 실패함.
   ![시도 3 궤적 파탄](results/attempt3_chamfer_fail.png)

4. **시도 4 (ResNet 구조 도입 + L1 Loss):**
   - **날짜:** 2026-05-28
   - **내용:** Expanding MLP 계층 사이에 **ResBlock(Skip Connection, Pre-Activation)** 삽입 및 순수 **L1 Loss(MAE)** 적용.
   - **목적:** 손실 함수 조정만으로는 본질적인 불확실성(평균화)을 피할 수 없다고 판단, 깊은 층에서도 고주파(High-frequency) 정보를 보존하기 위해 아키텍처를 개선함. L1 Loss를 통해 중앙값 추적을 강제함.
   - **결과:** **불안정한 궤적(Wobbling/Jittering) 발생**. ResNet 도입으로 고주파 정보를 표현할 능력은 확보하였고 L1 Loss로 과도한 스무딩은 방지했으나, 데이터 밀도 부족(차원의 저주)으로 인해 바운드 타이밍을 정확히 추론하지 못함. 결과적으로 바운드 지점 근처에서 심하게 요동치는 궤적이 생성됨.
   ![시도 4 지렁이 궤적](results/attempt4_resnet_fail.png)

5. **시도 5 (High-Density Data + ResNet + L1 Loss):**
   - **날짜:** 2026-05-29
   - **내용:** 파라미터 범위를 축소하여 데이터 밀도를 대폭 높인 데이터셋 적용 + Expanding MLP + ResBlock + L1 Loss.
   - **데이터 증강 및 파라미터 범위 변경 사항:**
     | 파라미터 | 이전 범위 | 변경된 범위 |
     |---|---|---|
     | speed | 1.0 ~ 100.0 | 1.0 ~ 80.0 |
     | theta_v | -85.0 ~ 85.0 | -50.0 ~ 50.0 |
     | omega_top | -800.0 ~ 800.0 | -200.0 ~ 200.0 |
     | omega_side | -800.0 ~ 800.0 | -200.0 ~ 200.0 |
     | hit_x | -3.0 ~ 3.0 | -3.0 ~ 3.0 |
     | hit_y | -4.0 ~ 0.0 | -4.0 ~ 0.0 |
     | hit_z | -0.7 ~ 2.0 | -0.76 ~ 1.5 |
     | theta_h | -80.0 ~ 80.0 | -65.0 ~ 65.0 |
   - **목적:** 데이터 밀도 증가 시 ResNet 기반 모델이 바운드 지점의 불확실성을 얼마나 해소할 수 있는지 검증.
   - **결과:** Test MAE **0.017 (약 1.7cm)**로 높은 정밀도를 기록했으나, 각 좌표를 독립적으로 예측하는 모델의 한계로 인해 시각화(`app.py`) 결과 여전히 궤적에 **미세한 진동(Jittering) 현상이 잔존**함.
   ![시도 5 지렁이 현상 잔존](results/attempt5_resnet_jitter.png)

6. **시도 6 (Padding Masking 단독 적용):**
   - **날짜:** 2026-05-29
   - **내용:** 바닥(Z<=-0.75) 충돌 이후 시뮬레이터가 정지한 좌표를 반복 복사(Padding)하는 현상 발견. 훈련 코드에 **바닥 충돌 이후 구간을 오차 계산에서 완전히 배제하는 Padding Masking** 로직 도입.
   - **목적:** 정지된 패딩 구간을 맞춰 오차율이 낮게 측정되는 착시 현상을 제거하고, 모델의 학습 용량을 실제 비행 궤적에 집중시키기 위함.
   - **결과:** 마스킹 적용 후 157 Epoch 만에 실제 오차 0.013(1.3cm) 달성. 시각화 결과, **바운드 이전의 비행 궤도가 진동 없이 안정적인 포물선으로 개선됨(궤도 불안정 해결)**. 단, 바운드 지점에서 V자가 아닌 U자로 뭉개지는(Bounce Smoothing) 현상은 여전히 남아 있음.
   ![시도 6 스무딩 현상 잔존 및 지렁이 현상 해결](results/attempt6_mask.png)

7. **시도 7 (Physics-Informed Cosine Similarity Loss):**
   - **날짜:** 2026-05-29
   - **내용:** 속도 벡터의 진행 방향(각도) 일치 여부를 평가하는 코사인 유사도(Cosine Similarity) 방향 Loss 투입. (가속도 및 속도의 크기(Magnitude) 제약은 배제됨)
   - **목적:** 바운드 순간의 급격한 방향 전환을 학습하도록 유도하여 날카로운 V자 바운스를 구현함.
   - **결과:** 궤적이 심하게 구불거리고 바운드 이후 중력을 무시한 수평 이동 현상 발생. 모델이 위치 정확도보다 각도 점수를 높이기 위해 속도를 극단적으로 줄이며 방향만 맞추는 **의도치 않은 편향(Local Minima)**에 빠짐.
   ![시도 7 코사인 꼼수 실패](results/attempt7_cosine.png)

8. **시도 8 (Physics-Informed Kinematic Derivative Loss):**
   - **날짜:** 2026-05-30
   - **내용:** 속도 벡터(크기+방향)의 1차 미분(L1)과 **가속도의 2차 미분(L1) Loss**를 위치 Loss와 1:1 비율로 추가. (`total_loss = pos_loss + 1.0 * vel_loss + 1.0 * acc_loss`)
   - **목적:** 가속도 Loss를 통해 공중에서의 중력과 바운드 순간의 수직 가속도 스파이크(+4500m/s²)를 강제 학습시킴 (Gradient-Weighted Loss).
   - **결과:** Train MAE 0.008(0.8cm), Test MAE 0.014 달성. 진동 현상이 완전히 사라지고 코사인 Loss의 편향 문제도 해결됨. 현재까지 가장 안정적인 물리 법칙 모방 결과를 보임.
   ![시도 8 완벽한 물리 법칙 모방](results/attempt8_kinematic.png)

9. **시도 9 (Strong Kinematic Derivative Loss):**
   - **날짜:** 2026-05-30
   - **내용:** 속도(3.0)와 가속도(5.0)의 페널티 가중치를 극단적으로 상향하여 훈련 진행.
   - **목적:** 바운드 시점의 극단적인 가속도 변화를 더 강하게 강제하여 더욱 날카로운 V자 궤적을 유도함.
   - **결과:** **과도한 페널티로 인한 역효과 (Over-smoothing) 발생.** 가속도 오차에 대한 페널티가 과도하여, 모델이 바운드 순간의 스파이크를 회피하고자 바운드 지점을 둥글게 처리하는(U자 궤적 및 부유 현상) 결과를 낳음. (시도 2의 스무딩 현상 재발)
   ![시도 9 가중치 과다로 인한 과도한 스무딩](results/attempt9_strong_kinematic.png)

10. **시도 10 (1D CNN Decoder 도입 - 구조적 한계 극복):**
    - **날짜:** 2026-05-30
    - **내용:** MLP 구조의 연속적 보간(Interpolation) 특성으로 인한 스무딩 현상이 근본 원인이라 판단. 속도/가속도 Loss를 제거하고 `pos_loss`만 유지한 채 모델 아키텍처를 `1D CNN Decoder`로 전면 교체. (Checkerboard Artifact 방지를 위해 Overlap 커널 K=9 적용)
    - **결과:** **심각한 고주파 노이즈(지그재그) 발생.** `kernel_size=9`와 `stride=5`의 조합이 불균등한 Overlap을 유발하여 체커보드 아티팩트가 발생함. 물리적 제약(속도/가속도 Loss) 부재로 노이즈 억제에 실패함.
    ![시도 10 체커보드 아티팩트 발생](results/attempt10_checkerboard.png)

11. **시도 11 (Upsample + Conv1d 및 Loss 절제 연구):**
    - **날짜:** 2026-05-31
    - **내용:** 체커보드 아티팩트 해결을 위해 `ConvTranspose1d`를 폐기하고 `nn.Upsample` + `nn.Conv1d` 조합으로 개편. CNN의 순수 형태 포착 능력 테스트를 위해 속도/가속도 Loss는 비활성화(0.0) 유지.
    - **결과:** **체커보드 노이즈는 해결되었으나 궤적 흔들림(Wobbling) 발생.** 고주파 노이즈는 제거되었으나 물리적 제약이 없어 전역적 평활도(Global Smoothness)를 유지하지 못하고 국소적 예측에 머무름.
    ![시도 11 궤적 흔들림(Wobbling)](results/attempt11_wobble.png)

12. **시도 12 (물리 Loss 복구 및 가속 도입 - 실패):**
    - **날짜:** 2026-05-31
    - **내용:** 궤적 흔들림 완화를 위해 속도/가속도 Loss(`vel_weight=1.0`, `acc_weight=0.1`) 재도입. 훈련 가속을 위해 `cuDNN benchmark`와 혼합 정밀도(`AMP`) 적용 및 바닥 충돌점 마스킹 누락 버그(`cumsum <= 1`) 수정.
    - **결과:** **심각한 궤적 붕괴(Staircase Derivative Explosion) 발생.** `Upsample(mode='nearest')`의 계단식 배열에 미분(Derivative) Loss를 강제 적용하면서, 모델이 속도 0과 무한대 사이에서 발산하여 학습에 실패함.
    ![시도 12 궤적 붕괴(Staircase Explosion)](results/attempt12_explosion.png)

13. **시도 13 (Nearest + Kernel Size 7 최적화 및 CNN의 한계 확인):**
    - **날짜:** 2026-05-31
    - **내용:** 계단 폭발 방지를 위해 `Upsample(nearest)` + `Conv1d(kernel_size=7)` 조합 적용. (필터 크기 확장을 통한 평활화 유도)
    - **결과:** **CNN 아키텍처의 전역 위치 파악 한계(Global Position Drift) 확인.** V자 바운드의 국소적 형태는 우수하게 형성되었으나, 전체 시간 흐름에 따른 절대적 위치를 파악하지 못해 궤적이 정답에서 약 11cm(MAE 0.11) 이탈하는 표류 현상 발생.
    ![시도 13 궤적 표류(Drift)](results/attempt13_wiggle.png)

14. **시도 14 (Time-Conditioned MLP + Fourier Positional Encoding):**
    - **날짜:** 2026-06-01
    - **내용:** CNN 구조 폐기 및 **암묵적 표현(Implicit Representation)** 아키텍처 도입. 입력에 시간 $t$를 분리하고 **푸리에 위치 인코딩(Fourier Positional Encoding, `num_freqs=10`)**을 적용하여 고주파(바운드) 자극 주입.
    - **결과:** **스무딩 및 위치 상실 대폭 완화, 그러나 바닥 잔상(Gibbs/Ringing) 발생.** 표류(Drift) 현상이 해결되고 오차가 0.015(1.5cm)로 안정화됨. 단, `ReLU` 활성화 함수의 한계로 0.002 목표에는 미달함. 정지된 패딩 구간을 완벽한 직선으로 추론하지 못해 메인 궤적 하단에 잔상이 렌더링되는 현상 관찰됨.
    ![시도 14 바닥 잔상 및 0.015 정체기](results/attempt14_fourier.png)

15. **시도 15 (마스킹 버그 및 물리 엔진 모순 발견):**
    - **날짜:** 2026-06-02
    - **내용:** Time-Conditioned MLP 구조 유지 및 모델 규모 확장 (폭 256 -> 512, ResBlock 2층 -> 3층).
    - **결과:** V자 형태 구현이 가장 우수하게 나타났으나, 오차율이 0.02 수준에 머무르고 궤적이 다소 불안정하여 공이 튀는 현상이 간헐적으로 발생함.
    ![시도 15 V자 해결 및 궤도불안정](results/attempt15_deep_res_fourier.png)

16. **시도 16 (패딩 마스킹 개선, SiLU, 최적화 모델 스케일링):**
    - **날짜:** 2026-06-03
    - **내용:** 마스킹 버그(`cumsum <= 1`)를 `cummax` 기반 로직으로 수정. 활성화 함수를 `ReLU`에서 `SiLU`로 교체하여 부드러운 보간 극대화. 메모리 최적화를 위해 폭을 256으로 축소하고 배치 사이즈를 1024로 조정.
    - **결과:** **목표 달성 (Test MAE 0.0067 / Train MAE 0.0032)!** 바운스 이후의 노이즈 데이터를 차단(Masking)하여 약 3~6mm의 초정밀 오차율을 기록함. 탁구대 반사 및 바닥 충돌 시 시각적으로 유사한 궤적을 확인함. 딥러닝의 연속 함수 특성상 바운드 순간의 V자가 미세하게 둥글어지는 현상은 잔존하나 시뮬레이션 용도로 적합함.
    ![시도 16 안정권 돌입](results/attempt16_silu.png)

17. **시도 17 (가우시안 정규 분포 기반 데이터셋 밀도 최적화):**
    - **날짜:** 2026-06-04
    - **내용:** 데이터 품질 개선을 위해 파라미터 무작위 샘플링 방식을 균등 분포(`np.random.uniform`)에서 **정규 분포(`np.random.normal`)**로 전면 교체 (speed는 로그정규분포 채용).
    - **목적:** 정상적인 랠리 궤적의 비율을 높이기 위해 각 8D 파라미터의 평균을 일반적인 랠리 조건에 맞추고 표준편차를 조절하여 신규 데이터셋(`dataset_gaussian_mixed.npz`) 500만 개 생성.
    - **결과:** 궤적의 세부 정밀도가 크게 향상됨.

18. **시도 18 (100% 기각 샘플링 및 750스텝 연장):**
    - **날짜:** 2026-06-07
    - **내용:** 궤적 길이를 500스텝(1.0초)에서 750스텝(1.5초)으로 연장. 이상치 뭉침 현상 방지를 위해 범위를 벗어나면 재샘플링하는 기각 샘플링(Truncated Normal) 도입. 이상치에 강건한 `SmoothL1Loss`(Huber Loss) 적용.
    - **목적:** 장기 궤적 예측 능력 확보 및 경계선 데이터 뭉침 원천 차단을 통해 극단적 이상치에서의 정밀도 향상.
    - **결과:** 338 에포크 조기 종료. Test MAE **0.00053 (0.53mm)**로 수치상 역대 최고 정밀도 달성. 그러나 시각화 결과 바운드 지점이 둥근 U자로 심하게 뭉개지는(Over-smoothing) 현상이 발생하여 시도 17 대비 시각적 품질은 하락함.

19. **시도 19 (고주파 표현 확장 & 순수 L1 Loss 롤백):**
    - **날짜:** 2026-06-11
    - **내용:** 스무딩의 원인인 `SmoothL1Loss`를 제거하고 `순수 L1Loss`로 복귀. 푸리에 위치 인코딩의 주파수 대역(`num_freqs=10`)을 통해 고주파 표현 범위를 확장하여 훈련 진행.
    - **목적:** 순수 L1 Loss와 극한의 고주파수 해상도의 시너지를 통해 모델 스스로 날카로운 1프레임 V자 바운스를 복원하도록 유도.
    - **결과:** 탁구대 위 바운스는 의도대로 날카로운 V자로 복원됨(스무딩 해결). 그러나 푸리에 고주파수 대역폭 한계 확장으로 인해, 바운스가 없는 허공이나 바닥 궤적에서도 선이 요동치거나 가짜 바운스(Phantom Bounces / Rippling Artifacts)가 발생하는 고주파수 과적합 현상 관찰됨.
    ![시도 19 안정권 돌입](results/attempt19_fourier_freq10_pure_l1.png)

---

## 📁 디렉토리 구조

```
/root/tabletennis_trajec_model/
├── app.py                    # 물리엔진 vs AI 비교 3D 대시보드 (Gradio)
├── generate_dataset.py       # 물리엔진으로 500만개 데이터셋 생성 스크립트
├── train_baseline.py         # AI(Time-Conditioned MLP) 딥러닝 모델 훈련 스크립트
├── RESEARCH_NOTES.md         # 이 연구 노트
└── database/
    ├── dataset_random.npz                 # 기존 균등 분포 정답지
    ├── dataset_gaussian_mixed.npz         # [New] 정규 분포 최적화 정답지 (시도 17 이후)
    ├── ... (시도 1~15 이전 모델 파일들 생략)
    ├── mlp_norm_standard_random_attempt16_scaled_mlp.npy
    ├── mlp_standard_random_attempt16_scaled_mlp.pth
    ├── mlp_norm_standard_gaussian_mixed_attempt17_gaussian_mixed.npy
    ├── mlp_standard_gaussian_mixed_attempt17_gaussian_mixed.pth
    ├── mlp_norm_standard_gaussian_mixed_attempt18_truncnorm.npy
    ├── mlp_standard_gaussian_mixed_attempt18_truncnorm.pth
    ├── mlp_norm_standard_gaussian_mixed_attempt19_fourier_freq10_pure_l1.npy
    └── mlp_standard_gaussian_mixed_attempt19_fourier_freq10_pure_l1.pth
```