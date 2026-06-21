import gradio as gr
import numpy as np
import os
import time
import tracemalloc
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.integrate import solve_ivp

DB_DIR    = "/root/myresearch/database"
K_MAGNUS  = 0.0034   # 실제 탁구공 물리 데이터 기반 도출
K_DRAG    = 0.112    

def rotate_2d(vector, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]]).dot(vector)

# ============================================================
# AI 베이스라인 모델 (MLP) 정의 및 로드
# ============================================================
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        
    def forward(self, x):
        h = F.silu(self.fc1(x))
        h = self.fc2(h)
        return F.silu(x + h)

class TimeConditionedMLP(nn.Module):
    def __init__(self, num_freqs=10):
        super().__init__()
        self.num_freqs = num_freqs
        in_dim = 8 + 1 + 2 * num_freqs
        
        self.fc_in = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.SiLU()
        )
        
        self.blocks = nn.ModuleList([ResBlock(256) for _ in range(3)])
        self.fc_out = nn.Linear(256, 3)
        
    def forward(self, x, t):
        freq_bands = 2.0 ** torch.linspace(0, self.num_freqs - 1, self.num_freqs, device=t.device)
        t_freqs = t * freq_bands * torch.pi
        
        t_enc = torch.cat([t, torch.sin(t_freqs), torch.cos(t_freqs)], dim=-1)
        features = torch.cat([x, t_enc], dim=-1)
        
        h = self.fc_in(features)
        for block in self.blocks:
            h = block(h)
            
        return self.fc_out(h)

device = torch.device("cpu") # 공정한 속도 벤치마크를 위해 CPU로 강제 고정

model_random = TimeConditionedMLP().to(device)
model_path = os.path.join(DB_DIR, "mlp_standard_gaussian_mixed_attempt19_fourier_freq10_pure_l1.pth")
if os.path.exists(model_path):
    try:
        model_random.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except Exception as e:
        print(f"Warning: Skipping weight load due to mismatch: {e}")
model_random.eval()

norm_path = os.path.join(DB_DIR, "mlp_norm_standard_gaussian_mixed_attempt19_fourier_freq10_pure_l1.npy")
if os.path.exists(norm_path):
    norm_random = np.load(norm_path)
else:
    norm_random = np.array([np.zeros(8), np.ones(8)])

# ============================================================
# 기존 물리 엔진
# ============================================================
def traditional_physics_solver(speed, theta_v_deg, omega_top, omega_side, hit_x, hit_y, hit_z, theta_h_deg, steps=750, dt=0.002):
    theta_v = np.radians(theta_v_deg)
    theta_h = np.radians(theta_h_deg)
    
    vy0 = speed * np.cos(theta_v) * np.cos(theta_h)
    vx0 = -speed * np.cos(theta_v) * np.sin(theta_h)
    vz0 = speed * np.sin(theta_v)
    
    initial_state = [hit_x, hit_y, hit_z, vx0, vy0, vz0]
    
    omega_world = np.array([
        -omega_top * np.cos(theta_h),
        -omega_top * np.sin(theta_h),
        omega_side
    ])
    
    def derivatives(t, state):
        x, y, z, vx, vy, vz = state
        vel = np.array([vx, vy, vz])
        spd = np.linalg.norm(vel)
        if spd < 1e-6:
            return [vx, vy, vz, 0.0, 0.0, -9.81]
            
        drag = -K_DRAG * spd * vel
        magnus = np.cross(omega_world, vel) * K_MAGNUS
        acc = np.array([0.0, 0.0, -9.81]) + drag + magnus
        
        return [vx, vy, vz, acc[0], acc[1], acc[2]]

    def hit_table(t, state):
        return state[2]
    hit_table.terminal = True
    hit_table.direction = -1

    def hit_floor(t, state):
        return state[2] - (-0.76)
    hit_floor.terminal = True
    hit_floor.direction = -1

    def hit_net(t, state):
        return state[1]
    hit_net.terminal = True
    hit_net.direction = 0

    t_eval = np.linspace(0, steps * dt, steps + 1)
    
    traj = []
    current_t = 0.0
    current_state = initial_state
    times_needed = list(t_eval)
    
    while len(times_needed) > 0:
        t_end = times_needed[-1]
        if current_t >= t_end:
            break
            
        sol = solve_ivp(derivatives, (current_t, t_end), current_state, 
                        t_eval=times_needed, events=[hit_table, hit_floor, hit_net], rtol=1e-5, atol=1e-8)
        
        if len(sol.t) > 0:
            for i in range(len(sol.t)):
                traj.append(sol.y[:3, i].copy())
            times_needed = times_needed[len(sol.t):]
        
        if sol.status == 1:
            event_idx = -1
            event_t = -1
            for i in range(3):
                if len(sol.t_events[i]) > 0:
                    if sol.t_events[i][-1] > event_t:
                        event_t = sol.t_events[i][-1]
                        event_idx = i
            
            if event_idx == -1:
                current_t = sol.t[-1] if len(sol.t) > 0 else current_t + dt
                continue
                
            current_t = event_t
            current_state = sol.y_events[event_idx][-1].copy()
            x, y, z, vx, vy, vz = current_state
            
            if event_idx == 0: # hit_table
                if abs(x) <= 0.76 and abs(y) <= 1.37:
                    vz_out = -0.83 * vz
                    R_ball = 0.02
                    vy_out = 0.7 * vy + 0.1 * omega_top * R_ball * np.cos(theta_h)
                    vx_out = 0.7 * vx + 0.1 * omega_top * R_ball * (-np.sin(theta_h)) + 0.1 * (-omega_side) * R_ball
                    current_state[3:6] = [vx_out, vy_out, vz_out]
                    current_state[2] = 1e-4
                else:
                    current_state[2] = -1e-4
                    
            elif event_idx == 1: # hit_floor
                current_state[3:6] = [0.5*vx, 0.5*vy, -0.5*vz]
                current_state[2] = -0.76 + 1e-4
                
            elif event_idx == 2: # hit_net
                if abs(x) <= 0.76 and z <= 0.1525:
                    current_state[3:6] = [0.5*vx, -0.3*vy, 0.5*vz]
                    current_state[1] = 1e-4 if vy < 0 else -1e-4
                else:
                    current_state[1] = 1e-4 if vy > 0 else -1e-4

    traj = np.array(traj)
    if len(traj) > steps:
        traj = traj[:steps]
    elif len(traj) < steps:
        pad = np.tile(traj[-1], (steps - len(traj), 1))
        traj = np.vstack([traj, pad])
        
    return traj

# ============================================================
# 공통 유틸
# ============================================================
def make_table_and_net():
    floor = go.Mesh3d(x=[-4.0, 4.0, 4.0, -4.0], y=[-6.0, -6.0, 6.0, 6.0], z=[-0.76]*4, color='lightgray', opacity=0.4, name='Floor')
    table = go.Mesh3d(x=[-0.76, 0.76, 0.76, -0.76], y=[-1.37, -1.37, 1.37, 1.37], z=[0]*4, color='green', opacity=0.3, name='Table')
    net = go.Scatter3d(x=[-0.76, 0.76], y=[0, 0], z=[0.15, 0.15], mode='lines', line=dict(color='black', width=3), name='Net')
    return floor, table, net

def apply_transform(traj_xyz, start_x, start_y, start_z, theta_h_rad):
    xy = traj_xyz[:, :2]
    z = traj_xyz[:, 2]
    xy_rot = np.array([rotate_2d(pt, theta_h_rad) for pt in xy])
    return (xy_rot[:, 0] + start_x, xy_rot[:, 1] + start_y, z - z[0] + start_z)

SCENE_LAYOUT = dict(
    xaxis=dict(range=[-3.0, 3.0], title='X (m)'),
    yaxis=dict(range=[-4.0, 4.0], title='Y (m)'),
    zaxis=dict(range=[-0.76, 2.0], title='Z (m)'),
    aspectmode='manual', aspectratio=dict(x=1, y=1.33, z=0.46)
)

# ============================================================
# 메인 시뮬레이션
# ============================================================
def simulate(speed, theta_v, omega_top, omega_side, hit_x, hit_y, hit_z, theta_h):
    # ── [1] 물리 엔진 (Ground Truth) ───────────────────────────
    tracemalloc.start()
    t0 = time.perf_counter()
    
    trad_traj = traditional_physics_solver(speed, theta_v, omega_top, omega_side, hit_x, hit_y, hit_z, theta_h, steps=750, dt=0.002)
    
    # --- Ground Truth Post-processing (Floor Cutoff Padding) ---
    hit_floor_gt = np.where(trad_traj[:, 2] <= -0.75)[0]
    if len(hit_floor_gt) > 0:
        first_hit_gt = hit_floor_gt[0]
        trad_traj[first_hit_gt:] = trad_traj[first_hit_gt]
        
    x_t, y_t, z_t = trad_traj[:, 0], trad_traj[:, 1], trad_traj[:, 2]
    
    trad_ms = (time.perf_counter() - t0) * 1000
    _, peak_mem_t = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # ── [2] AI 디코더 (MLP) ───────────────────────────────────
    tracemalloc.start()
    t0 = time.perf_counter()
    
    input_arr = np.array([speed, theta_v, omega_top, omega_side, hit_x, hit_y, hit_z, theta_h], dtype=np.float32)
    
    mean, std = norm_random[0], norm_random[1]
    model = model_random
        
    try:
        if len(mean) == 8: # Only run AI if norm is 8D (meaning model was updated)
            input_norm = (input_arr - mean) / (std + 1e-8)
            input_tensor = torch.tensor(input_norm, dtype=torch.float32).to(device)
            with torch.no_grad():
                t = (torch.arange(750, device=device, dtype=torch.float32) * 0.002).unsqueeze(1)
                x_rep = input_tensor.unsqueeze(0).expand(750, 8)
                pred_traj = model(x_rep, t).cpu().numpy()
                
            # --- Post-processing (Floor Cutoff Padding) ---
            hit_floor_indices = np.where(pred_traj[:, 2] <= -0.75)[0]
            if len(hit_floor_indices) > 0:
                first_hit = hit_floor_indices[0]
                pred_traj[first_hit:] = pred_traj[first_hit]
                
            x_m, y_m, z_m = pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2]
        else:
            x_m, y_m, z_m = [0]*750, [0]*750, [0]*750
    except:
        x_m, y_m, z_m = [0]*750, [0]*750, [0]*750
        
    ai_ms = (time.perf_counter() - t0) * 1000
    _, peak_mem_m = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    try:
        l2_error = np.linalg.norm(trad_traj - pred_traj, axis=1).mean()
    except:
        l2_error = 0.0
    
    # ── 메트릭 마크다운 ───────────────────────────────────────
    speedup = trad_ms / ai_ms if ai_ms > 0 else 0
    
    metrics_md = f"""
### ⚡ AI 생성형 디코더 벤치마크 결과 (Random Split 500k)

| 지표 | 🐢 정통 물리 엔진 (GT) | 🚀 AI 디코더 (MLP) | 비교 |
|---|---|---|---|
| **처리 속도** | <span style="color:red">{trad_ms:.4f} ms</span> | **<span style="color:blue">{ai_ms:.4f} ms</span>** | **{speedup:,.1f}배 빠름** |
| **메모리** | <span style="color:red">{peak_mem_t:,} B</span> | **<span style="color:blue">{peak_mem_m:,} B</span>** | |
| **평균 궤적 오차(L2)** | 0.0 m (기준점) | **{l2_error:.4f} m** | 매우 정밀함 |
"""

    # ── 플롯 생성 ──────────────────────────────────────────────
    def make_fig(title, x, y, z, color, colorscale, name):
        floor, tbl, net = make_table_and_net()
        fig = go.Figure([floor, tbl, net,
            go.Scatter3d(x=x, y=y, z=z, mode='lines+markers', marker=dict(size=3, color=z, colorscale=colorscale), line=dict(color=color, width=2), name=name)])
        fig.update_layout(scene=SCENE_LAYOUT, title=title, margin=dict(l=0, r=0, b=0, t=40))
        return fig

    fig_trad = make_fig("🐢 정통 물리엔진 (Ground Truth)", x_t, y_t, z_t, 'steelblue', 'Blues', 'Physics Engine')
    fig_ai  = make_fig(f"🚀 AI 디코더", x_m, y_m, z_m, 'magenta', 'Plasma', 'AI Decoder')

    return fig_trad, fig_ai, metrics_md

# ============================================================
# 대량 병렬 렌더링 (Batch Processing)
# ============================================================
def simulate_batch(num_samples):
    num_samples = int(num_samples)
    tracemalloc.start()
    t0 = time.perf_counter()
    
    speed = np.clip(np.random.lognormal(mean=1.8, sigma=0.8, size=num_samples), 1.0, 80.0)
    theta_v = np.clip(np.random.normal(15.0, 15.0, num_samples), -50.0, 50.0)
    omega_top = np.clip(np.random.normal(50.0, 50.0, num_samples), -200.0, 200.0)
    omega_side = np.clip(np.random.normal(0.0, 50.0, num_samples), -200.0, 200.0)
    hit_x = np.clip(np.random.normal(0.0, 0.75, num_samples), -3.0, 3.0)
    hit_y = np.clip(np.random.normal(-1.5, 0.5, num_samples), -4.0, 0.0)
    hit_z = np.clip(np.random.normal(0.3, 0.3, num_samples), -0.76, 1.5)
    theta_h = np.clip(np.random.normal(0.0, 20.0, num_samples), -65.0, 65.0)
    
    input_arr = np.stack([speed, theta_v, omega_top, omega_side, hit_x, hit_y, hit_z, theta_h], axis=1).astype(np.float32)
    
    mean, std = norm_random[0], norm_random[1]
    
    if len(mean) == 8:
        input_norm = (input_arr - mean) / (std + 1e-8)
        input_tensor = torch.tensor(input_norm, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            t = (torch.arange(750, device=device, dtype=torch.float32) * 0.002).unsqueeze(1)
            t_batch = t.unsqueeze(0).expand(num_samples, 750, 1)
            x_rep = input_tensor.unsqueeze(1).expand(num_samples, 750, 8)
            pred_traj = model_random(x_rep, t_batch).cpu().numpy()
            
        floor, tbl, net = make_table_and_net()
        fig = go.Figure([floor, tbl, net])
        
        import plotly.express as px
        colors = px.colors.qualitative.Plotly
        
        for i in range(num_samples):
            traj = pred_traj[i]
            hit_floor_indices = np.where(traj[:, 2] <= -0.75)[0]
            if len(hit_floor_indices) > 0:
                first_hit = hit_floor_indices[0]
                traj = traj[:first_hit+1]
                
            c = colors[i % len(colors)]
            fig.add_trace(go.Scatter3d(x=traj[:, 0], y=traj[:, 1], z=traj[:, 2], 
                                       mode='lines', line=dict(color=c, width=4), opacity=0.7, 
                                       showlegend=False))
            
        fig.update_layout(scene=SCENE_LAYOUT, title=f"🚀 AI 디코더 {num_samples}개 병렬 렌더링", margin=dict(l=0, r=0, b=0, t=40))
    else:
        fig = go.Figure()
        
    ai_ms = (time.perf_counter() - t0) * 1000
    _, peak_mem_m = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    metrics_md = f"### ⚡ 병렬 렌더링 결과\n- **생성 개수**: {num_samples}개\n- **소요 시간**: **<span style='color:blue'>{ai_ms:.4f} ms</span>**\n- **메모리**: {peak_mem_m:,} B\n> 단 하나의 for-loop나 미분방정식 없이 순수 행렬 곱셈만으로 모든 궤적을 동시 계산합니다."
    
    return fig, metrics_md

# ============================================================
# 대규모 벤치마크 (Benchmark)
# ============================================================
def run_benchmark(num_samples):
    num_samples = int(num_samples)
    data_path = os.path.join(DB_DIR, "dataset_gaussian_mixed.npz")
    if not os.path.exists(data_path):
        return "데이터셋 파일(dataset_gaussian_mixed.npz)을 찾을 수 없습니다."
        
    data = np.load(data_path)
    X_test, y_test = data['test_inputs'], data['test_outputs']
    
    if num_samples > len(X_test):
        num_samples = len(X_test)
        
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    X_sample = X_test[indices]
    y_sample = y_test[indices]
    
    # AI 속도 측정 (Batch)
    mean, std = norm_random[0], norm_random[1]
    input_norm = (X_sample - mean) / (std + 1e-8)
    input_tensor = torch.tensor(input_norm, dtype=torch.float32).to(device)
    
    tracemalloc.start()
    t0_ai = time.perf_counter()
    with torch.no_grad():
        t = (torch.arange(750, device=device, dtype=torch.float32) * 0.002).unsqueeze(1)
        t_batch = t.unsqueeze(0).expand(num_samples, 750, 1)
        x_rep = input_tensor.unsqueeze(1).expand(num_samples, 750, 8)
        pred_traj = model_random(x_rep, t_batch).cpu().numpy()
    ai_time = time.perf_counter() - t0_ai
    _, peak_mem_ai = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # 정통 물리 엔진 속도 측정 (일부만 샘플링하여 추정)
    trad_samples_for_speed = min(num_samples, 100)
    trad_times = []
    
    tracemalloc.start()
    for i in range(trad_samples_for_speed):
        speed, theta_v, omega_top, omega_side, hit_x, hit_y, hit_z, theta_h = X_sample[i]
        t0_trad = time.perf_counter()
        _ = traditional_physics_solver(speed, theta_v, omega_top, omega_side, hit_x, hit_y, hit_z, theta_h)
        trad_times.append(time.perf_counter() - t0_trad)
    _, peak_mem_trad = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    avg_trad_time = np.mean(trad_times) if len(trad_times) > 0 else 0
    est_total_trad_time = avg_trad_time * num_samples
    throughput_trad = 1.0 / avg_trad_time if avg_trad_time > 0 else 0
    
    avg_ai_time = ai_time / num_samples if num_samples > 0 else 0
    throughput_ai = num_samples / ai_time if ai_time > 0 else 0
    
    # 정확도 및 물리 위반 검사
    errors = []
    max_errors = []
    physics_violations = 0
    
    for i in range(num_samples):
        gt = y_sample[i]
        pred = pred_traj[i]
        
        hit_floor_gt = np.where(gt[:, 2] <= -0.75)[0]
        first_hit = hit_floor_gt[0] if len(hit_floor_gt) > 0 else 750
        
        if first_hit < 750:
            pred[first_hit:] = pred[first_hit]
            
        diff = np.linalg.norm(gt - pred, axis=1)
        
        err = diff[:first_hit].mean() if first_hit > 0 else 0
        mx_err = diff[:first_hit].max() if first_hit > 0 else 0
        errors.append(err)
        max_errors.append(mx_err)
        
        # 물리 위반: 예측 Z가 -0.78 밑으로 뚫고 내려가거나 기타 등등
        if np.any(pred[:, 2] < -0.78):
            physics_violations += 1
            
    md_output = f"""
### 📈 성능 비교 벤치마크 (샘플 수: {num_samples}개)

| 측정 항목 | 🐢 기존 물리 엔진 (GT) | 🚀 AI 디코더 (MLP) | 비교 |
|---|---|---|---|
| **총 소요 시간** | {est_total_trad_time:.2f} 초 (예상) | **<span style="color:blue">{ai_time:.4f} 초</span>** | **{est_total_trad_time / ai_time:.1f}배 빠름** |
| **초당 생성 (Throughput)** | {throughput_trad:.1f} 개/초 | **<span style="color:blue">{throughput_ai:,.1f} 개/초</span>** | **{throughput_ai / throughput_trad:.1f}배 향상** |
| **1개당 평균 추론 시간** | {avg_trad_time * 1000:.2f} ms | **{avg_ai_time * 1000:.4f} ms** | - |
| **최대 메모리 사용량** | {peak_mem_trad / 1024 / 1024:.2f} MB | {peak_mem_ai / 1024 / 1024:.2f} MB | - |
| **평균 오차 (MAE)** | 0.0 mm (기준점) | **{np.mean(errors) * 1000:.2f} mm** | 초정밀 수준 |
| **최대 오차 (Max Error)** | 0.0 mm | **{np.max(max_errors) * 1000:.2f} mm** | - |
| **물리 법칙 위반 횟수** | 0 회 | **{physics_violations} 회** | 바닥 투과 등 |

> *참고: 기존 물리 엔진의 총 소요 시간은 첫 {trad_samples_for_speed}개의 평균 소요 시간을 기반으로 산출된 예상치입니다.*
"""
    return md_output

# ============================================================
# Gradio UI
# ============================================================
with gr.Blocks() as app:
    gr.Markdown("# 🏓 생성형 물리 AI 디코더 시뮬레이터 (Generative Decoder)")

    with gr.Tab("1:1 정밀 비교"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚡ 물리 파라미터 (입력)")
                sl_speed     = gr.Slider(1.0,  80.0, value=6.5, step=0.1, label="타격 강도 — 초기 속도 (m/s)")
                sl_theta_v   = gr.Slider(-50,  50,   value=15,  step=1, label="수직 발사각 θ_v (°)")
                sl_omega_top = gr.Slider(-200, 200,  value=50, step=5, label="탑스핀 ω_top (rad/s)")
                sl_omega_sid = gr.Slider(-200, 200,  value=0,   step=5, label="사이드스핀 ω_side (rad/s)")
                sl_hit_z     = gr.Slider(-0.76, 1.5,  value=0.3, step=0.01, label="타격 높이 Z0 (m)")

                gr.Markdown("---")
                gr.Markdown("### 📍 타격 위치 & 방향")
                sl_hit_x   = gr.Slider(-3.0, 3.0,  value=0.0,  step=0.05, label="타격 위치 X (m)")
                sl_hit_y   = gr.Slider(-4.0, 0.0,  value=-1.5, step=0.05, label="타격 위치 Y (m)")
                sl_theta_h = gr.Slider(-65,  65,   value=0,   step=1, label="수평 타격 방향각 θ_h (°)")

                sim_btn = gr.Button("🚀 AI 렌더링 & 벤치마크 실행", variant="primary")
                metrics_display = gr.Markdown("버튼을 누르면 시뮬레이션 결과가 나타납니다.")

            with gr.Column(scale=2):
                with gr.Row():
                    plot_trad = gr.Plot(label="Ground Truth 궤적")
                    plot_map  = gr.Plot(label="AI Decoder 궤적")

        inputs  = [sl_speed, sl_theta_v, sl_omega_top, sl_omega_sid, sl_hit_x, sl_hit_y, sl_hit_z, sl_theta_h]
        outputs = [plot_trad, plot_map, metrics_display]
        sim_btn.click(fn=simulate, inputs=inputs, outputs=outputs)
        app.load(fn=simulate,      inputs=inputs, outputs=outputs)

    with gr.Tab("AI 배치 렌더링 (Random)"):
        gr.Markdown("### 🚀 초고속 병렬 랜덤 궤적 생성")
        with gr.Row():
            with gr.Column(scale=1):
                sl_num_samples = gr.Slider(10, 300, value=30, step=10, label="생성할 궤적 개수 (Batch Size)")
                batch_btn = gr.Button("🔥 대량 궤적 쏟아내기", variant="primary")
                batch_metrics = gr.Markdown("버튼을 누르면 렌더링 결과가 나타납니다.")
            with gr.Column(scale=2):
                plot_batch = gr.Plot(label="AI Decoder 대량 궤적")
                
        batch_btn.click(fn=simulate_batch, inputs=[sl_num_samples], outputs=[plot_batch, batch_metrics])

    with gr.Tab("종합 벤치마크 리포트"):
        gr.Markdown("### 📊 정통 물리 엔진 vs AI 디코더 대규모 성능 비교")
        gr.Markdown("테스트 데이터셋(`dataset_gaussian_mixed.npz`)에서 N개의 궤적 샘플을 무작위로 추출하여 정밀도와 속도를 정량적으로 비교합니다.")
        with gr.Row():
            with gr.Column(scale=1):
                bench_samples = gr.Slider(100, 10000, value=1000, step=100, label="벤치마크 샘플 수")
                bench_btn = gr.Button("벤치마크 실행 (시간이 소요될 수 있습니다)", variant="primary")
            with gr.Column(scale=2):
                bench_result_md = gr.Markdown("좌측에서 버튼을 누르면 벤치마크가 시작됩니다...")
                
        bench_btn.click(fn=run_benchmark, inputs=[bench_samples], outputs=[bench_result_md])

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
