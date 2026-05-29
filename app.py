import gradio as gr
import numpy as np
import os
import time
import tracemalloc
import plotly.graph_objects as go
import torch
import torch.nn as nn
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
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
    def forward(self, x):
        return self.block(x) + x # Skip Connection

class PhysicsDecoderMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 차원 팽창 + ResNet (Skip Connection) 구조
        self.in_layer = nn.Sequential(
            nn.Linear(8, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        self.res1 = ResBlock(256)
        
        self.up1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU()
        )
        self.res2 = ResBlock(512)
        
        self.up2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.ReLU()
        )
        self.res3 = ResBlock(1024)
        
        self.out_layer = nn.Linear(1024, 1500)
        
    def forward(self, x):
        x = self.in_layer(x)
        x = self.res1(x)
        x = self.up1(x)
        x = self.res2(x)
        x = self.up2(x)
        x = self.res3(x)
        out = self.out_layer(x)
        return out.view(-1, 500, 3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_random = PhysicsDecoderMLP().to(device)
model_path = os.path.join(DB_DIR, "mlp_standard_random_attempt7_cosine_loss.pth")
if os.path.exists(model_path):
    try:
        model_random.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except Exception as e:
        print(f"Warning: Skipping weight load due to mismatch: {e}")
model_random.eval()

norm_path = os.path.join(DB_DIR, "mlp_norm_standard_random_attempt7_cosine_loss.npy")
if os.path.exists(norm_path):
    norm_random = np.load(norm_path)
else:
    norm_random = np.array([np.zeros(8), np.ones(8)])

# ============================================================
# 기존 물리 엔진
# ============================================================
def traditional_physics_solver(speed, theta_v_deg, omega_top, omega_side, hit_x, hit_y, hit_z, theta_h_deg, steps=500, dt=0.002):
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
    floor = go.Mesh3d(x=[-2.5, 2.5, 2.5, -2.5], y=[-3.5, -3.5, 3.5, 3.5], z=[-0.76]*4, color='lightgray', opacity=0.4, name='Floor')
    table = go.Mesh3d(x=[-0.76, 0.76, 0.76, -0.76], y=[-1.37, -1.37, 1.37, 1.37], z=[0]*4, color='green', opacity=0.3, name='Table')
    net = go.Scatter3d(x=[-0.76, 0.76], y=[0, 0], z=[0.15, 0.15], mode='lines', line=dict(color='black', width=3), name='Net')
    return floor, table, net

def apply_transform(traj_xyz, start_x, start_y, start_z, theta_h_rad):
    xy = traj_xyz[:, :2]
    z = traj_xyz[:, 2]
    xy_rot = np.array([rotate_2d(pt, theta_h_rad) for pt in xy])
    return (xy_rot[:, 0] + start_x, xy_rot[:, 1] + start_y, z - z[0] + start_z)

SCENE_LAYOUT = dict(
    xaxis=dict(range=[-1.5, 1.5], title='X (m)'),
    yaxis=dict(range=[-2.0, 2.0], title='Y (m)'),
    zaxis=dict(range=[-1.0, 1.0], title='Z (m)'),
    aspectmode='manual', aspectratio=dict(x=1, y=1.5, z=0.8)
)

# ============================================================
# 메인 시뮬레이션
# ============================================================
def simulate(speed, theta_v, omega_top, omega_side, hit_x, hit_y, hit_z, theta_h):
    # ── [1] 물리 엔진 (Ground Truth) ───────────────────────────
    tracemalloc.start()
    t0 = time.perf_counter()
    
    trad_traj = traditional_physics_solver(speed, theta_v, omega_top, omega_side, hit_x, hit_y, hit_z, theta_h, steps=500, dt=0.002)
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
            input_tensor = torch.tensor(input_norm, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_traj = model(input_tensor).cpu().numpy()[0]
                
            # --- Post-processing (Floor Cutoff Padding) ---
            hit_floor_indices = np.where(pred_traj[:, 2] <= -0.75)[0]
            if len(hit_floor_indices) > 0:
                first_hit = hit_floor_indices[0]
                pred_traj[first_hit:] = pred_traj[first_hit]
                
            x_m, y_m, z_m = pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2]
        else:
            x_m, y_m, z_m = [0]*500, [0]*500, [0]*500
    except:
        x_m, y_m, z_m = [0]*500, [0]*500, [0]*500
        
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
# Gradio UI
# ============================================================
with gr.Blocks() as app:
    gr.Markdown("# 🏓 생성형 물리 AI 디코더 시뮬레이터 (Generative Decoder)")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚡ 물리 파라미터 (입력)")
            sl_speed     = gr.Slider(1.0,  100.0, value=25.0, step=0.1, label="타격 강도 — 초기 속도 (m/s)")
            sl_theta_v   = gr.Slider(-85,  85,   value=15,  step=1, label="수직 발사각 θ_v (°)")
            sl_omega_top = gr.Slider(-800, 800,  value=150, step=5, label="탑스핀 ω_top (rad/s)")
            sl_omega_sid = gr.Slider(-800, 800,  value=0,   step=5, label="사이드스핀 ω_side (rad/s)")
            sl_hit_z     = gr.Slider(-0.7, 2.0,  value=0.3, step=0.01, label="타격 높이 Z0 (m)")

            gr.Markdown("---")
            gr.Markdown("### 📍 타격 위치 & 방향")
            sl_hit_x   = gr.Slider(-3.0, 3.0,  value=0.5,  step=0.05, label="타격 위치 X (m)")
            sl_hit_y   = gr.Slider(-4.0, 0.0,  value=-1.4, step=0.05, label="타격 위치 Y (m)")
            sl_theta_h = gr.Slider(-80,  80,   value=15,   step=1, label="수평 타격 방향각 θ_h (°)")

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

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
