import argparse
import os
import time
import json
import tracemalloc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.integrate import solve_ivp

DB_DIR = "/root/myresearch/database"
RESULTS_DIR = "/root/myresearch/results"
K_MAGNUS = 0.0034
K_DRAG = 0.112

os.makedirs(RESULTS_DIR, exist_ok=True)

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
        self.fc_in = nn.Sequential(nn.Linear(in_dim, 256), nn.SiLU())
        self.blocks = nn.ModuleList([ResBlock(256) for _ in range(3)])
        self.fc_out = nn.Linear(256, 3)
    def forward(self, x, t):
        freq_bands = 2.0 ** torch.linspace(0, self.num_freqs - 1, self.num_freqs, device=t.device)
        t_freqs = t * freq_bands * torch.pi
        t_enc = torch.cat([t, torch.sin(t_freqs), torch.cos(t_freqs)], dim=-1)
        features = torch.cat([x, t_enc], dim=-1)
        h = self.fc_in(features)
        for block in self.blocks: h = block(h)
        return self.fc_out(h)

def traditional_physics_solver(speed, theta_v_deg, omega_top, omega_side, hit_x, hit_y, hit_z, theta_h_deg, steps=750, dt=0.002):
    theta_v = np.radians(theta_v_deg)
    theta_h = np.radians(theta_h_deg)
    vy0 = speed * np.cos(theta_v) * np.cos(theta_h)
    vx0 = -speed * np.cos(theta_v) * np.sin(theta_h)
    vz0 = speed * np.sin(theta_v)
    initial_state = [hit_x, hit_y, hit_z, vx0, vy0, vz0]
    omega_world = np.array([-omega_top * np.cos(theta_h), -omega_top * np.sin(theta_h), omega_side])
    
    def derivatives(t, state):
        x, y, z, vx, vy, vz = state
        vel = np.array([vx, vy, vz])
        spd = np.linalg.norm(vel)
        if spd < 1e-6: return [vx, vy, vz, 0.0, 0.0, -9.81]
        drag = -K_DRAG * spd * vel
        magnus = np.cross(omega_world, vel) * K_MAGNUS
        acc = np.array([0.0, 0.0, -9.81]) + drag + magnus
        return [vx, vy, vz, acc[0], acc[1], acc[2]]

    def hit_table(t, state): return state[2]
    hit_table.terminal = True
    hit_table.direction = -1

    def hit_floor(t, state): return state[2] - (-0.76)
    hit_floor.terminal = True
    hit_floor.direction = -1

    def hit_net(t, state): return state[1]
    hit_net.terminal = True
    hit_net.direction = 0

    t_eval = np.linspace(0, steps * dt, steps + 1)
    traj = []
    current_t, current_state = 0.0, initial_state
    times_needed = list(t_eval)
    
    while len(times_needed) > 0:
        t_end = times_needed[-1]
        if current_t >= t_end: break
        sol = solve_ivp(derivatives, (current_t, t_end), current_state, t_eval=times_needed, events=[hit_table, hit_floor, hit_net], rtol=1e-5, atol=1e-8)
        if len(sol.t) > 0:
            for i in range(len(sol.t)): traj.append(sol.y[:3, i].copy())
            times_needed = times_needed[len(sol.t):]
        if sol.status == 1:
            event_idx, event_t = -1, -1
            for i in range(3):
                if len(sol.t_events[i]) > 0 and sol.t_events[i][-1] > event_t:
                    event_t = sol.t_events[i][-1]
                    event_idx = i
            if event_idx == -1:
                current_t = sol.t[-1] if len(sol.t) > 0 else current_t + dt
                continue
            current_t = event_t
            current_state = sol.y_events[event_idx][-1].copy()
            x, y, z, vx, vy, vz = current_state
            if event_idx == 0:
                if abs(x) <= 0.76 and abs(y) <= 1.37:
                    vz_out = -0.83 * vz
                    R_ball = 0.02
                    vy_out = 0.7 * vy + 0.1 * omega_top * R_ball * np.cos(theta_h)
                    vx_out = 0.7 * vx + 0.1 * omega_top * R_ball * (-np.sin(theta_h)) + 0.1 * (-omega_side) * R_ball
                    current_state[3:6] = [vx_out, vy_out, vz_out]
                    current_state[2] = 1e-4
                else: current_state[2] = -1e-4
            elif event_idx == 1:
                break  # 바닥에 닿으면 즉시 궤적 시뮬레이션 종료
            elif event_idx == 2:
                if abs(x) <= 0.76 and z <= 0.1525:
                    current_state[3:6] = [0.5*vx, -0.3*vy, 0.5*vz]
                    current_state[1] = 1e-4 if vy < 0 else -1e-4
                else: current_state[1] = 1e-4 if vy > 0 else -1e-4

    traj = np.array(traj)
    if len(traj) > steps: traj = traj[:steps]
    elif len(traj) < steps: traj = np.vstack([traj, np.tile(traj[-1], (steps - len(traj), 1))])
    return traj

def run_benchmark(model_name, norm_name, dataset_name, n_samples):
    import psutil
    print(f"🔄 Starting Rigorous Benchmark...")
    
    # 1. 모델 및 데이터 로드
    model = TimeConditionedMLP()
    model.load_state_dict(torch.load(os.path.join(DB_DIR, model_name), map_location="cpu", weights_only=True))
    model.eval()
    
    norm_data = np.load(os.path.join(DB_DIR, norm_name))
    mean, std = norm_data[0], norm_data[1]
    
    data = np.load(os.path.join(DB_DIR, dataset_name))
    X_test, y_test = data['test_inputs'], data['test_outputs']
    n_samples = min(n_samples, len(X_test))
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    X_sample, y_sample = X_test[indices], y_test[indices]
    
    # 엣지 케이스 인덱스
    speeds = X_sample[:, 0]
    spins = np.sqrt(X_sample[:, 2]**2 + X_sample[:, 3]**2)
    edge_idx_speed = np.where(speeds >= np.percentile(speeds, 95))[0]
    edge_idx_spin = np.where(spins >= np.percentile(spins, 95))[0]
    
    input_norm = (X_sample - mean) / (std + 1e-8)
    input_cpu = torch.tensor(input_norm, dtype=torch.float32)
    
    process = psutil.Process(os.getpid())
    
    # 2. GT 물리엔진 성능 측정
    print(">> Measuring GT Physics Engine...")
    gt_samples = n_samples
    mem_before_gt = process.memory_info().rss
    t0_trad = time.perf_counter()
    for i in range(gt_samples):
        _ = traditional_physics_solver(*X_sample[i])
    trad_time = time.perf_counter() - t0_trad
    mem_after_gt = process.memory_info().rss
    peak_mem_trad = max(0, mem_after_gt - mem_before_gt)
    
    trad_latency_ms = (trad_time / gt_samples) * 1000
    trad_throughput = 1000 / trad_latency_ms if trad_latency_ms > 0 else 0
    
    # 3. CPU 벤치마크
    print(">> Measuring AI on CPU...")
    model.to("cpu")
    t_cpu = (torch.arange(750, dtype=torch.float32) * 0.002).unsqueeze(1)
    
    # CPU 단일 처리 지연시간
    t0 = time.perf_counter()
    for i in range(gt_samples):
        with torch.no_grad():
            x_rep = input_cpu[i].unsqueeze(0).expand(750, 8)
            _ = model(x_rep, t_cpu).numpy()
    cpu_latency_ms = ((time.perf_counter() - t0) / gt_samples) * 1000
    
    # CPU 배치(전체) 처리 속도 및 메모리
    mem_before_ai = process.memory_info().rss
    t0 = time.perf_counter()
    batch_size_cpu = 500
    with torch.no_grad():
        for i in range(0, n_samples, batch_size_cpu):
            bs = min(i + batch_size_cpu, n_samples) - i
            t_batch = t_cpu.unsqueeze(0).expand(bs, 750, 1)
            x_rep = input_cpu[i:i+bs].unsqueeze(1).expand(bs, 750, 8)
            _ = model(x_rep, t_batch).numpy()
    cpu_batch_time = time.perf_counter() - t0
    mem_after_ai = process.memory_info().rss
    peak_mem_cpu = max(0, mem_after_ai - mem_before_ai)
    cpu_throughput = n_samples / cpu_batch_time if cpu_batch_time > 0 else 0
    
    # 4. GPU 벤치마크 및 결과물 저장
    gpu_latency_ms, gpu_throughput, peak_vram_gpu = 0, 0, 0
    pred_traj_list = []
    
    if torch.cuda.is_available():
        print(">> Measuring AI on GPU...")
        device_gpu = torch.device("cuda")
        model.to(device_gpu)
        input_gpu = input_cpu.to(device_gpu)
        t_gpu = t_cpu.to(device_gpu)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # GPU 단일 처리 지연시간
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(gt_samples):
            with torch.no_grad():
                x_rep = input_gpu[i].unsqueeze(0).expand(750, 8)
                _ = model(x_rep, t_gpu).cpu().numpy()
        torch.cuda.synchronize()
        gpu_latency_ms = ((time.perf_counter() - t0) / gt_samples) * 1000
        
        # GPU 배치(전체) 처리 속도 및 메모리
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        batch_size_gpu = 500
        with torch.no_grad():
            for i in range(0, n_samples, batch_size_gpu):
                bs = min(i + batch_size_gpu, n_samples) - i
                t_batch = t_gpu.unsqueeze(0).expand(bs, 750, 1)
                x_rep = input_gpu[i:i+bs].unsqueeze(1).expand(bs, 750, 8)
                pred = model(x_rep, t_batch).cpu().numpy()
                pred_traj_list.append(pred)
        torch.cuda.synchronize()
        gpu_batch_time = time.perf_counter() - t0
        peak_vram_gpu = torch.cuda.max_memory_allocated(device_gpu)
        gpu_throughput = n_samples / gpu_batch_time
    else:
        print("GPU not available, copying CPU outputs.")
        # GPU 없으면 CPU 계산 결과 사용
        batch_size_cpu = 500
        with torch.no_grad():
            for i in range(0, n_samples, batch_size_cpu):
                bs = min(i + batch_size_cpu, n_samples) - i
                t_batch = t_cpu.unsqueeze(0).expand(bs, 750, 1)
                x_rep = input_cpu[i:i+bs].unsqueeze(1).expand(bs, 750, 8)
                pred = model(x_rep, t_batch).numpy()
                pred_traj_list.append(pred)
                
    pred_traj = np.concatenate(pred_traj_list, axis=0)
    
    # 5. 오차 계산
    print(">> Calculating Accuracy...")
    def calculate_metrics(idx_list):
        if len(idx_list) == 0: return {"MAE": 0, "RMSE": 0, "Max": 0, "Count": 0}
        err_l1, err_l2, max_err = [], [], []
        for i in idx_list:
            gt, pred = y_sample[i], pred_traj[i]
            hf_gt = np.where(gt[:, 2] <= -0.75)[0]
            first_hit = hf_gt[0] if len(hf_gt) > 0 else 750
            if first_hit < 750: pred[first_hit:] = pred[first_hit]
            
            diff = np.linalg.norm(gt - pred, axis=1)
            err_l1.append(diff[:first_hit].mean() if first_hit > 0 else 0)
            err_l2.append(np.sqrt(np.mean(diff[:first_hit]**2)) if first_hit > 0 else 0)
            max_err.append(diff[:first_hit].max() if first_hit > 0 else 0)
            
        return {"MAE": float(np.mean(err_l1)*1000), "RMSE": float(np.mean(err_l2)*1000), "Max": float(np.max(max_err)*1000),
                "Count": len(idx_list)}

    ovr = calculate_metrics(np.arange(n_samples))
    espd = calculate_metrics(edge_idx_speed)
    espn = calculate_metrics(edge_idx_spin)
    
    # 결과 요약
    results = {
        "benchmark": {"model": model_name, "samples": n_samples},
        "performance": {
            "GT": {"latency_ms": trad_latency_ms, "throughput": trad_throughput, "RAM_MB": peak_mem_trad/1024/1024},
            "AI_CPU": {"latency_ms": cpu_latency_ms, "throughput": cpu_throughput, "RAM_MB": peak_mem_cpu/1024/1024},
            "AI_GPU": {"latency_ms": gpu_latency_ms, "throughput": gpu_throughput, "VRAM_MB": peak_vram_gpu/1024/1024}
        },
        "accuracy": {"overall": ovr, "speed": espd, "spin": espn}
    }
    
    with open(os.path.join(RESULTS_DIR, "benchmark_report.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n" + "="*60)
    print(" 📊 탁구 AI 엄밀 벤치마크 리포트 (CPU/GPU 분리)")
    print("="*60)
    print(f"1. 전체 궤도 출력 속도 및 2. CPU/GPU 성능 비교")
    print(f"  [GT 물리 엔진] 1개당: {trad_latency_ms:6.2f} ms | 초당 렌더링: {trad_throughput:7.1f} 개")
    print(f"  [AI 디코더 CPU] 1개당: {cpu_latency_ms:6.2f} ms | 초당 렌더링: {cpu_throughput:7.1f} 개")
    print(f"  [AI 디코더 GPU] 1개당: {gpu_latency_ms:6.2f} ms | 초당 렌더링: {gpu_throughput:7.1f} 개")
    print("-" * 60)
    print(f"3. 오차 분석 & 엣지 케이스")
    def print_err(name, m):
        print(f"  [{name}] N={m['Count']} | MAE: {m['MAE']:.2f}mm, Max: {m['Max']:.1f}mm")
    print_err("전체 평균", ovr)
    print_err("극단적 스피드", espd)
    print_err("극단적 스핀", espn)
    print("-" * 60)
    print(f"5. 메모리 사용량 (Peak)")
    print(f"  [GT 시스템 RAM] {peak_mem_trad/1024/1024:.2f} MB")
    print(f"  [AI 시스템 RAM] {peak_mem_cpu/1024/1024:.2f} MB")
    print(f"  [AI GPU VRAM]  {peak_vram_gpu/1024/1024:.2f} MB")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mlp_standard_gaussian_mixed_attempt19_fourier_freq10_pure_l1.pth")
    parser.add_argument("--norm", type=str, default="mlp_norm_standard_gaussian_mixed_attempt19_fourier_freq10_pure_l1.npy")
    parser.add_argument("--dataset", type=str, default="dataset_gaussian_mixed.npz")
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()
    run_benchmark(args.model, args.norm, args.dataset, args.samples)
