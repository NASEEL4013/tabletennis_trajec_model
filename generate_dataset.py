import numpy as np
import os
import time
from app import traditional_physics_solver

DB_DIR = "/root/myresearch/database"
NUM_SAMPLES = 5000000

def sample_trunc_norm(mean, std, min_val, max_val, size):
    result = np.empty(size)
    remaining_indices = np.arange(size)
    while len(remaining_indices) > 0:
        needed = len(remaining_indices)
        samples = np.random.normal(mean, std, needed)
        valid_mask = (samples >= min_val) & (samples <= max_val)
        
        result[remaining_indices[valid_mask]] = samples[valid_mask]
        remaining_indices = remaining_indices[~valid_mask]
    return result

def sample_trunc_lognorm(mean, sigma, min_val, max_val, size):
    result = np.empty(size)
    remaining_indices = np.arange(size)
    while len(remaining_indices) > 0:
        needed = len(remaining_indices)
        samples = np.random.lognormal(mean, sigma, needed)
        valid_mask = (samples >= min_val) & (samples <= max_val)
        
        result[remaining_indices[valid_mask]] = samples[valid_mask]
        remaining_indices = remaining_indices[~valid_mask]
    return result

def generate_samples(num_samples):
    # 입력: [speed, theta_v_deg, omega_top, omega_side, hit_x, hit_y, hit_z, theta_h]
    inputs = np.zeros((num_samples, 8), dtype=np.float32)
    # 출력: 750스텝의 (X, Y, Z) 좌표
    outputs = np.zeros((num_samples, 750, 3), dtype=np.float32)
    
    print(f"Generating {num_samples} trajectories...")
    start_time = time.time()
    
    speed_all = sample_trunc_lognorm(1.8, 0.8, 1.0, 80.0, num_samples)
    theta_v_all = sample_trunc_norm(15.0, 15.0, -50.0, 50.0, num_samples)
    omega_top_all = sample_trunc_norm(50.0, 50.0, -200.0, 200.0, num_samples)
    omega_side_all = sample_trunc_norm(0.0, 50.0, -200.0, 200.0, num_samples)
    hit_x_all = sample_trunc_norm(0.0, 0.75, -3.0, 3.0, num_samples)
    hit_y_all = sample_trunc_norm(-1.5, 0.5, -4.0, 0.0, num_samples)
    hit_z_all = sample_trunc_norm(0.3, 0.3, -0.76, 1.5, num_samples)
    theta_h_all = sample_trunc_norm(0.0, 20.0, -65.0, 65.0, num_samples)
    
    for i in range(num_samples):
        if i > 0 and i % 5000 == 0:
            elapsed = time.time() - start_time
            print(f"Progress: {i}/{num_samples} (Elapsed: {elapsed:.2f}s)")
            
        speed = speed_all[i]
        theta_v = theta_v_all[i]
        omega_top = omega_top_all[i]
        omega_side = omega_side_all[i]
        hit_x = hit_x_all[i]
        hit_y = hit_y_all[i]
        hit_z = hit_z_all[i]
        theta_h = theta_h_all[i]
        
        inputs[i] = np.round([speed, theta_v, omega_top, omega_side, hit_x, hit_y, hit_z, theta_h], 3)
        
        # 월드 좌표 기반 물리 엔진으로 궤적 생성
        traj = traditional_physics_solver(speed, theta_v, omega_top, omega_side, hit_x, hit_y, hit_z, theta_h, steps=750, dt=0.002)
        outputs[i] = np.round(traj, 3)
        
    return inputs, outputs

if __name__ == "__main__":
    np.random.seed(42) # 재현성을 위해 시드 고정
    
    inputs, outputs = generate_samples(NUM_SAMPLES)
    os.makedirs(DB_DIR, exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. Random Split 데이터셋 (80% Train, 20% Test)
    # ---------------------------------------------------------
    indices = np.random.permutation(NUM_SAMPLES)
    train_size = int(0.8 * NUM_SAMPLES)
    train_idx, test_idx = indices[:train_size], indices[train_size:]
    
    np.savez_compressed(os.path.join(DB_DIR, "dataset_gaussian_mixed.npz"),
                        train_inputs=inputs[train_idx], train_outputs=outputs[train_idx],
                        test_inputs=inputs[test_idx], test_outputs=outputs[test_idx])
    
    print("\n✅ 데이터셋 생성 완료!")
    print(f"순수 무작위(Random) 80/20 분할 -> Train: {len(train_idx)}개, Test: {len(test_idx)}개")
