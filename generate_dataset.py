import numpy as np
import os
import time
from app import traditional_physics_solver

DB_DIR = "/root/myresearch/database"
NUM_SAMPLES = 3000000

def generate_samples(num_samples):
    # 입력: [speed, theta_v_deg, omega_top, omega_side, hit_x, hit_y, hit_z, theta_h]
    inputs = np.zeros((num_samples, 8), dtype=np.float32)
    # 출력: 500스텝의 (X, Y, Z) 좌표
    outputs = np.zeros((num_samples, 500, 3), dtype=np.float32)
    
    print(f"Generating {num_samples} trajectories...")
    start_time = time.time()
    
    for i in range(num_samples):
        if i > 0 and i % 5000 == 0:
            elapsed = time.time() - start_time
            print(f"Progress: {i}/{num_samples} (Elapsed: {elapsed:.2f}s)")
            
        # 파라미터 무작위 샘플링 (정규 분포: 정상 범위 70% / 이상치 30% 비율로 정밀 튜닝)
        speed = np.clip(np.random.lognormal(mean=1.8, sigma=0.8), 1.0, 80.0)
        theta_v = np.clip(np.random.normal(15.0, 15.0), -50.0, 50.0)
        omega_top = np.clip(np.random.normal(50.0, 50.0), -200.0, 200.0)
        omega_side = np.clip(np.random.normal(0.0, 50.0), -200.0, 200.0)
        hit_x = np.clip(np.random.normal(0.0, 0.75), -3.0, 3.0)
        hit_y = np.clip(np.random.normal(-1.5, 0.5), -4.0, 0.0)
        hit_z = np.clip(np.random.normal(0.3, 0.3), -0.76, 1.5)
        theta_h = np.clip(np.random.normal(0.0, 20.0), -65.0, 65.0)
        
        inputs[i] = np.round([speed, theta_v, omega_top, omega_side, hit_x, hit_y, hit_z, theta_h], 3)
        
        # 월드 좌표 기반 물리 엔진으로 궤적 생성
        traj = traditional_physics_solver(speed, theta_v, omega_top, omega_side, hit_x, hit_y, hit_z, theta_h, steps=500, dt=0.002)
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
