import numpy as np
import os
import time
from app import traditional_physics_solver

DB_DIR = "/root/myresearch/database"
NUM_SAMPLES = 2000000

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
            
        # 파라미터 무작위 샘플링 (프로 레벨 강스매시 및 하이 스핀 커버)
        speed = np.random.uniform(1.0, 100.0)
        theta_v = np.random.uniform(-85.0, 85.0)
        omega_top = np.random.uniform(-800.0, 800.0)
        omega_side = np.random.uniform(-800.0, 800.0)
        hit_x = np.random.uniform(-3.0, 3.0)
        hit_y = np.random.uniform(-4.0, 0.0)
        hit_z = np.random.uniform(-0.7, 2.0)
        theta_h = np.random.uniform(-80.0, 80.0)
        
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
    
    np.savez_compressed(os.path.join(DB_DIR, "dataset_random.npz"),
                        train_inputs=inputs[train_idx], train_outputs=outputs[train_idx],
                        test_inputs=inputs[test_idx], test_outputs=outputs[test_idx])
    
    print("\n✅ 데이터셋 생성 완료!")
    print(f"순수 무작위(Random) 80/20 분할 -> Train: {len(train_idx)}개, Test: {len(test_idx)}개")
