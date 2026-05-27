import numpy as np
import os

file_path = '/root/myresearch/database/dataset_random.npz'
file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
print(f"✅ File Size: {file_size_mb:.2f} MB\n")

data = np.load(file_path)

print("✅ Dataset Shapes & Integrity:")
for key in data.keys():
    arr = data[key]
    has_nan = np.isnan(arr).any()
    has_inf = np.isinf(arr).any()
    print(f" - [{key}] Shape: {arr.shape}, Dtype: {arr.dtype} | NaN: {has_nan}, Inf: {has_inf}")

# 물리 시뮬레이션 한계치 검사 (랜덤 1,000개 샘플링)
train_out = data['train_outputs'][:1000]
print("\n✅ Physics Engine Sanity Check (1,000 samples):")
print(f" - Z (고도) Min/Max : {np.min(train_out[:, :, 2]):.3f}m ~ {np.max(train_out[:, :, 2]):.3f}m  (바닥 -0.76m 이내 유지 여부)")
print(f" - X (좌우) Min/Max : {np.min(train_out[:, :, 0]):.3f}m ~ {np.max(train_out[:, :, 0]):.3f}m")
print(f" - Y (앞뒤) Min/Max : {np.min(train_out[:, :, 1]):.3f}m ~ {np.max(train_out[:, :, 1]):.3f}m")
