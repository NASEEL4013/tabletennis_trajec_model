import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

DB_DIR = "/root/myresearch/database"
EPOCHS = 500
BATCH_SIZE = 4096
LR = 0.001
PATIENCE = 30  # Early Stopping 인내심

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
        return self.block(x) + x # Skip Connection (Pre-Activation)

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

def train(split_type="random", attempt_name="attempt7_cosine_loss"):
    print(f"==================================================")
    print(f"🚀 Training [REBOOT - {attempt_name.upper()}] Model with [{split_type.upper()}] Split")
    print(f"==================================================")
    
    # 1. 데이터 로드
    if split_type == "random":
        data = np.load(os.path.join(DB_DIR, "dataset_random.npz"))
    else:
        data = np.load(os.path.join(DB_DIR, "dataset_disjoint_speed.npz"))
        
    X_train, y_train = data['train_inputs'], data['train_outputs']
    X_test, y_test = data['test_inputs'], data['test_outputs']
    
    # 2. 입력 파라미터 Z-score 정규화
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    
    np.save(os.path.join(DB_DIR, f"mlp_norm_standard_{split_type}_{attempt_name}.npy"), np.vstack([mean, std]))
    
    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)
    
    # 3. PyTorch DataLoader 구성
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    # 4. 모델, 손실 함수, 옵티마이저 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = PhysicsDecoderMLP().to(device)
    criterion = nn.L1Loss() # MAE Loss (중앙값 추적) 도입
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    best_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    model_save_path = os.path.join(DB_DIR, f"mlp_standard_{split_type}_{attempt_name}.pth")
    
    def compute_masked_loss(preds, batch_y):
        # 1. 바닥 충돌 패딩 마스킹 (Z <= -0.75)
        # batch_y shape: (batch_size, 500, 3)
        z_coords = batch_y[:, :, 2]
        is_floor = z_coords <= -0.75
        
        # cumsum을 사용하여 바닥에 처음 닿은 시점(inclusive) 이후의 마스크 생성
        # hit_floor_cumsum == 0 인 구간만 True (살림)
        mask = (is_floor.cumsum(dim=1) == 0).float() # (batch_size, 500)
        
        # 2. 오차 계산 (Mask 적용)
        error = torch.abs(preds - batch_y) * mask.unsqueeze(-1)
        
        # 3. 평균 나누기 (활성화된 스텝 개수로만 나눔)
        # mask.sum()은 전체 배치 내에서 살려진 총 스텝 개수
        total_active_elements = mask.sum() * 3
        
        if total_active_elements > 0:
            pos_loss = error.sum() / total_active_elements
        else:
            pos_loss = error.sum() * 0.0 # fallback
            
        # 4. 방향 일치 (Cosine Similarity) Loss 추가
        pred_vel = preds[:, 1:, :] - preds[:, :-1, :]
        true_vel = batch_y[:, 1:, :] - batch_y[:, :-1, :]
        
        cos_sim = torch.nn.functional.cosine_similarity(pred_vel, true_vel, dim=-1) # (batch_size, 499)
        
        # t와 t+1 모두 유효한(마스킹 안 된) 스텝만 비교에 포함
        vel_mask = mask[:, :-1] * mask[:, 1:] # (batch_size, 499)
        
        dir_error = (1.0 - cos_sim) * vel_mask
        total_active_vels = vel_mask.sum()
        
        if total_active_vels > 0:
            dir_loss = dir_error.sum() / total_active_vels
        else:
            dir_loss = dir_error.sum() * 0.0
            
        # 최종 Loss: 위치 오차(pos_loss) + 방향 오차 가중치(alpha)
        alpha = 1.0
        total_loss = pos_loss + alpha * dir_loss
            
        # 반환값: 역전파용 total_loss, 그리고 실제 MAE 로깅용 위치 오차합(error.sum())
        return total_loss, error.sum(), total_active_elements

    # 5. 학습 루프
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_error = 0.0
        total_train_active = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{EPOCHS} [Train]", leave=False)
        for batch_X, batch_y in train_pbar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            preds = model(batch_X)
            
            loss, batch_error_sum, batch_active_count = compute_masked_loss(preds, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 기울기 폭발 방지
            optimizer.step()
            
            total_train_error += batch_error_sum.item()
            total_train_active += batch_active_count.item()
            
            train_pbar.set_postfix({'MAE': f"{loss.item():.4f}"})
            
        train_loss = total_train_error / total_train_active if total_train_active > 0 else 0.0
        
        # 6. 평가 (Test Loss)
        model.eval()
        total_test_error = 0.0
        total_test_active = 0.0
        
        test_pbar = tqdm(test_loader, desc=f"Epoch {epoch:03d}/{EPOCHS} [Test ]", leave=False)
        with torch.no_grad():
            for batch_X, batch_y in test_pbar:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                preds = model(batch_X)
                
                loss, batch_error_sum, batch_active_count = compute_masked_loss(preds, batch_y)
                
                total_test_error += batch_error_sum.item()
                total_test_active += batch_active_count.item()
                test_pbar.set_postfix({'MAE': f"{loss.item():.4f}"})
                
        test_loss = total_test_error / total_test_active if total_test_active > 0 else 0.0
        
        scheduler.step(test_loss)
        
        if epoch % 1 == 0:
            print(f"Epoch {epoch:03d}/{EPOCHS} | Train MAE: {train_loss:.5f} | Test MAE: {test_loss:.5f}")
            
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            
        if patience_counter >= PATIENCE:
            print(f"\n🛑 Early stopping triggered at epoch {epoch}! No improvement for {PATIENCE} epochs.")
            break
            
    total_time = time.time() - start_time
    print(f"✅ Training complete in {total_time:.1f}s! Best Test MAE: {best_loss:.5f}")
    print(f"💾 Model saved to: {model_save_path}\n")

if __name__ == "__main__":
    train(split_type="random", attempt_name="attempt7_cosine_loss")
