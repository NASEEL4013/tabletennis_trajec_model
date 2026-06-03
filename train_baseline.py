import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

DB_DIR = "/root/myresearch/database"
EPOCHS = 500
BATCH_SIZE = 1024
LR = 0.001
PATIENCE = 30  # Early Stopping 인내심

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

def train(split_type="random", attempt_name="attempt16_scaled_mlp"):
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
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = TimeConditionedMLP().to(device)
    criterion = nn.L1Loss() # MAE Loss (중앙값 추적) 도입
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    scaler = torch.cuda.amp.GradScaler()
    
    best_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    model_save_path = os.path.join(DB_DIR, f"mlp_standard_{split_type}_{attempt_name}.pth")
    
    def compute_masked_loss(preds, batch_y):
        # 1. 바닥 충돌 패딩 마스킹 (Z <= -0.75)
        # batch_y shape: (batch_size, 500, 3)
        z_coords = batch_y[:, :, 2]
        is_floor = (z_coords <= -0.75).float()
        
        # 바닥에 한 번이라도 닿으면 이후 영원히 1이 유지되도록 함
        has_hit = (is_floor.cumsum(dim=1) > 0).float()
        # 한 칸 오른쪽으로 밀어서 바닥에 닿는 첫 프레임까지는 마스크가 1이 되도록 허용
        has_hit_shifted = torch.cat([torch.zeros_like(has_hit[:, :1]), has_hit[:, :-1]], dim=1)
        mask = 1.0 - has_hit_shifted # (batch_size, 500)
        
        # 2. 오차 계산 (Mask 적용)
        error = torch.abs(preds - batch_y) * mask.unsqueeze(-1)
        
        # 3. 평균 나누기 (활성화된 스텝 개수로만 나눔)
        total_active_elements = mask.sum() * 3
        
        if total_active_elements > 0:
            pos_loss = error.sum() / total_active_elements
        else:
            pos_loss = error.sum() * 0.0 # fallback
            
        total_loss = pos_loss
            
        return total_loss, error.sum(), total_active_elements

    # 5. 학습 루프
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_error = 0.0
        total_train_active = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{EPOCHS} [Train]", leave=False)
        for batch_X, batch_y in train_pbar:
            batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                B = batch_X.shape[0]
                t = (torch.arange(500, device=device, dtype=torch.float32) * 0.002).unsqueeze(-1)
                x_expanded = batch_X.unsqueeze(1).expand(B, 500, 8).reshape(B * 500, 8)
                t_expanded = t.unsqueeze(0).expand(B, 500, 1).reshape(B * 500, 1)
                preds = model(x_expanded, t_expanded).view(B, 500, 3)
                loss, batch_error_sum, batch_active_count = compute_masked_loss(preds, batch_y)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 기울기 폭발 방지
            scaler.step(optimizer)
            scaler.update()
            
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
                B = batch_X.shape[0]
                t = (torch.arange(500, device=device, dtype=torch.float32) * 0.002).unsqueeze(-1)
                x_expanded = batch_X.unsqueeze(1).expand(B, 500, 8).reshape(B * 500, 8)
                t_expanded = t.unsqueeze(0).expand(B, 500, 1).reshape(B * 500, 1)
                preds = model(x_expanded, t_expanded).view(B, 500, 3)
                
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
    train(split_type="random", attempt_name="attempt16_scaled_mlp")
