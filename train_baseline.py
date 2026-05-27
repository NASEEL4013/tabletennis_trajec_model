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
BATCH_SIZE = 256
LR = 0.001
PATIENCE = 30  # Early Stopping 인내심

class PhysicsDecoderMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 기교 없는 순수 Expanding MLP (병목현상 제거)
        self.net = nn.Sequential(
            nn.Linear(8, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 1500)
        )
        
    def forward(self, x):
        out = self.net(x)
        return out.view(-1, 500, 3)

def get_kinematic_loss(pred, target, dt=0.002):
    # Velocity (속도) - 폭발 방지를 위해 L1 Loss 사용
    v_pred = (pred[:, 1:, :] - pred[:, :-1, :]) / dt
    v_target = (target[:, 1:, :] - target[:, :-1, :]) / dt
    loss_v = nn.L1Loss()(v_pred, v_target)
    
    # Acceleration (가속도) - 폭발 방지를 위해 L1 Loss 사용
    a_pred = (v_pred[:, 1:, :] - v_pred[:, :-1, :]) / dt
    a_target = (v_target[:, 1:, :] - v_target[:, :-1, :]) / dt
    loss_a = nn.L1Loss()(a_pred, a_target)
    
    return loss_v, loss_a

def train(split_type="random", attempt_name="attempt2_kinematic"):
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
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # 4. 모델, 손실 함수, 옵티마이저 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = PhysicsDecoderMLP().to(device)
    criterion = nn.MSELoss() # 순수 MSE Loss (시도 1 베이스)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    best_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    model_save_path = os.path.join(DB_DIR, f"mlp_standard_{split_type}_{attempt_name}.pth")
    
    # 5. 학습 루프
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_pos_loss = 0.0
        train_kin_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{EPOCHS} [Train]", leave=False)
        for batch_X, batch_y in train_pbar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            preds = model(batch_X)
            
            # Position Loss
            loss_pos = criterion(preds, batch_y)
            
            # Kinematic Loss (L1 - 가속도 폭발 방지)
            loss_v, loss_a = get_kinematic_loss(preds, batch_y, dt=0.002)
            
            # Total Loss (가중치 밸런싱)
            kin_total = 0.01 * loss_v + 0.0001 * loss_a
            loss = loss_pos + kin_total
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 기울기 폭발 방지
            optimizer.step()
            
            bs = batch_X.size(0)
            train_loss += loss.item() * bs
            train_pos_loss += loss_pos.item() * bs
            train_kin_loss += kin_total.item() * bs
            
            train_pbar.set_postfix({'Pos': f"{loss_pos.item():.4f}", 'Kin': f"{kin_total.item():.4f}"})
            
        train_loss /= len(train_loader.dataset)
        train_pos_loss /= len(train_loader.dataset)
        train_kin_loss /= len(train_loader.dataset)
        
        # 6. 평가 (Test Loss)
        model.eval()
        test_loss = 0.0
        test_pos_loss = 0.0
        test_kin_loss = 0.0
        
        test_pbar = tqdm(test_loader, desc=f"Epoch {epoch:03d}/{EPOCHS} [Test ]", leave=False)
        with torch.no_grad():
            for batch_X, batch_y in test_pbar:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                preds = model(batch_X)
                
                loss_pos = criterion(preds, batch_y)
                loss_v, loss_a = get_kinematic_loss(preds, batch_y, dt=0.002)
                kin_total = 0.01 * loss_v + 0.0001 * loss_a
                loss = loss_pos + kin_total
                
                bs = batch_X.size(0)
                test_loss += loss.item() * bs
                test_pos_loss += loss_pos.item() * bs
                test_kin_loss += kin_total.item() * bs
                test_pbar.set_postfix({'Pos': f"{loss_pos.item():.4f}", 'Kin': f"{kin_total.item():.4f}"})
                
        test_loss /= len(test_loader.dataset)
        test_pos_loss /= len(test_loader.dataset)
        test_kin_loss /= len(test_loader.dataset)
        
        scheduler.step(test_loss)
        
        if epoch % 1 == 0:
            print(f"Epoch {epoch:03d}/{EPOCHS} | Train [Pos: {train_pos_loss:.4f}, Kin: {train_kin_loss:.4f}] | Test [Pos: {test_pos_loss:.4f}, Kin: {test_kin_loss:.4f}]")
            
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
    print(f"✅ Training complete in {total_time:.1f}s! Best Test MSE: {best_loss:.5f}")
    print(f"💾 Model saved to: {model_save_path}\n")

if __name__ == "__main__":
    train(split_type="random", attempt_name="attempt2_kinematic")
