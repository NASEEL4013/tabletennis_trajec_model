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

def train(split_type="random", attempt_name="attempt5_resnet"):
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
    
    # 5. 학습 루프
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{EPOCHS} [Train]", leave=False)
        for batch_X, batch_y in train_pbar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            preds = model(batch_X)
            
            # Position Loss (L1 Loss)
            loss = criterion(preds, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 기울기 폭발 방지
            optimizer.step()
            
            bs = batch_X.size(0)
            train_loss += loss.item() * bs
            
            train_pbar.set_postfix({'MAE': f"{loss.item():.4f}"})
            
        train_loss /= len(train_loader.dataset)
        
        # 6. 평가 (Test Loss)
        model.eval()
        test_loss = 0.0
        
        test_pbar = tqdm(test_loader, desc=f"Epoch {epoch:03d}/{EPOCHS} [Test ]", leave=False)
        with torch.no_grad():
            for batch_X, batch_y in test_pbar:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                preds = model(batch_X)
                
                loss = criterion(preds, batch_y)
                
                bs = batch_X.size(0)
                test_loss += loss.item() * bs
                test_pbar.set_postfix({'MAE': f"{loss.item():.4f}"})
                
        test_loss /= len(test_loader.dataset)
        
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
    train(split_type="random", attempt_name="attempt5_resnet")
