import os
import math
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.spatial import distance

# =========================
# 1. 共通設定
# =========================
T_STEPS = 48
D = 10
BATCH_SIZE = 256
EPOCHS = 30
LR = 2e-4
DIFF_STEPS = 1000

TARGET_NUM = 2000      # 生成するエージェント数
GEN_BATCH = 500        # 生成時のバッチサイズ

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 2. データセット定義
# =========================
class TripDataset(Dataset):
    def __init__(self, x_path, m_path):
        self.X = np.load(x_path).astype(np.float32)
        self.M = np.load(m_path).astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.M[idx])

# =========================
# 3. 拡散モデル構造 (Conv1d + Transformer)
# =========================
def sinusoidal_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device).float() / half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((t.shape[0], 1), device=t.device)], dim=1)
    return emb

class DiffusionTransformer(nn.Module):
    def __init__(self, d_in=10, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_model)
        self.local_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.pos_emb = nn.Parameter(torch.zeros(1, T_STEPS, d_model))
        self.t_mlp = nn.Sequential(
            nn.Linear(128, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.out_proj = nn.Linear(d_model, d_in)

    def forward(self, x, t):
        h = self.in_proj(x).permute(0, 2, 1)
        h = self.act(self.local_conv(h)).permute(0, 2, 1)
        h = h + self.pos_emb
        h = h + self.t_mlp(sinusoidal_embedding(t, 128)).unsqueeze(1)
        return self.out_proj(self.encoder(h))

# =========================
# 4. DDPM & DDIM サンプリング処理
# =========================
class DDPM:
    def __init__(self, n_steps=1000):
        self.n_steps = n_steps
        self.betas = torch.linspace(1e-4, 0.02, n_steps).to(DEVICE)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise):
        a = torch.sqrt(self.alphas_bar[t]).view(-1, 1, 1)
        b = torch.sqrt(1.0 - self.alphas_bar[t]).view(-1, 1, 1)
        return a * x0 + b * noise

@torch.no_grad()
def sample_ddim(model, ddpm, n_samples, mean, std):
    """決定論的サンプリング (eta=0)"""
    model.eval()
    x = torch.randn(n_samples, T_STEPS, D, device=DEVICE)

    for step in reversed(range(ddpm.n_steps)):
        t = torch.full((n_samples,), step, device=DEVICE, dtype=torch.long)
        eps = model(x, t)

        alpha_bar = ddpm.alphas_bar[step]
        alpha_bar_prev = ddpm.alphas_bar[step - 1] if step > 0 else torch.tensor(1.0).to(DEVICE)

        pred_x0 = (x - torch.sqrt(1 - alpha_bar) * eps) / torch.sqrt(alpha_bar)
        x = torch.sqrt(alpha_bar_prev) * pred_x0 + torch.sqrt(1 - alpha_bar_prev) * eps

    return (x * std + mean).cpu().numpy()

# =========================
# 5. ポストプロセス処理群
# =========================
def smooth_vectors(samples, window_size=5):
    """ベクトル平滑化によるノイズ除去"""
    smoothed = np.zeros_like(samples)
    for i in range(samples.shape[0]):
        df_smooth = pd.DataFrame(samples[i]).rolling(window=window_size, center=True, min_periods=1).mean()
        smoothed[i] = df_smooth.values
    return smoothed

def decode_to_mesh_codes(smooth_samples, dict_csv_path, decode_batch=2000):
    """潜在空間のベクトルをメッシュコードに逆変換"""
    df_dict = pd.read_csv(dict_csv_path)
    dict_vecs = df_dict.filter(like="z").values
    dict_codes = df_dict["mesh4"].astype(str).values

    N_total, T, _ = smooth_samples.shape
    flat_samples = smooth_samples.reshape(N_total * T, D)
    
    results = []
    for i in range(0, len(flat_samples), decode_batch):
        batch = flat_samples[i:i + decode_batch]
        idxs = np.argmin(distance.cdist(batch, dict_vecs), axis=1)
        results.append(dict_codes[idxs])

    return np.concatenate(results).reshape(N_total, T)

# =========================
# 6. 学習及びメインルーチン
# =========================
def train_or_load_model(model, ddpm, x_path, m_path, model_save_path, mean, std):
    if os.path.exists(model_save_path):
        print(f"Loading existing model from {model_save_path}...")
        model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
        return model

    print("Model not found. Starting Training...")
    loader = DataLoader(TripDataset(x_path, m_path), batch_size=BATCH_SIZE, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        losses = []
        for x, m in loader:
            x, m = x.to(DEVICE), m.to(DEVICE)
            x_norm = (x - mean) / std

            t = torch.randint(0, DIFF_STEPS, (x.shape[0],), device=DEVICE)
            noise = torch.randn_like(x_norm)
            pred_noise = model(ddpm.q_sample(x_norm, t, noise), t)

            # Masked MSE (mask=1 only)
            loss = ((pred_noise - noise) ** 2 * m.unsqueeze(-1)).sum() / (m.sum() * D + 1e-8)

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {np.mean(losses):.4f}")

    torch.save(model.state_dict(), model_save_path)
    return model

def run_one(tag, x_path, m_path, dict_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model_save_path = os.path.join(out_dir, f"best_model_{tag}.pth")
    # 上書きを防ぐためのタイムスタンプ付き識別子
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*60}\nRUN: {tag}")
    
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Missing data file: {x_path}\n"
                                "Please place the sample array data in the './data/input/' directory.")

    # マスク箇所のみ抽出した統計量計算
    X_all, M_all = np.load(x_path).astype(np.float32), np.load(m_path).astype(np.float32)
    X_masked = X_all[M_all == 1]
    mean = torch.tensor(X_masked.mean(0), device=DEVICE)
    std = torch.tensor(X_masked.std(0) + 1e-8, device=DEVICE)

    model = train_or_load_model(DiffusionTransformer().to(DEVICE), DDPM(DIFF_STEPS), x_path, m_path, model_save_path, mean, std)

    print(f"Generating {TARGET_NUM} samples...")
    generated_list = []
    num_generated = 0
    while num_generated < TARGET_NUM:
        batch_samples = sample_ddim(model, DDPM(DIFF_STEPS), min(GEN_BATCH, TARGET_NUM - num_generated), mean, std)
        generated_list.append(batch_samples)
        num_generated += len(batch_samples)

    raw_samples = np.concatenate(generated_list, axis=0)
    smooth_samples = smooth_vectors(raw_samples, window_size=5)
    final_codes = decode_to_mesh_codes(smooth_samples, dict_path)

    # 保存処理
    csv_path = os.path.join(out_dir, f"final_trips_{tag}_{TARGET_NUM}_{timestamp}.csv")
    pd.DataFrame(final_codes).to_csv(csv_path, index=False)
    np.save(os.path.join(out_dir, f"generated_raw_{tag}_{TARGET_NUM}_{timestamp}.npy"), raw_samples.astype(np.float32))
    np.save(os.path.join(out_dir, f"generated_smooth_{tag}_{TARGET_NUM}_{timestamp}.npy"), smooth_samples.astype(np.float32))

    print(f"Process complete. Results saved to {out_dir}")

def main():
    print(f"DEVICE = {DEVICE}")
    # ポートフォリオ用の相対パス設計
    configs = [
        {
            "tag": "MSE",
            "x_path": r"./data/input/X_trip_48x10.npy",
            "m_path": r"./data/input/mask_48.npy",
            "dict_path": r"./data/dictionary/mesh4_to_z10_MSE.csv",
            "out_dir": r"./outputs/diffusion",
        }
    ]

    for cfg in configs:
        run_one(**cfg)

if __name__ == "__main__":
    main()
