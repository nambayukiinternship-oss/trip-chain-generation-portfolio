import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.spatial import distance

# =========================
# 共通設定
# =========================
T_STEPS = 48
D = 10
BATCH_SIZE = 256
EPOCHS = 30
LR = 2e-4
DIFF_STEPS = 1000

TARGET_NUM = 2000      
GEN_BATCH = 500        

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE =", DEVICE)

# =========================
# Dataset
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
# Model (Conv1d + Transformer)
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
            nn.Linear(128, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4 * d_model, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.out_proj = nn.Linear(d_model, d_in)

    def forward(self, x, t):
        h = self.in_proj(x)
        # Conv1d expects (B, C, T)
        h = h.permute(0, 2, 1)
        h = self.act(self.local_conv(h))
        h = h.permute(0, 2, 1)

        h = h + self.pos_emb
        t_emb = sinusoidal_embedding(t, 128)
        h = h + self.t_mlp(t_emb).unsqueeze(1)

        h = self.encoder(h)
        return self.out_proj(h)

# =========================
# DDPM & DDIM
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

        # x0 推定
        pred_x0 = (x - torch.sqrt(1 - alpha_bar) * eps) / torch.sqrt(alpha_bar)

        # 次のx（ノイズ項なし）
        x = torch.sqrt(alpha_bar_prev) * pred_x0 + torch.sqrt(1 - alpha_bar_prev) * eps

    return (x * std + mean).cpu().numpy()

# =========================
# ベクトル平滑化
# =========================
def smooth_vectors(samples, window_size=5):
    N, T, D_dim = samples.shape
    smoothed = np.zeros_like(samples)
    for i in range(N):
        df_temp = pd.DataFrame(samples[i])
        df_smooth = df_temp.rolling(window=window_size, center=True, min_periods=1).mean()
        smoothed[i] = df_smooth.values
    return smoothed

# =========================
# デコード（最近傍）: バッチ処理
# =========================
def decode_to_mesh_codes(smooth_samples, dict_csv_path, decode_batch=2000):
    df_dict = pd.read_csv(dict_csv_path)
    dict_vecs = df_dict.filter(like="z").values
    dict_codes = df_dict["mesh4"].astype(str).values

    N_total, T, _ = smooth_samples.shape
    flat_samples = smooth_samples.reshape(N_total * T, D)

    results = []
    for i in range(0, len(flat_samples), decode_batch):
        batch = flat_samples[i:i + decode_batch]
        dists = distance.cdist(batch, dict_vecs)
        idxs = np.argmin(dists, axis=1)
        results.append(dict_codes[idxs])

        if (i // decode_batch) % 10 == 0:
            print(f"Decoding... {(i / len(flat_samples)) * 100:.1f}%")

    final_codes = np.concatenate(results).reshape(N_total, T)
    return final_codes

# =========================
# 学習 or ロード
# =========================
def train_or_load_model(model, ddpm, x_path, m_path, model_save_path, mean, std):
    if os.path.exists(model_save_path):
        print(f"Loading existing model from {model_save_path}...")
        model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
        return model

    print("Model not found. Starting Training...")
    ds = TripDataset(x_path, m_path)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        losses = []
        for x, m in loader:
            x = x.to(DEVICE)
            m = m.to(DEVICE)

            x_norm = (x - mean) / std

            t = torch.randint(0, DIFF_STEPS, (x.shape[0],), device=DEVICE)
            noise = torch.randn_like(x_norm)
            x_t = ddpm.q_sample(x_norm, t, noise)

            pred_noise = model(x_t, t)

            # Masked MSE (mask=1 only)
            # 安全のため分母にeps
            denom = m.sum() * D + 1e-8
            loss = ((pred_noise - noise) ** 2 * m.unsqueeze(-1)).sum() / denom

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {float(np.mean(losses)):.4f}")

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    return model

# =========================
# 1モデル分の一連処理
# =========================
def run_one(tag, x_path, m_path, dict_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model_save_path = os.path.join(out_dir, f"best_model_{tag}.pth")

    print("\n" + "=" * 60)
    print(f"RUN: {tag}")
    print("X_PATH   =", x_path)
    print("M_PATH   =", m_path)
    print("DICT_PATH=", dict_path)
    print("OUT_DIR  =", out_dir)

    # ---- 1) 統計量（mask=1の箇所のみ） ----
    print("Loading Data & Calculating Stats...")
    X_all = np.load(x_path).astype(np.float32)
    M_all = np.load(m_path).astype(np.float32)

    X_masked = X_all[M_all == 1]
    mean = torch.tensor(X_masked.mean(0), device=DEVICE)
    std = torch.tensor(X_masked.std(0) + 1e-8, device=DEVICE)

    # ---- 2) モデル準備 ----
    model = DiffusionTransformer().to(DEVICE)
    ddpm = DDPM(DIFF_STEPS)
    model = train_or_load_model(model, ddpm, x_path, m_path, model_save_path, mean, std)

    # ---- 3) 生成（2000人） ----
    print(f"Generating {TARGET_NUM} samples (batch={GEN_BATCH})...")
    generated_list = []
    num_generated = 0
    while num_generated < TARGET_NUM:
        current_batch = min(GEN_BATCH, TARGET_NUM - num_generated)
        batch_samples = sample_ddim(model, ddpm, current_batch, mean, std)
        generated_list.append(batch_samples)
        num_generated += current_batch
        print(f"Progress: {num_generated} / {TARGET_NUM}")

    raw_samples = np.concatenate(generated_list, axis=0)  # (N,48,10)
    print("Generation Complete. Shape:", raw_samples.shape)

    # ---- 4) 平滑化 ----
    print("Applying Vector Smoothing (Window=5)...")
    smooth_samples = smooth_vectors(raw_samples, window_size=5)

    # ---- 5) デコード ----
    print("Decoding to Mesh Codes...")
    final_codes = decode_to_mesh_codes(smooth_samples, dict_path, decode_batch=2000)

    # 保存（CSV）
    save_path = os.path.join(out_dir, f"final_trips_{tag}_{TARGET_NUM}.csv")
    pd.DataFrame(final_codes).to_csv(save_path, index=False)
    print(f"Saved {TARGET_NUM} trips to {save_path}")

    # 生成テンソルも保存（必要なら）
    np.save(os.path.join(out_dir, f"generated_raw_{tag}_{TARGET_NUM}.npy"), raw_samples.astype(np.float32))
    np.save(os.path.join(out_dir, f"generated_smooth_{tag}_{TARGET_NUM}.npy"), smooth_samples.astype(np.float32))
    print("Saved generated npy (raw & smooth).")

# =========================
# Main
# =========================
def main():
    # 入力
    configs = [
        {
            "tag": "KL",
            "x_path": r"trip_outputs\trip_outputs_KL\X_trip_48x10.npy",
            "m_path": r"trip_outputs\trip_outputs_KL\mask_48.npy",
            "dict_path": r"C:/anaconda/Lib/site-packages/pandas/io/formats/mesh4_to_z10_KL.csv",
            "out_dir": r"diffusion_final_KL",
        },
        {
            "tag": "MSE",
            "x_path": r"trip_outputs\trip_outputs_MSE\X_trip_48x10.npy",
            "m_path": r"trip_outputs\trip_outputs_MSE\mask_48.npy",
            "dict_path": r"C:/anaconda/Lib/site-packages/pandas/io/formats/mesh4_to_z10_MSE.csv",
            "out_dir": r"diffusion_final_MSE",
        }
    ]

    for cfg in configs:
        run_one(**cfg)

    print("\n=== ALL DONE ===")

if __name__ == "__main__":
    main()

