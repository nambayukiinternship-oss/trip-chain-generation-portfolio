import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


# 1. 実行設定・パス構成

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ポートフォリオ用の相対パス設計
INPUT_CSV_PATH = r"./data/input/processed_data_alphabet_split.csv"
OUT_DIR = r"./outputs/autoencoder"
os.makedirs(OUT_DIR, exist_ok=True)

PROCESSED_CSV_PATH = os.path.join(OUT_DIR, f"processed_data_ready_AE_MSE_{timestamp}.csv")
MODEL_SAVE_PATH = os.path.join(OUT_DIR, f"ae_MSE_only_model_{timestamp}.pth")
HISTORY_SAVE_PATH = os.path.join(OUT_DIR, f"loss_history_AE_MSE_only_{timestamp}.csv")
DICT_SAVE_PATH = os.path.join(OUT_DIR, f"mesh4_to_z10_MSE_{timestamp}.csv")

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.2
LATENT_DIM = 10
SEED = 42

USE_GRAD_BALANCE = False
GRAD_BALANCE_CLAMP = (0.2, 5.0)

np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. データ前処理
# ==========================================
def preprocess_data(input_path, output_path):
    print(f"Loading data from: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}\n"
                                "Please ensure the dummy/sample data is placed in the './data/input/' directory.")

    df = pd.read_csv(input_path)

    id_cols = ['4mesh code']
    coord_cols = ['x_diff', 'y_diff']
    pop_cols = [str(x) for x in range(0, 2400, 100)]  # 0..2300 (24列)
    poi_cols = [c for c in df.columns if c not in id_cols + coord_cols + pop_cols]

    df_log = df.copy()

    # 数値化（欠損は0）
    numeric_cols = pop_cols + poi_cols
    df_log[numeric_cols] = df_log[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    # log1p 変換
    df_log[pop_cols] = np.log1p(df_log[pop_cols])
    df_log[poi_cols] = np.log1p(df_log[poi_cols])

    # pop: 全体最大で割る
    pop_max = df_log[pop_cols].max().max()
    df_log[pop_cols] = df_log[pop_cols] / (pop_max if pop_max != 0 else 1.0)

    # poi: 列最大で割る
    for col in poi_cols:
        m = df_log[col].max()
        df_log[col] = df_log[col] / (m if m != 0 else 1.0)

    # coord: 座標系の最大値で割る
    coord_max = max(df['x_diff'].max(), df['y_diff'].max())
    df_log['x_diff'] = df['x_diff'] / (coord_max if coord_max != 0 else 1.0)
    df_log['y_diff'] = df['y_diff'] / (coord_max if coord_max != 0 else 1.0)

    df_log.to_csv(output_path, index=False)
    print(f"Saved processed CSV: {output_path}")
    return pop_cols, poi_cols, coord_cols

# ==========================================
# 3. データセットとモデル定義 (AutoEncoder)
# ==========================================
class MeshDataset(Dataset):
    def __init__(self, csv_file, pop_cols, poi_cols, coord_cols):
        self.df = pd.read_csv(csv_file)
        self.feature_cols = coord_cols + pop_cols + poi_cols
        self.data = torch.tensor(self.df[self.feature_cols].values, dtype=torch.float32)

        self.slices = {
            'coord': slice(0, len(coord_cols)),
            'pop': slice(len(coord_cols), len(coord_cols) + len(pop_cols)),
            'poi': slice(len(coord_cols) + len(pop_cols), None)
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class AE(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_z = nn.Linear(64, latent_dim)

        self.fc3 = nn.Linear(latent_dim, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(128, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc_z(h)

    def decode(self, z):
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        return torch.sigmoid(self.fc5(h))

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

# ==========================================
# 4. 損失関数と勾配調整
# ==========================================
class Loss_MSE_only(nn.Module):
    def forward(self, recon_x, x, slices):
        return {
            "Pop_MSE": F.mse_loss(recon_x[:, slices['pop']], x[:, slices['pop']]),
            "POI_MSE": F.mse_loss(recon_x[:, slices['poi']], x[:, slices['poi']]),
            "Coord_MSE": F.mse_loss(recon_x[:, slices['coord']], x[:, slices['coord']]),
        }

def grad_balance_total(losses_dict, params_for_balance, eps=1e-8, clamp=(0.2, 5.0)):
    names = list(losses_dict.keys())
    g_list = []
    
    for n in names:
        grads = torch.autograd.grad(
            losses_dict[n], params_for_balance,
            retain_graph=True, create_graph=False, allow_unused=True
        )
        norm_sq = sum(torch.sum(g.detach() ** 2) for g in grads if g is not None)
        g_list.append(torch.sqrt(norm_sq + eps))

    g_stack = torch.stack(g_list)
    g_bar = torch.mean(g_stack)
    scales = (g_bar / (g_stack + eps)).clamp(clamp[0], clamp[1])

    total = 0.0
    scaled_terms, grad_norms = {}, {}
    for i, n in enumerate(names):
        total += scales[i] * losses_dict[n]
        scaled_terms[n] = float(scales[i].item())
        grad_norms[n] = float(g_stack[i].item())

    return total, scaled_terms, grad_norms


# 5. 学習ループ

def run_epoch(model, loader, optimizer, loss_fn, slices, params_for_balance=None, train=True):
    model.train() if train else model.eval()
    sum_losses, sum_scales, sum_gnorms = {}, {}, {}
    sum_total = 0.0

    for batch in loader:
        batch = batch.to(DEVICE)
        if train:
            optimizer.zero_grad()

        recon, _ = model(batch)
        losses = loss_fn(recon, batch, slices)

        if USE_GRAD_BALANCE:
            total, scales, gnorms = grad_balance_total(
                losses_dict=losses, params_for_balance=params_for_balance, clamp=GRAD_BALANCE_CLAMP
            )
        else:
            total = sum(losses.values())
            scales = {k: 1.0 for k in losses.keys()}
            gnorms = {k: 0.0 for k in losses.keys()}

        if train:
            total.backward()
            optimizer.step()

        sum_total += float(total.item())
        for k in losses.keys():
            sum_losses[k] = sum_losses.get(k, 0.0) + float(losses[k].item())
            sum_scales[k] = sum_scales.get(k, 0.0) + float(scales[k])
            sum_gnorms[k] = sum_gnorms.get(k, 0.0) + float(gnorms[k])

    n = len(loader)
    return (
        sum_total / n,
        {k: v / n for k, v in sum_losses.items()},
        {k: v / n for k, v in sum_scales.items()},
        {k: v / n for k, v in sum_gnorms.items()}
    )

def main():
    print(f"DEVICE = {DEVICE}")
    pop_cols, poi_cols, coord_cols = preprocess_data(INPUT_CSV_PATH, PROCESSED_CSV_PATH)

    full_dataset = MeshDataset(PROCESSED_CSV_PATH, pop_cols, poi_cols, coord_cols)
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_dataset, val_dataset = random_split(full_dataset, [len(full_dataset) - val_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = AE(input_dim=len(full_dataset.feature_cols), latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = Loss_MSE_only()
    params_for_balance = [model.fc1.weight, model.fc1.bias, model.fc2.weight, model.fc2.bias]

    loss_names = ["Pop_MSE", "POI_MSE", "Coord_MSE"]
    history = {k: {"train": [], "val": []} for k in (loss_names + ["Total_Loss"])}

    print("=== Training Started: AutoEncoder (MSE-only) ===")
    for epoch in range(EPOCHS):
        tr_total, tr_losses, _, _ = run_epoch(model, train_loader, optimizer, loss_fn, full_dataset.slices, params_for_balance, train=True)
        va_total, va_losses, _, _ = run_epoch(model, val_loader, optimizer, loss_fn, full_dataset.slices, params_for_balance, train=False)

        history["Total_Loss"]["train"].append(tr_total)
        history["Total_Loss"]["val"].append(va_total)
        for k in loss_names:
            history[k]["train"].append(tr_losses.get(k, 0.0))
            history[k]["val"].append(va_losses.get(k, 0.0))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[Epoch {epoch+1:3d}] train_total={tr_total:.6f} | val_total={va_total:.6f}")

    # モデル・履歴の保存
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    df_train = pd.DataFrame({k: history[k]["train"] for k in history.keys()}).add_prefix("train_")
    df_val = pd.DataFrame({k: history[k]["val"] for k in history.keys()}).add_prefix("val_")
    pd.concat([df_train, df_val], axis=1).to_csv(HISTORY_SAVE_PATH, index_label="epoch")

    # 潜在変数の保存 (Z次元マッピング)
    model.eval()
    mesh4 = full_dataset.df["4mesh code"].astype(str).values
    X_all = torch.tensor(full_dataset.df[full_dataset.feature_cols].values, dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        z_all = model.encode(X_all).cpu().numpy()

    latent_df = pd.DataFrame({"mesh4": mesh4})
    for i in range(LATENT_DIM):
        latent_df[f"z{i+1}"] = z_all[:, i]
    latent_df.to_csv(DICT_SAVE_PATH, index=False)
    print("=== Training Complete and Assets Saved ===")

if __name__ == "__main__":
    main()
