
AE (MSE-only for pop/poi/coord)
- Preprocess: log1p + normalization (same as KL-only)
- Loss:
    Pop_MSE, POI_MSE, Coord_MSE (ONLY)
- Output:
    ae_MSE_only_model.pth
    loss_history_AE_MSE_only.csv
    AE_MSE_only_scales_and_gnorms.csv (optional if grad-balance enabled)
    mesh4_to_z10_MSE.csv

Author: 難波祐樹
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# ==========================================
# 1. 設定
# ==========================================
input_csv_path = r"C:\卒論各データ\元データ\4mesh\processed_data_alphabet_split.csv"

processed_csv_path = "processed_data_ready_AE_MSE_only.csv"
model_save_path = "ae_MSE_only_model.pth"
history_save_path = "loss_history_AE_MSE_only.csv"
aux_save_path = "AE_MSE_only_scales_and_gnorms.csv"
figure_save_path = "all_loss_curves_AE_MSE_only.png"
dict_save_path = "mesh4_to_z10_MSE.csv"

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
print("DEVICE =", DEVICE)

# ==========================================
# 2. 前処理（あなたのコード踏襲）
# ==========================================
def preprocess_data(input_path, output_path):
    print(f"Loading data from: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    id_cols = ['4mesh code']
    coord_cols = ['x_diff', 'y_diff']
    pop_cols = [str(x) for x in range(0, 2400, 100)]  # 0..2300 (24列)
    poi_cols = [c for c in df.columns if c not in id_cols + coord_cols + pop_cols]

    df_log = df.copy()

    # 数値化（欠損は0）
    numeric_cols = pop_cols + poi_cols
    df_log[numeric_cols] = df_log[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    # log1p
    df_log[pop_cols] = np.log1p(df_log[pop_cols])
    df_log[poi_cols] = np.log1p(df_log[poi_cols])

    # pop: 全体最大で割る
    pop_max = df_log[pop_cols].max().max()
    if pop_max == 0:
        pop_max = 1.0
    df_log[pop_cols] = df_log[pop_cols] / pop_max

    # poi: 列最大で割る
    for col in poi_cols:
        m = df_log[col].max()
        if m == 0:
            m = 1.0
        df_log[col] = df_log[col] / m

    # coord: 入力元 df の最大で割る（元コード踏襲）
    coord_max = max(df['x_diff'].max(), df['y_diff'].max())
    if coord_max == 0:
        coord_max = 1.0
    df_log['x_diff'] = df['x_diff'] / coord_max
    df_log['y_diff'] = df['y_diff'] / coord_max

    df_log.to_csv(output_path, index=False)
    print(f"Saved processed CSV: {output_path}")
    return pop_cols, poi_cols, coord_cols


# ==========================================
# 3. Dataset / Model（AE）
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
    """MSE-onlyなので出力は [0,1] に収める sigmoid でOK（元コード踏襲）"""
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
# 4. 損失（MSE-only）
# ==========================================
class Loss_MSE_only(nn.Module):
    def forward(self, recon_x, x, slices):
        x_coord = x[:, slices['coord']]
        r_coord = recon_x[:, slices['coord']]
        x_pop = x[:, slices['pop']]
        r_pop = recon_x[:, slices['pop']]
        x_poi = x[:, slices['poi']]
        r_poi = recon_x[:, slices['poi']]

        return {
            "Pop_MSE": F.mse_loss(r_pop, x_pop),
            "POI_MSE": F.mse_loss(r_poi, x_poi),
            "Coord_MSE": F.mse_loss(r_coord, x_coord),
        }


# ==========================================
# 5. 勾配ノルム揃え（任意）
# ==========================================
def grad_balance_total(losses_dict, params_for_balance, eps=1e-8, clamp=(0.2, 5.0)):
    names = list(losses_dict.keys())

    g_list = []
    for n in names:
        grads = torch.autograd.grad(
            losses_dict[n],
            params_for_balance,
            retain_graph=True,
            create_graph=False,
            allow_unused=True
        )
        norm_sq = 0.0
        for g in grads:
            if g is None:
                continue
            norm_sq = norm_sq + torch.sum(g.detach() ** 2)
        g = torch.sqrt(norm_sq + eps)
        g_list.append(g)

    g_stack = torch.stack(g_list)
    g_bar = torch.mean(g_stack)
    scales = (g_bar / (g_stack + eps)).clamp(clamp[0], clamp[1])

    total = 0.0
    scaled_terms = {}
    grad_norms = {}
    for i, n in enumerate(names):
        total = total + scales[i] * losses_dict[n]
        scaled_terms[n] = float(scales[i].item())
        grad_norms[n] = float(g_stack[i].item())

    return total, scaled_terms, grad_norms


# ==========================================
# 6. 学習・評価
# ==========================================
def run_epoch(model, loader, optimizer, loss_fn, slices, params_for_balance=None, train=True):
    if train:
        model.train()
    else:
        model.eval()

    sum_losses = {}
    sum_total = 0.0
    sum_scales = {}
    sum_gnorms = {}

    for batch in loader:
        batch = batch.to(DEVICE)

        if train:
            optimizer.zero_grad()

        recon, _ = model(batch)
        losses = loss_fn(recon, batch, slices)

        if USE_GRAD_BALANCE:
            total, scales, gnorms = grad_balance_total(
                losses_dict=losses,
                params_for_balance=params_for_balance,
                clamp=GRAD_BALANCE_CLAMP
            )
        else:
            total = sum(losses.values())
            scales = {k: 1.0 for k in losses.keys()}
            gnorms = {k: 0.0 for k in losses.keys()}

        if train:
            total.backward()
            optimizer.step()

        sum_total += float(total.item())
        for k, v in losses.items():
            sum_losses[k] = sum_losses.get(k, 0.0) + float(v.item())
        for k, v in scales.items():
            sum_scales[k] = sum_scales.get(k, 0.0) + float(v)
        for k, v in gnorms.items():
            sum_gnorms[k] = sum_gnorms.get(k, 0.0) + float(v)

    n = len(loader)
    avg_losses = {k: v / n for k, v in sum_losses.items()}
    avg_scales = {k: v / n for k, v in sum_scales.items()}
    avg_gnorms = {k: v / n for k, v in sum_gnorms.items()}
    avg_total = sum_total / n

    return avg_total, avg_losses, avg_scales, avg_gnorms


def main():
    pop_cols, poi_cols, coord_cols = preprocess_data(input_csv_path, processed_csv_path)

    full_dataset = MeshDataset(processed_csv_path, pop_cols, poi_cols, coord_cols)
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    input_dim = len(full_dataset.feature_cols)
    model = AE(input_dim=input_dim, latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = Loss_MSE_only()

    params_for_balance = [model.fc1.weight, model.fc1.bias, model.fc2.weight, model.fc2.bias]

    loss_names = ["Pop_MSE", "POI_MSE", "Coord_MSE"]
    history = {k: {"train": [], "val": []} for k in (loss_names + ["Total_Loss"])}
    scale_hist = {k: {"train": [], "val": []} for k in loss_names}
    gnorm_hist = {k: {"train": [], "val": []} for k in loss_names}

    print("=== Train: AE MSE-only (pop/poi/coord MSE) ===")

    for epoch in range(EPOCHS):
        tr_total, tr_losses, tr_scales, tr_gnorms = run_epoch(
            model, train_loader, optimizer, loss_fn, full_dataset.slices,
            params_for_balance=params_for_balance, train=True
        )
        va_total, va_losses, va_scales, va_gnorms = run_epoch(
            model, val_loader, optimizer, loss_fn, full_dataset.slices,
            params_for_balance=params_for_balance, train=False
        )

        history["Total_Loss"]["train"].append(tr_total)
        history["Total_Loss"]["val"].append(va_total)

        for k in loss_names:
            history[k]["train"].append(tr_losses.get(k, 0.0))
            history[k]["val"].append(va_losses.get(k, 0.0))
            scale_hist[k]["train"].append(tr_scales.get(k, 1.0))
            scale_hist[k]["val"].append(va_scales.get(k, 1.0))
            gnorm_hist[k]["train"].append(tr_gnorms.get(k, 0.0))
            gnorm_hist[k]["val"].append(va_gnorms.get(k, 0.0))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[Epoch {epoch+1:3d}] train_total={tr_total:.6f}  val_total={va_total:.6f} "
                  f"| Pop_MSE={tr_losses['Pop_MSE']:.6f}, POI_MSE={tr_losses['POI_MSE']:.6f}, Coord_MSE={tr_losses['Coord_MSE']:.6f}")

    # 保存
    torch.save(model.state_dict(), model_save_path)
    print(f"Saved model: {model_save_path}")

    # history保存
    df_train = pd.DataFrame({k: history[k]["train"] for k in history.keys()})
    df_val = pd.DataFrame({k: history[k]["val"] for k in history.keys()})
    df_all = pd.concat([df_train.add_prefix("train_"), df_val.add_prefix("val_")], axis=1)
    df_all.index.name = "epoch"
    df_all.to_csv(history_save_path, index=True)
    print(f"Saved loss history: {history_save_path}")

    # aux保存
    df_sc_train = pd.DataFrame({k: scale_hist[k]["train"] for k in loss_names}).add_prefix("train_scale_")
    df_sc_val = pd.DataFrame({k: scale_hist[k]["val"] for k in loss_names}).add_prefix("val_scale_")
    df_gn_train = pd.DataFrame({k: gnorm_hist[k]["train"] for k in loss_names}).add_prefix("train_gnorm_")
    df_gn_val = pd.DataFrame({k: gnorm_hist[k]["val"] for k in loss_names}).add_prefix("val_gnorm_")
    df_aux = pd.concat([df_sc_train, df_sc_val, df_gn_train, df_gn_val], axis=1)
    df_aux.index.name = "epoch"
    df_aux.to_csv(aux_save_path, index=True)
    print(f"Saved: {aux_save_path}")

    # loss曲線
    plot_keys = ["Pop_MSE", "POI_MSE", "Coord_MSE", "Total_Loss"]
    plt.figure()
    for k in plot_keys:
        plt.plot(history[k]["train"], label=f"train_{k}")
        plt.plot(history[k]["val"], linestyle="--", label=f"val_{k}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_save_path, dpi=200)
    print(f"Saved loss curves: {figure_save_path}")
    plt.show()

    # mesh4 -> latent(z) 保存
    model.eval()
    mesh4 = full_dataset.df["4mesh code"].astype(str).values
    X_all = torch.tensor(full_dataset.df[full_dataset.feature_cols].values, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        z_all = model.encode(X_all).cpu().numpy()

    latent_df = pd.DataFrame({"mesh4": mesh4})
    for i in range(LATENT_DIM):
        latent_df[f"z{i+1}"] = z_all[:, i]
    latent_df.to_csv(dict_save_path, index=False)
    print(f"Saved dictionary: {dict_save_path}")


if __name__ == "__main__":
    main()
