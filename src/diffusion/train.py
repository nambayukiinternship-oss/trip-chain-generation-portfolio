# ============================================================
# Diffusion Model for Trip Chain Generation
#
# メモリ安全版：
#   ・モデル構造は変更しない
#   ・学習データは変更しない
#   ・損失関数は変更しない
#   ・diff_steps / ddim_steps は変更しない
#   ・Datasetで全データをtorch.tensor化しない
#   ・生成は2000件を小分けに実行する
#   ・CSVも小分けに追記保存する
#
# 入力：
#   C:\卒論データ\diffusion_input_mesh_filled\X_trip_96x10_mesh_filled.npy
#
# 出力：
#   diffusion_model_96.pth
#   training_history.csv
#   z_mean.npy
#   z_std.npy
#   generated_z_96.npy
#   generated_z_96.csv
# ============================================================

import os
import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. 設定
# ============================================================

input_x_path = r"C:\卒論データ\diffusion_input_mesh_filled\X_trip_96x10_mesh_filled.npy"

output_dir = r"C:\卒論データ\diffusion_model_96"
os.makedirs(output_dir, exist_ok=True)

seq_len = 96          # 15分単位の1日
feature_dim = 10      # AE潜在ベクトル z1〜z10

batch_size = 128
epochs = 100
learning_rate = 1e-4

diff_steps = 1000     # DDPMの拡散ステップ数
ddim_steps = 100      # 生成時のDDIMステップ数

d_model = 128
nhead = 4
num_layers = 4
dropout = 0.1

# 2000件生成
num_generate = 2000

# 一度に生成する件数
# ここは生成品質には影響しない
# メモリが不安なら 50，余裕があれば 100〜200
generate_batch_size = 100

random_seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用デバイス:", device)

np.random.seed(random_seed)
torch.manual_seed(random_seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)


# ============================================================
# 2. データ読み込み
# ============================================================

# ここは元コードと同じく全体を読み込む
# ただし float32 にしてメモリ使用量を抑える
X = np.load(input_x_path).astype(np.float32)

print("読み込み完了")
print("X shape:", X.shape)

if X.ndim != 3:
    raise ValueError("X は [N, 96, 10] の3次元配列である必要があります。")

if X.shape[1] != seq_len:
    raise ValueError(f"時系列長が {seq_len} ではありません。現在: {X.shape[1]}")

if X.shape[2] != feature_dim:
    raise ValueError(f"特徴量次元が {feature_dim} ではありません。現在: {X.shape[2]}")


# ============================================================
# 3. 標準化
# ============================================================

# Diffusionは連続値を扱うため，
# z1〜z10を平均0・標準偏差1にすると学習が安定しやすい

mean = X.reshape(-1, feature_dim).mean(axis=0).astype(np.float32)
std = X.reshape(-1, feature_dim).std(axis=0).astype(np.float32)

std[std == 0] = 1.0

X_norm = ((X - mean) / std).astype(np.float32)

mean_path = os.path.join(output_dir, "z_mean.npy")
std_path = os.path.join(output_dir, "z_std.npy")

np.save(mean_path, mean)
np.save(std_path, std)

print("標準化完了")
print("mean:", mean)
print("std:", std)


# ============================================================
# 4. Dataset
#    メモリ安全版：
#    Dataset作成時に torch.tensor(X_norm) を作らない
# ============================================================

class TripDataset(Dataset):
    def __init__(self, X_norm):
        self.X_norm = X_norm

    def __len__(self):
        return len(self.X_norm)

    def __getitem__(self, idx):
        # 必要な1サンプルだけTensor化する
        x = self.X_norm[idx]
        return torch.from_numpy(x.astype(np.float32))


dataset = TripDataset(X_norm)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print("データセット数:", len(dataset))


# ============================================================
# 5. Diffusionスケジュール
# ============================================================

def make_beta_schedule(diff_steps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, diff_steps)


betas = make_beta_schedule(diff_steps).to(device)
alphas = 1.0 - betas
alphas_bar = torch.cumprod(alphas, dim=0)


def q_sample(x0, t, noise):
    """
    x0    : [B, T, D]
    t     : [B]
    noise : [B, T, D]

    x0に時刻tのノイズを加えたx_tを作る
    """

    sqrt_ab = torch.sqrt(alphas_bar[t]).view(-1, 1, 1)
    sqrt_omb = torch.sqrt(1.0 - alphas_bar[t]).view(-1, 1, 1)

    x_t = sqrt_ab * x0 + sqrt_omb * noise

    return x_t


# ============================================================
# 6. Diffusion timestep 埋め込み
# ============================================================

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: [B]
        """

        half_dim = self.dim // 2

        emb_scale = math.log(10000) / (half_dim - 1)

        emb = torch.exp(
            torch.arange(half_dim, device=t.device) * -emb_scale
        )

        emb = t[:, None].float() * emb[None, :]

        emb = torch.cat(
            [torch.sin(emb), torch.cos(emb)],
            dim=1
        )

        return emb


# ============================================================
# 7. Transformer型 Diffusion Model
# ============================================================

class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        feature_dim=10,
        seq_len=96,
        d_model=128,
        nhead=4,
        num_layers=4,
        dropout=0.1
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.d_model = d_model

        # z10 → d_model
        self.input_proj = nn.Linear(feature_dim, d_model)

        # 96スロットの位置埋め込み
        self.pos_emb = nn.Parameter(
            torch.randn(1, seq_len, d_model) * 0.02
        )

        # diffusion step t の埋め込み
        self.time_emb = SinusoidalTimeEmbedding(d_model)

        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # d_model → z10 のノイズ予測
        self.output_proj = nn.Linear(d_model, feature_dim)

    def forward(self, x_t, t):
        """
        x_t : [B, T, D]
        t   : [B]
        """

        h = self.input_proj(x_t)

        h = h + self.pos_emb[:, :h.size(1), :]

        t_emb = self.time_emb(t)
        t_emb = self.time_mlp(t_emb)

        h = h + t_emb[:, None, :]

        h = self.transformer(h)

        pred_noise = self.output_proj(h)

        return pred_noise


model = DiffusionTransformer(
    feature_dim=feature_dim,
    seq_len=seq_len,
    d_model=d_model,
    nhead=nhead,
    num_layers=num_layers,
    dropout=dropout
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=1e-4
)

print("\nモデル構造:")
print(model)


# ============================================================
# 8. 学習
# ============================================================

model_path = os.path.join(output_dir, "diffusion_model_96.pth")

history = []

for epoch in range(1, epochs + 1):

    model.train()

    total_loss = 0.0
    total_count = 0

    for batch_idx, x0 in enumerate(loader):

        x0 = x0.to(device, non_blocking=True)

        B = x0.size(0)

        # 各サンプルに対してランダムな拡散時刻tを選ぶ
        t = torch.randint(
            0,
            diff_steps,
            (B,),
            device=device
        ).long()

        # 真のノイズ
        noise = torch.randn_like(x0)

        # x0にノイズを加えてx_tを作る
        x_t = q_sample(x0, t, noise)

        # モデルがノイズを予測
        pred_noise = model(x_t, t)

        # 真のノイズと予測ノイズのMSE
        loss = ((pred_noise - noise) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item() * B
        total_count += B

    avg_loss = total_loss / total_count

    history.append({
        "epoch": epoch,
        "loss": avg_loss
    })

    if epoch == 1 or epoch % 10 == 0:
        print(f"Epoch [{epoch:03d}/{epochs}] Loss: {avg_loss:.6f}")

    if epoch % 10 == 0:
        torch.save(model.state_dict(), model_path)

# 最終モデル保存
torch.save(model.state_dict(), model_path)

history_df = pd.DataFrame(history)

history_path = os.path.join(output_dir, "training_history.csv")

history_df.to_csv(
    history_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n学習完了")
print("モデル保存:", model_path)
print("学習履歴保存:", history_path)


# ============================================================
# 9. DDIM Sampling
# ============================================================

@torch.no_grad()
def ddim_sample(
    model,
    num_samples,
    seq_len,
    feature_dim,
    ddim_steps=100
):
    model.eval()

    # ランダムノイズから開始
    x = torch.randn(
        num_samples,
        seq_len,
        feature_dim,
        device=device
    )

    # diff_stepsからddim_stepsへ間引く
    step_indices = torch.linspace(
        diff_steps - 1,
        0,
        ddim_steps,
        device=device
    ).long()

    for i in range(len(step_indices) - 1):

        t = step_indices[i].repeat(num_samples)
        t_next = step_indices[i + 1].repeat(num_samples)

        pred_noise = model(x, t)

        alpha_bar_t = alphas_bar[t].view(-1, 1, 1)
        alpha_bar_next = alphas_bar[t_next].view(-1, 1, 1)

        # x0を予測
        x0_pred = (
            x - torch.sqrt(1.0 - alpha_bar_t) * pred_noise
        ) / torch.sqrt(alpha_bar_t)

        # DDIM eta=0
        x = (
            torch.sqrt(alpha_bar_next) * x0_pred
            + torch.sqrt(1.0 - alpha_bar_next) * pred_noise
        )

    return x.cpu().numpy()


# ============================================================
# 10. 生成
#     メモリ安全版：
#     2000件を一気に生成せず，小分けに生成する
# ============================================================

generated_csv_path = os.path.join(output_dir, "generated_z_96.csv")
generated_path = os.path.join(output_dir, "generated_z_96.npy")

generated_list = []

first_write = True

print("\n生成開始")
print("num_generate:", num_generate)
print("generate_batch_size:", generate_batch_size)

for start in range(0, num_generate, generate_batch_size):

    end = min(start + generate_batch_size, num_generate)
    current_n = end - start

    print(f"生成中: sample_id {start} ～ {end - 1}")

    generated_norm_batch = ddim_sample(
        model=model,
        num_samples=current_n,
        seq_len=seq_len,
        feature_dim=feature_dim,
        ddim_steps=ddim_steps
    )

    print("generated_norm_batch shape:", generated_norm_batch.shape)

    # 標準化を元に戻す
    generated_batch = generated_norm_batch * mean.reshape(1, 1, -1) * 0.0 + generated_norm_batch * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
    generated_batch = generated_batch.astype(np.float32)

    generated_list.append(generated_batch)

    # CSVを小分けに作成
    rows = []

    for local_sample_id in range(current_n):

        sample_id = start + local_sample_id

        for slot_idx in range(seq_len):

            hour = slot_idx // 4
            minute = (slot_idx % 4) * 15

            row = {
                "sample_id": sample_id,
                "slot_idx": slot_idx,
                "time": f"{hour:02d}:{minute:02d}",
                "ae_time": hour * 100
            }

            for j in range(feature_dim):
                row[f"z{j + 1}"] = generated_batch[local_sample_id, slot_idx, j]

            rows.append(row)

    batch_df = pd.DataFrame(rows)

    batch_df.to_csv(
        generated_csv_path,
        mode="w" if first_write else "a",
        header=first_write,
        index=False,
        encoding="utf-8-sig"
    )

    first_write = False

    print(f"CSV追記完了: sample_id {start} ～ {end - 1}")

    # GPUメモリを少し解放
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 11. npy保存
# ============================================================

generated = np.concatenate(generated_list, axis=0)

np.save(generated_path, generated)

print("\n生成npy保存:")
print(generated_path)

print("\n生成CSV保存:")
print(generated_csv_path)

print("\ngenerated shape:")
print(generated.shape)


# ============================================================
# 12. 確認
# ============================================================

print("\n生成結果の確認")
print("generated shape:", generated.shape)

print("\n生成zの統計:")

# 確認用にCSV全体を読み込まず，npyから統計を計算
flat_generated = generated.reshape(-1, feature_dim)

stats_df = pd.DataFrame(
    flat_generated,
    columns=[f"z{i}" for i in range(1, 11)]
).describe()

print(stats_df)

stats_path = os.path.join(output_dir, "generated_z_96_stats.csv")

stats_df.to_csv(
    stats_path,
    encoding="utf-8-sig"
)

print("\n生成統計保存:")
print(stats_path)

print("\n完了")
