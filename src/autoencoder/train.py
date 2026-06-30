# ============================================================
# 4次メッシュ × 時刻 の AutoEncoder
# 入力：
#   x_diff, y_diff
#   population_at_time
#   sin_time, cos_time
#   産業・施設特徴量
#
# 出力：
#   mesh_time_id ごとの10次元潜在ベクトル
# ============================================================

import os
import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# ============================================================
# 1. 設定
# ============================================================

file_path = r"C:\卒論データ\processed_data_alphabet_split.csv"

output_dir = r"C:\卒論データ\ae_mesh_time_embedding"
os.makedirs(output_dir, exist_ok=True)

latent_dim = 10
batch_size = 256
epochs = 100
learning_rate = 1e-3
random_seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用デバイス:", device)

np.random.seed(random_seed)
torch.manual_seed(random_seed)


# ============================================================
# 2. CSV読み込み
# ============================================================

df = pd.read_csv(file_path)

print("元データ形状:", df.shape)
print("列名:")
print(df.columns.tolist())


# ============================================================
# 3. 時刻列を取得
# ============================================================

# 0, 100, 200, ..., 2300 の列を自動取得
time_cols = []

for col in df.columns:
    col_str = str(col)

    if col_str.isdigit():
        v = int(col_str)

        if 0 <= v <= 2300:
            time_cols.append(col)

time_cols = sorted(time_cols, key=lambda x: int(x))

print("時刻列:")
print(time_cols)


# ============================================================
# 4. 列の整理
# ============================================================

mesh_col = "4mesh code"
coord_cols = ["x_diff", "y_diff"]

# mesh_col, 時刻列, 座標列以外を産業・施設特徴量とする
exclude_cols = [mesh_col] + time_cols + coord_cols

poi_cols = [col for col in df.columns if col not in exclude_cols]

print("産業・施設特徴量の列数:", len(poi_cols))
print("産業・施設特徴量:")
print(poi_cols)


# ============================================================
# 5. 横持ち人口データを縦持ちに変換
# ============================================================

long_df = df.melt(
    id_vars=[mesh_col] + coord_cols + poi_cols,
    value_vars=time_cols,
    var_name="time",
    value_name="population"
)

long_df["time"] = long_df["time"].astype(int)

# mesh_time_id を作成
# 例：483074102_0800
long_df["mesh_time_id"] = (
    long_df[mesh_col].astype(str)
    + "_"
    + long_df["time"].astype(str).str.zfill(4)
)

print("時刻展開後の形状:", long_df.shape)
print(long_df.head())


# ============================================================
# 6. 時刻特徴量を作成
# ============================================================

# 0, 100, 200, ... を hour に変換
long_df["hour"] = long_df["time"] // 100

# sin_time, cos_time を作成
# -1〜1 ではなく，0〜1に変換してAEに入れる
long_df["sin_time"] = (
    np.sin(2 * np.pi * long_df["hour"] / 24) + 1
) / 2

long_df["cos_time"] = (
    np.cos(2 * np.pi * long_df["hour"] / 24) + 1
) / 2


# ============================================================
# 7. 入力特徴量の前処理
# ============================================================

# 人口は偏りが大きいので log1p を使う
long_df["population_log"] = np.log1p(long_df["population"])

# 産業・施設特徴量にも log1p を使う
for col in poi_cols:
    long_df[col] = np.log1p(long_df[col])

# AutoEncoderへの入力特徴量
feature_cols = (
    coord_cols
    + ["population_log", "sin_time", "cos_time"]
    + poi_cols
)

print("入力特徴量:")
print(feature_cols)

X_raw = long_df[feature_cols].values.astype(np.float32)

print("入力データ shape:", X_raw.shape)


# ============================================================
# 8. 0〜1に正規化
# ============================================================

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_raw).astype(np.float32)

# scalerを保存
scaler_path = os.path.join(output_dir, "scaler.pkl")
joblib.dump(scaler, scaler_path)

# 使用した特徴量列を保存
feature_cols_path = os.path.join(output_dir, "feature_cols.txt")

with open(feature_cols_path, "w", encoding="utf-8") as f:
    for col in feature_cols:
        f.write(str(col) + "\n")

print("正規化後 shape:", X_scaled.shape)


# ============================================================
# 9. 学習データ・検証データに分割
# ============================================================

train_X, val_X = train_test_split(
    X_scaled,
    test_size=0.2,
    random_state=random_seed,
    shuffle=True
)


class MeshTimeDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


train_dataset = MeshTimeDataset(train_X)
val_dataset = MeshTimeDataset(val_X)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)


# ============================================================
# 10. AutoEncoder モデル定義
# ============================================================

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.ReLU(),

            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


input_dim = X_scaled.shape[1]

model = AutoEncoder(
    input_dim=input_dim,
    latent_dim=latent_dim
).to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)

print(model)


# ============================================================
# 11. 学習
# ============================================================

best_val_loss = float("inf")
best_model_path = os.path.join(output_dir, "best_autoencoder.pth")

history = []

for epoch in range(1, epochs + 1):

    # -----------------------------
    # train
    # -----------------------------
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        recon, z = model(batch)

        loss = criterion(recon, batch)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch.size(0)

    train_loss /= len(train_loader.dataset)

    # -----------------------------
    # validation
    # -----------------------------
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            recon, z = model(batch)

            loss = criterion(recon, batch)

            val_loss += loss.item() * batch.size(0)

    val_loss /= len(val_loader.dataset)

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss
    })

    # 最良モデルを保存
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)

    if epoch == 1 or epoch % 10 == 0:
        print(
            f"Epoch [{epoch:03d}/{epochs}] "
            f"Train Loss: {train_loss:.6f} "
            f"Val Loss: {val_loss:.6f}"
        )


# 学習履歴を保存
history_df = pd.DataFrame(history)

history_df.to_csv(
    os.path.join(output_dir, "training_history.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("学習完了")
print("Best Val Loss:", best_val_loss)


# ============================================================
# 12. Encoderで全データを10次元潜在ベクトルに変換
# ============================================================

model.load_state_dict(
    torch.load(best_model_path, map_location=device)
)

model.eval()

X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

all_z = []

with torch.no_grad():
    for i in range(0, len(X_tensor), batch_size):
        batch = X_tensor[i:i + batch_size]

        z = model.encoder(batch)

        all_z.append(z.cpu().numpy())

all_z = np.vstack(all_z)

print("潜在ベクトル shape:", all_z.shape)


# ============================================================
# 13. mesh_time_id ごとの潜在ベクトルを保存
# ============================================================

z_cols = [f"z{i + 1}" for i in range(latent_dim)]

z_df = pd.DataFrame(all_z, columns=z_cols)

output_df = pd.concat(
    [
        long_df[
            [
                "mesh_time_id",
                mesh_col,
                "time",
                "hour"
            ]
        ].reset_index(drop=True),
        z_df
    ],
    axis=1
)

output_path = os.path.join(output_dir, "mesh_time_to_z10.csv")

output_df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("保存完了:")
print(output_path)

print(output_df.head())


# ============================================================
# 14. 時刻別にも保存
# ============================================================

time_split_dir = os.path.join(output_dir, "time_split_z10")
os.makedirs(time_split_dir, exist_ok=True)

for t, sub_df in output_df.groupby("time"):
    save_path = os.path.join(
        time_split_dir,
        f"mesh_time_z10_{str(t).zfill(4)}.csv"
    )

    sub_df.to_csv(
        save_path,
        index=False,
        encoding="utf-8-sig"
    )

print("時刻別ファイルも保存しました:")
print(time_split_dir)
