# ================= IMPORTS =================
import sys
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

import random
import os

# ================= SEED SETUP =================
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= FILE PATHS =================
output_train_path = "/home/admin/BurningGH2/PINN_CLEANED/train1_preprocessed.csv"
output_test_path = "/home/admin/BurningGH2/PINN_CLEANED/test2_preprocessed.csv"
val_out = "/home/admin/BurningGH2/PINN_CLEANED/val1_preprocessed.csv"

# ================= CONFIG =================
log_file_path = '/home/admin/BurningGH2/TRANSFORMER_WINDPOWER.log'
sys.stdout = open(log_file_path, 'w')
sys.stderr = sys.stdout
print = functools.partial(print, flush=True)

save_dir = '/home/admin/BurningGH2/TRANSFORMER_WINDPOWER'
os.makedirs(save_dir, exist_ok=True)
print(f"[INFO] Outputs will be saved to: {save_dir}")

# Save paths inside the same directory
model_path = os.path.join(save_dir, "Transformer_model_new_goal.h5")
predictions_path = os.path.join(save_dir, "Transformer_predictions_inv_h6_goal.csv")
y_seq_path = "/home/admin/BurningGH2/PINN_CLEANED/y_test_seq_goal.npy"


print("[INFO] Logging setup is working correctly.")
print(f"[INFO] Model will be saved to: {model_path}")
print(f"[INFO] Predictions will be saved to: {predictions_path}")
print(f"[INFO] y_test_seq will be saved to: {y_seq_path}")


# ================= HYPERPARAMETERS =================
seq_length = 18
horizons = 6
target_cols = ['Power (Hourly, SCADA)', 'Wind Speed (Hourly, SCADA)', 'rho_kgm-3']

# ================= FUNCTIONS =================
def create_sequences(X, y, seq_len, horizons):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len - horizons + 1):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len:i+seq_len+horizons])
    return np.array(X_seq), np.array(y_seq)

# ================= DATA LOADING =================
train_df = pd.read_csv(output_train_path)
test_df = pd.read_csv(output_test_path)
val_df = pd.read_csv(val_out)

X_train = train_df.drop(columns=target_cols).values
y_train = train_df[target_cols].values
X_test = test_df.drop(columns=target_cols).values
y_test = test_df[target_cols].values
X_val = val_df.drop(columns=target_cols).values
y_val = val_df[target_cols].values

X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length, horizons)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length, horizons)
X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_length, horizons)

np.save(y_seq_path, y_test_seq)

# ================= SCALING =================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train_seq.reshape(-1, X_train_seq.shape[-1])).reshape(X_train_seq.shape)
X_val_scaled = scaler_X.transform(X_val_seq.reshape(-1, X_val_seq.shape[-1])).reshape(X_val_seq.shape)
X_test_scaled = scaler_X.transform(X_test_seq.reshape(-1, X_test_seq.shape[-1])).reshape(X_test_seq.shape)
y_train_scaled = scaler_y.fit_transform(y_train_seq.reshape(-1, 3)).reshape(y_train_seq.shape)
y_val_scaled = scaler_y.transform(y_val_seq.reshape(-1, 3)).reshape(y_val_seq.shape)
y_test_scaled = scaler_y.transform(y_test_seq.reshape(-1, 3)).reshape(y_test_seq.shape)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

# ================= CP INTERPOLATION =================


# === Physics-Informed Loss === (as defined already)

# 4. Transformer Model
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, dropout=0.3):
        super().__init__()
        self.temper = np.sqrt(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.temper
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.3):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model)
        self.attention = ScaledDotProductAttention(d_model, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, mask=None):
        sz_b, len_q = q.size(0), q.size(1)
        residual = q
        q = self.w_qs(q).view(sz_b, len_q, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_ks(k).view(sz_b, len_q, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_vs(v).view(sz_b, len_q, self.n_head, self.d_v).transpose(1, 2)
        output, attn = self.attention(q, k, v, mask)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output, attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.3):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_in)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super().__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_angle(pos, i):
            return pos / np.power(10000, 2 * (i // 2) / d_hid)
        table = np.array([[get_angle(pos, i) for i in range(d_hid)] for pos in range(n_position)])
        table[:, 0::2] = np.sin(table[:, 0::2])
        table[:, 1::2] = np.cos(table[:, 1::2])
        return torch.FloatTensor(table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len, n_head=2, d_k=8, d_v=8, d_model=32, d_inner=192, dropout=0.4, horizons=6):
        super().__init__()
        self.horizons = horizons
        self.output_dim = output_dim
        self.enc_embedding = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, n_position=seq_len)
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, output_dim * horizons)
        )

    def forward(self, x):
        x = self.enc_embedding(x)
        x = self.pos_enc(x)
        x, _ = self.slf_attn(x, x, x)
        x = self.pos_ffn(x)
        x = self.avg_pool(x.transpose(1, 2)).squeeze(-1)
        output = self.fc_out(x)
        output = output.view(-1, self.horizons, self.output_dim)              
        return output


start_train = time.time()
# ================= TRAINING =================
model = TransformerModel(
    input_dim=X_train_seq.shape[2],
    output_dim=3,
    seq_len=seq_length,
    n_head=2,
    d_k=8,
    d_v=8,
    d_model=64,
    d_inner=192,
    dropout=0.3,
    horizons=horizons
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.000012)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=5)
loss_fn = nn.MSELoss()  # REMOVED PhysicsInformedLoss, use standard MSE

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=32)

train_losses, val_losses = [], []
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(100):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)  # REMOVED physics loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            vloss = loss_fn(out, yb)  # REMOVED physics loss
            val_loss += vloss.item()

    train_losses.append(train_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))
    scheduler.step(val_losses[-1])
    print(f"Epoch {epoch+1}, Train: {train_losses[-1]:.4f}, Val: {val_losses[-1]:.4f}")

    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        patience_counter = 0
        torch.save(model.state_dict(), model_path)
    else:
        patience_counter += 1
        if patience_counter >= 15:
            break

end_train = time.time()  # <--- End timing here
train_time = end_train - start_train
print(f"[TIMER] Training Time (s): {train_time:.3f}")

# === Inference ===
model.load_state_dict(torch.load(model_path))
model.eval()


# === Inference Timing Block ===
inference_times = []
with torch.no_grad():
    for i in range(X_test_tensor.shape[0]):
        sample = X_test_tensor[i:i+1].to(device)  # Single sample batch
        t0 = time.time()
        _ = model(sample)
        t1 = time.time()
        inference_times.append(t1 - t0)
inference_times = np.array(inference_times)
avg_inf_time_ms = 1000 * inference_times.mean()
std_inf_time_ms = 1000 * inference_times.std()
print(f"[TIMER] Average Inference Time per sample (ms): {avg_inf_time_ms:.4f}")
print(f"[TIMER] Inference Time Std Dev per sample (ms): {std_inf_time_ms:.4f}")





predictions = model(X_test_tensor.to(device)).cpu().detach().numpy()
pred_power = predictions[:, :, 0]
pred_speed = predictions[:, :, 1]
pred_rho   = predictions[:, :, 2]


# === Cp Plot ===

# === Save predicted values ===
pred_df = pd.DataFrame(pred_power, columns=[f"Horizon_{i+1}" for i in range(pred_power.shape[1])])
pred_df.to_csv(predictions_path, index=False)

# === Final Loss Plot ===
plt.figure(figsize=(12, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title("Transformer Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Evaluation ===
df_train = pd.read_csv(output_train_path)
pred_df = pd.read_csv(predictions_path).values
y_seq = np.load(y_seq_path)
gt_power = y_seq[:, :, 0]
assert pred_df.shape == gt_power.shape

gt_df = pd.DataFrame(gt_power, columns=[f"Horizon_{i+1}" for i in range(gt_power.shape[1])])
pred_df = pd.DataFrame(pred_df, columns=[f"Horizon_{i+1}" for i in range(pred_df.shape[1])])

power_scaler = MinMaxScaler()
power_scaler.fit(df_train[['Power (Hourly, SCADA)']].values)
pred_df_unscaled = power_scaler.inverse_transform(pred_df.values)
pred_df = pd.DataFrame(pred_df_unscaled, columns=[f"Horizon_{i+1}" for i in range(pred_df.shape[1])])

pd.options.display.float_format = '{:.2f}'.format
print("\n=== Ground Truth Power (First 5 Samples) ===")
print(gt_df.head())
print("\n=== Predicted Power (First 5 Samples) ===")
print(pred_df.head())
pred_df.to_csv(predictions_path, index=False)
print(f"[INFO] Predictions saved: {predictions_path}")
print(gt_power.shape)
print(pred_df.shape)
print("\nGround Truth Sample 0:", gt_power[0])
print("Prediction Sample 0    :", pred_df.iloc[0].values)

# === Save the Training vs Validation Loss Plot ===
loss_plot_path = os.path.join(save_dir, "training_validation_loss.png")
plt.savefig(loss_plot_path)
print(f"[INFO] Training vs Validation Loss plot saved to: {loss_plot_path}")


summary_row = {
    "Model": "TRANSFORMER",
    "Training Time (s)": train_time,
    "Average Inference Time per sample (ms)": avg_inf_time_ms,
    "Average Inference Std Dev per sample(ms)": std_inf_time_ms,
}
summary_df = pd.DataFrame([summary_row])
summary_df.to_csv(os.path.join(save_dir, "transformer_time_metrics.csv"), index=False)
print(summary_df)