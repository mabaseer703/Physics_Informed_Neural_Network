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
from scipy.interpolate import interp1d
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

# ================= FILE PATHS (NEW, NO OVERLAP) =================
output_train_path = "/home/admin/BurningGH2/PINN_CLEANED/train1_preprocessed.csv"
output_test_path  = "/home/admin/BurningGH2/PINN_CLEANED/test2_preprocessed.csv"
val_out           = "/home/admin/BurningGH2/PINN_CLEANED/val1_preprocessed.csv"

save_dir = "/home/admin/BurningGH2/PI_TRANSFORMER_WINDPOWER_IDEALGAS"
os.makedirs(save_dir, exist_ok=True)

log_file_path   = os.path.join(save_dir, "PI_TRANSFORMER_WINDPOWER_IDEALGAS.log")
sys.stdout = open(log_file_path, 'w')
sys.stderr = sys.stdout
print = functools.partial(print, flush=True)

model_path        = os.path.join(save_dir, "PI_Transformer_model_idealgas.h5")
predictions_path  = os.path.join(save_dir, "PI_Transformer_predictions_inv_h6_idealgas.csv")
y_seq_path        = os.path.join(save_dir, "y_test_seq_goal_idealgas.npy")
loss_plot_path    = os.path.join(save_dir, "training_validation_loss_idealgas.png")
time_metrics_path = os.path.join(save_dir, "PI_transformer_time_metrics_idealgas.csv")

print(f"[INFO] Outputs will be saved to: {save_dir}")
print(f"[INFO] Model will be saved to: {model_path}")
print(f"[INFO] Predictions will be saved to: {predictions_path}")
print(f"[INFO] y_test_seq will be saved to: {y_seq_path}")

# ================= HYPERPARAMETERS =================
seq_length = 18
horizons   = 6
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
test_df  = pd.read_csv(output_test_path)
val_df   = pd.read_csv(val_out)

X_train = train_df.drop(columns=target_cols).values
y_train = train_df[target_cols].values
X_test  = test_df.drop(columns=target_cols).values
y_test  = test_df[target_cols].values
X_val   = val_df.drop(columns=target_cols).values
y_val   = val_df[target_cols].values

X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length, horizons)
X_test_seq,  y_test_seq  = create_sequences(X_test,  y_test,  seq_length, horizons)
X_val_seq,   y_val_seq   = create_sequences(X_val,   y_val,   seq_length, horizons)

# --- Keep unscaled copies for physics (p, T) ---
X_train_seq_unscaled = X_train_seq.copy()
X_val_seq_unscaled   = X_val_seq.copy()
X_test_seq_unscaled  = X_test_seq.copy()

np.save(y_seq_path, y_test_seq)
print(f"[INFO] Saved y_test_seq (unscaled) to: {y_seq_path}")

# ================= SCALING =================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train_seq.reshape(-1, X_train_seq.shape[-1])).reshape(X_train_seq.shape)
X_val_scaled   = scaler_X.transform(X_val_seq.reshape(-1, X_val_seq.shape[-1])).reshape(X_val_seq.shape)
X_test_scaled  = scaler_X.transform(X_test_seq.reshape(-1, X_test_seq.shape[-1])).reshape(X_test_seq.shape)

y_train_scaled = scaler_y.fit_transform(y_train_seq.reshape(-1, 3)).reshape(y_train_seq.shape)
y_val_scaled   = scaler_y.transform(y_val_seq.reshape(-1, 3)).reshape(y_val_seq.shape)
y_test_scaled  = scaler_y.transform(y_test_seq.reshape(-1, 3)).reshape(y_test_seq.shape)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_val_tensor   = torch.tensor(X_val_scaled,   dtype=torch.float32)
y_val_tensor   = torch.tensor(y_val_scaled,   dtype=torch.float32)
X_test_tensor  = torch.tensor(X_test_scaled,  dtype=torch.float32)

# Also keep UN-SCALED X tensors for (p,T) in ideal-gas term
X_train_unscaled_t = torch.tensor(X_train_seq_unscaled, dtype=torch.float32)
X_val_unscaled_t   = torch.tensor(X_val_seq_unscaled,   dtype=torch.float32)
X_test_unscaled_t  = torch.tensor(X_test_seq_unscaled,  dtype=torch.float32)

# ================= CP INTERPOLATION =================
physics_df = train_df.copy()
physics_df['c_p'] = (physics_df['Power (Hourly, SCADA)'] * 1000) / (
    0.5 * (physics_df['Wind Speed (Hourly, SCADA)'] ** 3) * physics_df['rho_kgm-3'] * 6720
)
avg_cp = physics_df.groupby('Wind Speed (Hourly, SCADA)')['c_p'].mean()
x_new = avg_cp.index.values
y_new = np.clip(pd.Series(avg_cp.values).rolling(40, min_periods=1).mean().fillna(0).values, 0.05, 0.5)

# ================= Indices for p & T (use your exact column names) =================
feature_names = list(train_df.drop(columns=target_cols).columns)
idx_T = feature_names.index('temperature_K')  # Kelvin
idx_p = feature_names.index('surf_pres_Pa')   # Pascals

# === Physics-Informed Loss (Option B in SCALED space to match OLD loss scale) ===
class PhysicsInformedLoss(nn.Module):
    """
    - Data loss: MSE on SCALED power (exactly like OLD).
    - Physics power loss: unchanged (uses scaled pred_v, pred_rho & tiny beta).
    - NEW: Ideal-gas rho consistency, computed in REAL units then scaled with the SAME single scaler_y,
           and compared in SCALED space to keep magnitudes consistent with OLD.
    """
    def __init__(self, x_new, y_new, scaler_y, area=6720.0, alpha=1.0, beta=1e-6, gamma=1e-3):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.cp_interp = interp1d(x_new, y_new, kind='linear', bounds_error=False, fill_value="extrapolate")
        self.area = area
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.R_SPEC = 287.05  # J/(kg·K)
        self.scaler_y = scaler_y

    def _cp_from_speed_scaled(self, v_scaled):
        v_np = v_scaled.detach().cpu().numpy()
        v_np = np.clip(v_np, x_new.min(), x_new.max())  # preserve OLD guard
        cp_np = self.cp_interp(v_np)
        return torch.tensor(cp_np, dtype=torch.float32, device=v_scaled.device)

    def _scale_rho_like_target(self, rho_real):
        B, H = rho_real.shape
        dummy = np.zeros((B*H, 3), dtype=np.float64)
        dummy[:, 2] = rho_real.detach().cpu().numpy().reshape(-1)  # explicit NumPy
        rho_scaled_np = self.scaler_y.transform(dummy)[:, 2]
        return torch.tensor(rho_scaled_np.reshape(B, H), dtype=torch.float32, device=rho_real.device)


    def forward(self, forecast_scaled, y_true_scaled, X_inputs_unscaled, idx_p, idx_T):
        # Split predictions (scaled)
        P_pred_s   = forecast_scaled[:, :, 0]
        v_pred_s   = forecast_scaled[:, :, 1]
        rho_pred_s = forecast_scaled[:, :, 2]

        # Data loss in scaled space 
        P_true_s = y_true_scaled[:, :, 0]
        data_loss = self.mse_loss(P_pred_s, P_true_s)

        # Physics power loss (unchanged style, still scaled preds, tiny beta)
        cp = self._cp_from_speed_scaled(v_pred_s)
        expected_power_s = 0.5 * rho_pred_s * self.area * cp * (v_pred_s ** 3) / 1000.0
        physics_loss = self.mse_loss(P_pred_s, expected_power_s)

        # Ideal-gas rho consistency: compute rho_ideal in REAL, then scale with same scaler_y
        p = X_inputs_unscaled[:, -1, idx_p]  # Pa
        T = X_inputs_unscaled[:, -1, idx_T]  # K
        pH = p.unsqueeze(1).expand_as(P_pred_s)
        TH = T.unsqueeze(1).expand_as(P_pred_s)
        rho_ideal_real = pH / (self.R_SPEC * TH)  # kg/m^3

        rho_ideal_s = self._scale_rho_like_target(rho_ideal_real)  # scaled to match rho_pred_s scale
        rho_consistency = self.mse_loss(rho_pred_s, rho_ideal_s)

        total = self.alpha*data_loss + self.beta*physics_loss + self.gamma*rho_consistency
        return total, data_loss, physics_loss, rho_consistency

# 4. Transformer Model (UNCHANGED)
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
        table[:, 0::2] = np.sin(table[:, 0::2]); table[:, 1::2] = np.cos(table[:, 1::2])
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
        power_output = F.relu(output[:, :, 0])          # keep OLD activations
        speed_output = torch.sigmoid(output[:, :, 1])
        rho_output   = torch.sigmoid(output[:, :, 2])
        return torch.stack([power_output, speed_output, rho_output], dim=2)

# ================= TRAINING =================
start_train = time.time()

model = TransformerModel(
    input_dim=X_train_seq.shape[2],
    output_dim=3,
    seq_len=seq_length,
    n_head=2, d_k=8, d_v=8, d_model=64, d_inner=192, dropout=0.3,
    horizons=horizons
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.000012)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=5)

loss_fn = PhysicsInformedLoss(
    x_new, y_new, scaler_y=scaler_y, area=6720.0, alpha=1.0, beta=1e-6, gamma=1e-2
).to(device)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor, X_train_unscaled_t), batch_size=32, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_tensor,   y_val_tensor,   X_val_unscaled_t),   batch_size=32)

train_losses, val_losses = [], []
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(100):
    model.train()
    train_loss = 0
    for xb, yb, xb_u in train_loader:
        xb, yb, xb_u = xb.to(device), yb.to(device), xb_u.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss, dloss, ploss, rloss = loss_fn(out, yb, xb_u, idx_p, idx_T)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb, xb_u in val_loader:
            xb, yb, xb_u = xb.to(device), yb.to(device), xb_u.to(device)
            out = model(xb)
            vloss, _, _, _ = loss_fn(out, yb, xb_u, idx_p, idx_T)
            val_loss += vloss.item()

    train_losses.append(train_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))
    scheduler.step(val_losses[-1])
    print(f"Epoch {epoch+1:03d} | Train: {train_losses[-1]:.6f} | Val: {val_losses[-1]:.6f}")

    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        patience_counter = 0
        torch.save(model.state_dict(), model_path)
    else:
        patience_counter += 1
        if patience_counter >= 15:
            print("[EARLY STOP] Patience reached.")
            break

end_train = time.time()
train_time = end_train - start_train
print(f"[TIMER] Training Time (s): {train_time:.3f}")

# ================= INFERENCE =================
model.load_state_dict(torch.load(model_path))
model.eval()

inference_times = []
with torch.no_grad():
    for i in range(X_test_tensor.shape[0]):
        sample = X_test_tensor[i:i+1].to(device)
        t0 = time.time()
        _ = model(sample)
        t1 = time.time()
        inference_times.append(t1 - t0)
inference_times = np.array(inference_times)
avg_inf_time_ms = 1000 * inference_times.mean()
std_inf_time_ms = 1000 * inference_times.std()
print(f"[TIMER] Average Inference Time per sample (ms): {avg_inf_time_ms:.4f}")
print(f"[TIMER] Inference Time Std Dev per sample (ms): {std_inf_time_ms:.4f}")

# === Predict on test (scaled outputs) ===
predictions = model(X_test_tensor.to(device)).cpu().detach().numpy()
pred_power = predictions[:, :, 0]  # scaled power (as in OLD)
pred_speed = predictions[:, :, 1]
pred_rho   = predictions[:, :, 2]

# === Cp Plot (unchanged) ===
cp_values = np.interp(pred_speed, x_new, y_new)
physics_power = 0.5 * pred_rho * 6720 * cp_values * pred_speed**3 / 1000

plt.figure(figsize=(8, 5))
plt.plot(x_new, y_new, label="Smoothed Cp", linewidth=2)
plt.xlabel("Wind Speed (m/s)"); plt.ylabel("Power Coefficient (Cp)")
plt.title("Interpolated Cp vs Wind Speed"); plt.grid(True); plt.legend(); plt.tight_layout()
plt.show()

# === Save predicted POWER (scaled) then inverse-transform to kW as in OLD ===
pred_df = pd.DataFrame(pred_power, columns=[f"Horizon_{i+1}" for i in range(pred_power.shape[1])])
pred_df.to_csv(predictions_path, index=False)

# === Final Loss Plot ===
plt.figure(figsize=(12, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title("PI_Transformer (Ideal-Gas) Training vs Validation Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(loss_plot_path)
print(f"[INFO] Training vs Validation Loss plot saved to: {loss_plot_path}")

# === Evaluation ===
df_train = pd.read_csv(output_train_path)
pred_df_vals = pd.read_csv(predictions_path).values
y_seq = np.load(y_seq_path)
gt_power = y_seq[:, :, 0]
assert pred_df_vals.shape == gt_power.shape

gt_df   = pd.DataFrame(gt_power, columns=[f"Horizon_{i+1}" for i in range(gt_power.shape[1])])
pred_df = pd.DataFrame(pred_df_vals, columns=[f"Horizon_{i+1}" for i in range(pred_df_vals.shape[1])])

# Inverse-transform ONLY power to kW (fit scaler on train power column) — same as OLD
power_scaler = MinMaxScaler()
power_scaler.fit(df_train[['Power (Hourly, SCADA)']].values)
pred_df_unscaled = power_scaler.inverse_transform(pred_df.values)
pred_df = pd.DataFrame(pred_df_unscaled, columns=[f"Horizon_{i+1}" for i in range(pred_df.shape[1])])

pd.options.display.float_format = '{:.2f}'.format
print("\n=== Ground Truth Power (First 5 Samples) ===")
print(gt_df.head())
print("\n=== Predicted Power (First 5 Samples, kW) ===")
print(pred_df.head())
pred_df.to_csv(predictions_path, index=False)
print(f"[INFO] Predictions saved (kW): {predictions_path}")
print(gt_power.shape)
print(pred_df.shape)
print("\nGround Truth Sample 0:", gt_power[0])
print("Prediction Sample 0    :", pred_df.iloc[0].values)

# === Save time metrics ===
summary_df = pd.DataFrame([{
    "Model": "PI_TRANSFORMER_IDEALGAS",
    "Training Time (s)": train_time,
    "Average Inference Time per sample (ms)": avg_inf_time_ms,
    "Average Inference Std Dev per sample (ms)": std_inf_time_ms,
}])
summary_df.to_csv(time_metrics_path, index=False)
print(f"[INFO] Time metrics saved to: {time_metrics_path}")
