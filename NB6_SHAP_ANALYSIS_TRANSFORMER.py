# ===================== SHAP ANALYSIS FOR TRANSFORMER (with units & x-label) =====================
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import MinMaxScaler
import os
import sys
import functools

# ============ PATHS ============
output_train_path = "/home/admin/BurningGH2/PINN_CLEANED/train1_preprocessed.csv"
output_test_path  = "/home/admin/BurningGH2/PINN_CLEANED/test2_preprocessed.csv"
model_path        = "/home/admin/BurningGH2/TRANSFORMER_WINDPOWER/Transformer_model_new_goal.h5"
save_dir          = "/home/admin/BurningGH2/SHAP_ANALYSIS_TRANSFORMER_WINDPOWER"
os.makedirs(save_dir, exist_ok=True)

# ============ LOGGING ============
log_file_path = os.path.join(save_dir, "SHAP_ANALYSIS_TRANSFORMER_WINDPOWER.log")
sys.stdout = open(log_file_path, 'w')
sys.stderr = sys.stdout
print = functools.partial(print, flush=True)

# ============ PARAMS ============
seq_length = 18
horizons   = 6
target_cols = ['Power (Hourly, SCADA)', 'Wind Speed (Hourly, SCADA)', 'rho_kgm-3']

# ============ DATA LOADING ============
train_df = pd.read_csv(output_train_path)
test_df  = pd.read_csv(output_test_path)

X_train = train_df.drop(columns=target_cols).values
y_train = train_df[target_cols].values
X_test  = test_df.drop(columns=target_cols).values
y_test  = test_df[target_cols].values

feature_names = train_df.drop(columns=target_cols).columns.tolist()

# ---------- unitized pretty labels (NO other changes) ----------
def pretty_label(col: str) -> str:
    mapping = {
        "windspeed_ms": "Wind speed (m/s)",
        "u_ms": "u (m/s)",
        "v_ms": "v (m/s)",
        "temperature_K": "Temperature (K)",
        "surf_pres_Pa": "Surface pressure (Pa)",
        "Ambient temperature (Hourly, SCADA)": "Ambient temperature (°C)",  # change to (K) if that's what you decided
        "Rotor Speed (Hourly, SCADA)": "Rotor Speed (rpm)",
        "Wind Speed (Hourly, SCADA)": "Wind Speed (SCADA) (m/s)",
        "rho_kgm-3": r"Air density (kg/m³)",
    }
    return mapping.get(col, col)

pretty_feature_names = [pretty_label(c) for c in feature_names]
print("[INFO] Using pretty feature labels with units:")
for raw, nice in zip(feature_names, pretty_feature_names):
    print(f"  {raw}  ->  {nice}")

# ============ SEQUENCES ============
def create_sequences(X, y, seq_len, horizons):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len - horizons + 1):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len:i+seq_len+horizons])
    return np.array(X_seq), np.array(y_seq)

X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length, horizons)
X_test_seq,  y_test_seq  = create_sequences(X_test,  y_test,  seq_length, horizons)

# ============ SCALING ============
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train_seq.reshape(-1, X_train_seq.shape[-1])).reshape(X_train_seq.shape)
X_test_scaled  = scaler_X.transform(X_test_seq.reshape(-1,  X_test_seq.shape[-1])).reshape(X_test_seq.shape)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor  = torch.tensor(X_test_scaled,  dtype=torch.float32)

# ============ MODEL DEFINITION (unchanged) ============
class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, d_model, dropout=0.3):
        super().__init__()
        self.temper = np.sqrt(d_model)
        self.dropout = torch.nn.Dropout(dropout)
    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.temper
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.3):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = torch.nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = torch.nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = torch.nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = torch.nn.Linear(n_head * d_v, d_model)
        self.attention = ScaledDotProductAttention(d_model, dropout)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(d_model)
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

class PositionwiseFeedForward(torch.nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.3):
        super().__init__()
        self.w_1 = torch.nn.Linear(d_in, d_hid)
        self.w_2 = torch.nn.Linear(d_hid, d_in)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(d_in)
    def forward(self, x):
        residual = x
        x = self.w_2(torch.relu(self.w_1(x)))
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x

class PositionalEncoding(torch.nn.Module):
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

class TransformerModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, seq_len, n_head=2, d_k=8, d_v=8, d_model=64, d_inner=192, dropout=0.3, horizons=6):
        super().__init__()
        self.horizons = horizons
        self.output_dim = output_dim
        self.enc_embedding = torch.nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, n_position=seq_len)
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout)
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc_out = torch.nn.Sequential(
            torch.nn.Linear(d_model, 64),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(64, output_dim * horizons)
        )
    def forward(self, x):
        x = self.enc_embedding(x)
        x = self.pos_enc(x)
        x, _ = self.slf_attn(x, x, x)
        x = self.pos_ffn(x)
        x = self.avg_pool(x.transpose(1, 2)).squeeze(-1)
        output = self.fc_out(x)
        output = output.view(-1, self.horizons, self.output_dim)
        power_output = torch.relu(output[:, :, 0])
        speed_output = torch.sigmoid(output[:, :, 1])
        rho_output   = torch.sigmoid(output[:, :, 2])
        return torch.stack([power_output, speed_output, rho_output], dim=2)

# ============ Load pretrained model ============
input_dim = X_train_tensor.shape[2]
model = TransformerModel(
    input_dim=input_dim,
    output_dim=3,
    seq_len=seq_length,
    horizons=horizons
).to("cpu")

model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()
print("[INFO] Model loaded successfully.")

# ============ SHAP Preparation ============
class WrappedModel(torch.nn.Module):
    def __init__(self, base_model, horizon_idx=0, target_idx=0):
        super().__init__()
        self.base_model = base_model
        self.horizon_idx = horizon_idx
        self.target_idx = target_idx
    def forward(self, x):
        x = x.reshape(-1, seq_length, input_dim)
        out = self.base_model(x)
        return out[:, self.horizon_idx, self.target_idx].unsqueeze(1)

background_data = X_train_tensor[:100].mean(dim=1)
test_samples    = X_test_tensor[:200].mean(dim=1)

X_bg      = background_data.numpy()
X_test_np = test_samples.numpy()

all_shap_values = []

print(f"[INFO] SHAP for Power (Averaged over {horizons} horizons)")
for h in range(horizons):
    print(f"[DEBUG] Horizon {h+1}")
    wrapped_model = WrappedModel(model, horizon_idx=h, target_idx=0)
    def predict_fn(x):
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1).repeat(1, seq_length, 1)
        with torch.no_grad():
            preds = wrapped_model(x_tensor)
        return preds.numpy().flatten()

    explainer = shap.KernelExplainer(predict_fn, X_bg)
    shap_vals = explainer.shap_values(X_test_np, nsamples=50)
    shap_vals = np.array(shap_vals)
    if shap_vals.shape[0] != X_test_np.shape[0]:
        shap_vals = shap_vals.T
    all_shap_values.append(shap_vals)

# ============ Aggregate and Plot ============
shap_values     = np.mean(np.abs(all_shap_values), axis=0)
mean_shap_vals  = shap_values.mean(axis=0)
raw_power       = train_df[target_cols[0]].values
power_range     = raw_power.max() - raw_power.min()
adjusted_shap_vals = mean_shap_vals * power_range

sorted_idx   = np.argsort(adjusted_shap_vals)[::-1]
top_n        = min(20, len(feature_names))
top_idx      = sorted_idx[:top_n]
top_vals     = adjusted_shap_vals[top_idx]
top_features = np.array(pretty_feature_names)[top_idx]   # << use pretty labels here

# --------- Bar plot with units + Transformer x-label ----------
plt.figure(figsize=(10, 6 + 0.3 * top_n))
y_pos = range(len(top_vals))[::-1]
bars = plt.barh(y_pos, top_vals, color='crimson')

for pos, val in zip(y_pos, top_vals):
    label = f"{val:.2f}"
    if val >= top_vals.max() * 0.35:
        plt.text(val * 0.5, pos, label, va='center', ha='center', fontsize=11, color='white')
    else:
        plt.text(val + top_vals.max() * 0.02, pos, label, va='center', ha='left', fontsize=11, color='crimson')

plt.yticks(y_pos, top_features, fontsize=12)
plt.xlabel("Transformer SHAP value (impact on model output)", fontsize=12)   # << renamed
plt.title("Top Features by SHAP Value\n(Power Forecast)", fontsize=14, pad=15)
plt.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "shap_power_mean_all_horizons.png"), dpi=300, bbox_inches='tight')
print("[INFO] SHAP bar plot saved.")

# Save to CSV
shap_df = pd.DataFrame({'feature': pretty_feature_names, 'mean_shap_value': mean_shap_vals * power_range})
shap_df = shap_df.sort_values('mean_shap_value', ascending=False)
shap_df.to_csv(os.path.join(save_dir, "shap_values.csv"), index=False)
print("[INFO] SHAP values saved to CSV.")

# ========= Beeswarm (units in feature names + Transformer x-label) =========
print("[INFO] Generating SHAP beeswarm plot...")
shap_values_final  = np.mean(np.abs(all_shap_values), axis=0)   # [samples, features]
test_sample_values = test_samples.numpy()

top_n = min(20, shap_values_final.shape[1])
mean_shap = shap_values_final.mean(axis=0)
sorted_idx = np.argsort(mean_shap)[::-1][:top_n]

top_shap_vals   = shap_values_final[:, sorted_idx]
top_input_vals  = test_sample_values[:, sorted_idx]
top_feat_names  = [pretty_feature_names[i] for i in sorted_idx]

shap.summary_plot(
    top_shap_vals,
    top_input_vals,
    feature_names=top_feat_names,
    plot_type="dot",
    max_display=top_n,
    color_bar=True,
    show=False
)

# rename x-axis
fig = plt.gcf()
fig.axes[0].set_xlabel("Transformer SHAP value (impact on model output)")

plt.tight_layout()
beeswarm_path = os.path.join(save_dir, "shap_beeswarm_top20_features.png")
plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"[INFO] SHAP beeswarm plot saved: {beeswarm_path}")
