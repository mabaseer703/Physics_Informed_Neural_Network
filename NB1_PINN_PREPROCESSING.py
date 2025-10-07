import pandas as pd
import os
import sys
import functools

# === Paths ===
save_dir = "/home/admin/BurningGH2/PINN_CLEANED"
os.makedirs(save_dir, exist_ok=True)

# === Setup fixed log file ===
log_file = os.path.join(save_dir, "PINN_LOG.log")
sys.stdout = open(log_file, 'w')
sys.stderr = sys.stdout
print = functools.partial(print, flush=True)
print(f"[LOG STARTED] Saving logs to: {log_file}")


# === File paths ===
dataset1_path = "/home/admin/BurningGH2/Dataset.csv"
dataset2_path = "/home/admin/BurningGH2/Dataset2.txt"
train_out = os.path.join(save_dir, "train1_preprocessed.csv")
val_out   = os.path.join(save_dir, "val1_preprocessed.csv")
test_out  = os.path.join(save_dir, "test2_preprocessed.csv")

# === Load Dataset1 (Turbine 2) ===
print("[INFO] Loading Dataset1...")
df1 = pd.read_csv(dataset1_path)
df1["datetime"] = pd.to_datetime(df1["datetime"], format="%d/%m/%Y %H:%M")

# === Clip negative values ===
clip_cols = ["Power (Hourly, SCADA)", "Ambient temperature (Hourly, SCADA)", "Rotor Speed (Hourly, SCADA)"]
for col in clip_cols:
    if col in df1.columns:
        df1[col] = df1[col].clip(lower=0)

# === Sort and Split into Train/Val ===
df1 = df1.sort_values("datetime")
df1["year"] = df1["datetime"].dt.year
cutoff_idx = int(len(df1["datetime"].unique()) * 0.8)
cutoff_date = df1["datetime"].sort_values().unique()[cutoff_idx]

train_df = df1[df1["datetime"] < cutoff_date].copy()
val_df   = df1[df1["datetime"] >= cutoff_date].copy()
train_df.drop(columns=["datetime", "year"], inplace=True)
val_df.drop(columns=["datetime", "year"], inplace=True)

# === Save train and val ===
train_df.to_csv(train_out, index=False)
val_df.to_csv(val_out, index=False)
print(f"[INFO] Train shape: {train_df.shape}, saved to: {train_out}")
print(f"[INFO] Val shape  : {val_df.shape}, saved to: {val_out}")

# === Load Dataset2 (Turbine 6) ===
print("[INFO] Loading Dataset2...")
df2 = pd.read_csv(dataset2_path)
df2 = df2.loc[:, ~df2.columns.str.contains("^Unnamed")]
df2["datetime"] = pd.to_datetime(df2["datetime"], format="%d/%m/%Y %H:%M")

# === Clip and drop datetime ===
for col in clip_cols:
    if col in df2.columns:
        df2[col] = df2[col].clip(lower=0)
df2.drop(columns=["datetime"], inplace=True)
df2.to_csv(test_out, index=False)
print(f"[INFO] Test shape : {df2.shape}, saved to: {test_out}")

# === Save y_test_seq_goal.npy for common use across all models ===
import numpy as np
from sklearn.preprocessing import MinMaxScaler

print("\n[INFO] Generating sequence labels for ground truth (Turbine 6)...")

# === Define sequence parameters ===
seq_length = 18
horizons = 6
target_cols = ['Power (Hourly, SCADA)', 'Wind Speed (Hourly, SCADA)', 'rho_kgm-3']

# === Sanity check ===
for col in target_cols:
    if col not in df2.columns:
        raise ValueError(f"[ERROR] Missing column '{col}' in test set.")

# === Sequence generation ===
def create_sequences(X, y, seq_len, horizons):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len - horizons + 1):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len:i+seq_len+horizons])
    return np.array(X_seq), np.array(y_seq)

X_test = df2.drop(columns=target_cols).values
y_test = df2[target_cols].values
_, y_test_seq = create_sequences(X_test, y_test, seq_length, horizons)

# === Save to common file ===
y_seq_save_path = "/home/admin/BurningGH2/PINN_CLEANED/y_test_seq_goal.npy"
np.save(y_seq_save_path, y_test_seq)
print(f"[INFO] y_test_seq shape: {y_test_seq.shape}, saved to: {y_seq_save_path}")

# === Print 5 samples of Ground Truth Power (after y_test_seq is saved) ===
print("\n=== Sample Ground Truth Power (First 5 Rows) ===")
power_ground_truth = y_test_seq[:, :, 0]  # Extract only power (first target)
import pandas as pd
df_power = pd.DataFrame(power_ground_truth[:5], columns=[f"Horizon_{i+1}" for i in range(horizons)])
print(df_power)

# === Final Summary ===
print("\n=== [SUMMARY] ===")
print(f"Turbine 2 => Train: {train_df.shape}, Val: {val_df.shape}")
print(f"Turbine 6 => Test : {df2.shape}")
print("[INFO] All processing complete.")

