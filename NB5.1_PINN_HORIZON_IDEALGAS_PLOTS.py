import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error

# === CONFIG ===
save_dir = "/home/admin/BurningGH2/PINN_HORIZON_IDEALGAS_PLOTS"
os.makedirs(save_dir, exist_ok=True)

log_file = os.path.join(save_dir, "PINN_HORIZON_IDEALGAS_PLOTS.log")
sys.stdout = open(log_file, "w")
sys.stderr = sys.stdout

print(f"[INFO] Logging to: {log_file}")
print(f"[INFO] Saving plots to: {save_dir}")

# === Load Ground Truth (Already Inversed) ===
y_seq_path = "/home/admin/BurningGH2/PI_TRANSFORMER_WINDPOWER_IDEALGAS/y_test_seq_goal_idealgas.npy"
y_seq = np.load(y_seq_path)
ground_truth_full = pd.DataFrame(y_seq[:, :, 0], columns=[f"Horizon_{i+1}" for i in range(y_seq.shape[1])])

# === Model prediction paths ===
models_to_plot = {
    "PI_TRANSFORMER": "/home/admin/BurningGH2/PI_TRANSFORMER_WINDPOWER_IDEALGAS/PI_Transformer_predictions_inv_h6_idealgas.csv",
    "TRANSFORMER": "/home/admin/BurningGH2/TRANSFORMER_WINDPOWER/Transformer_predictions_inv_h6_goal.csv"
   
}


# === MAE Records ===
mae_records = []

for model_name, pred_path in models_to_plot.items():
    print(f"\n[INFO] Processing model: {model_name}")
    pred_df = pd.read_csv(pred_path)

    # Shift predictions for each horizon
    for h in range(6):
        pred_df[f"Horizon_{h+1}"] = pred_df[f"Horizon_{h+1}"].shift(-(h + 1))
    pred_df.dropna(inplace=True)

    # Align ground truth with predictions
    ground_truth_aligned = ground_truth_full.iloc[:len(pred_df)].copy()

    # Set date index
    start_date = pd.to_datetime("2020-06-30 18:00:00")
    date_range = pd.date_range(start=start_date, periods=len(pred_df), freq='h')
    pred_df["Date"] = date_range
    ground_truth_aligned["Date"] = date_range[:len(ground_truth_aligned)]

    merged_all = pd.merge(ground_truth_aligned, pred_df, on="Date", suffixes=("_actual", "_predicted"))
    
    # Get ACTUAL data range
    data_start = merged_all['Date'].min()
    data_end = merged_all['Date'].max()
    print(f"[DATA] Actual date range: {data_start} to {data_end}")

    # === Dynamic Time Windows ===
    time_windows = {
        "1_Year": (data_start.strftime("%Y-%m-%d"), data_end.strftime("%Y-%m-%d")),
        "6_Months": ((data_end - pd.DateOffset(months=6)).strftime("%Y-%m-%d"), 
                    data_end.strftime("%Y-%m-%d")),
        "1_Month": ((data_end - pd.DateOffset(months=1)).strftime("%Y-%m-%d"), 
                   data_end.strftime("%Y-%m-%d")),
        "1_Week": ((data_end - pd.Timedelta(weeks=1)).strftime("%Y-%m-%d"), 
                  data_end.strftime("%Y-%m-%d"))
    }

    for window_label, (start_str, end_str) in time_windows.items():
        print(f"[INFO]  ➤ Window: {window_label} ({start_str} to {end_str})")
        start_window = pd.to_datetime(start_str)
        end_window = pd.to_datetime(end_str)
        subset = merged_all[(merged_all["Date"] >= start_window) & (merged_all["Date"] <= end_window)].copy()
        
        if subset.empty:
            print(f"[WARNING] No data found for window {window_label}")
            continue

        # Extract and rename columns
        gt_filtered = subset[[col for col in subset.columns if col.endswith("_actual")]].copy()
        pred_filtered = subset[[col for col in subset.columns if col.endswith("_predicted")]].copy()
        gt_filtered.columns = [c.replace("_actual", "") for c in gt_filtered.columns]
        pred_filtered.columns = [c.replace("_predicted", "") for c in pred_filtered.columns]

        # === Compute MAE
        for h in range(6):
            mae = mean_absolute_error(gt_filtered[f"Horizon_{h+1}"], pred_filtered[f"Horizon_{h+1}"])
            mae_records.append({
                "Model": model_name,
                "Window": window_label,
                "Horizon": f"Horizon_{h+1}",
                "MAE": round(mae, 2)
            })

        # === Time-series Plot ===
        fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True)
        for h in range(6):
            row, col = divmod(h, 3)
            ax = axes[row][col]
            ax.plot(subset["Date"], gt_filtered[f"Horizon_{h+1}"], label="Actual", linewidth=1.5)
            ax.plot(subset["Date"], pred_filtered[f"Horizon_{h+1}"], label=f"Predicted (H{h+1})", linestyle="--", linewidth=1.5)
            ax.set_title(f"Horizon {h+1}")
            if row == 0:
                ax.set_xticklabels([])
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(fontsize=9)

        for ax in axes[1]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
            ax.tick_params(axis="x", rotation=45, labelsize=9)

        fig.text(0.04, 0.5, "Power (kW)", va="center", rotation="vertical", fontsize=13, fontweight="bold")
        fig.text(0.5, 0.02, "Date", ha="center", fontsize=13, fontweight="bold")
        start_fmt = pd.to_datetime(start_str).strftime("%d/%m/%Y")
        end_fmt = pd.to_datetime(end_str).strftime("%d/%m/%Y")
        title_str = f"{model_name} Wind Power Forecast – {window_label.replace('_', ' ')} ({start_fmt} to {end_fmt})"

        fig.suptitle(title_str, fontsize=15, fontweight="bold")
        plt.tight_layout(rect=[0.04, 0.05, 1, 0.95])
        plt.savefig(os.path.join(save_dir, f"{model_name}_{window_label}_PLOT.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # === 24-Hour Plot ===
    print(f"[INFO]  ➤ Plotting last 24h horizon for: {model_name}")
    day_end = data_end
    day_start = day_end - pd.Timedelta(hours=23)
    
    subset_day = merged_all[(merged_all["Date"] >= day_start) & (merged_all["Date"] <= day_end)].copy()
    
    if not subset_day.empty:
        gt_day = subset_day[[c for c in subset_day.columns if c.endswith("_actual")]].copy()
        pred_day = subset_day[[c for c in subset_day.columns if c.endswith("_predicted")]].copy()
        gt_day.columns = [c.replace("_actual", "") for c in gt_day.columns]
        pred_day.columns = [c.replace("_predicted", "") for c in pred_day.columns]

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        hours = np.arange(1, 25)
        hour_labels = [f"{(day_start + pd.Timedelta(hours=h-1)).strftime('%H:%M')}" for h in hours]


        for h in range(6):
            row, col = divmod(h, 3)
            ax = axes[row][col]
            
            gt_values = gt_day[f"Horizon_{h+1}"].values
            pred_values = pred_day[f"Horizon_{h+1}"].values
            
            ax.plot(hours, gt_values, label="Actual", linewidth=3)
            ax.plot(hours, pred_values, '--', label=f"Predicted (H{h+1})", linewidth=3)
            
            ax.set_title(f"Horizon {h+1}", fontsize=14)
            ax.set_xticks(hours)
            
            if row == 1:  # Only bottom row
                ax.set_xticklabels([f"{h}h" for h in hours], rotation=45, fontsize=11)
            else:
                ax.set_xticklabels([])  # Hide x labels on top row
            
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.legend(fontsize=12)


        # for h in range(6):
        #     row, col = divmod(h, 3)
        #     ax = axes[row][col]
            
        #     gt_values = gt_day[f"Horizon_{h+1}"].values
        #     pred_values = pred_day[f"Horizon_{h+1}"].values
            
        #     ax.plot(hours, gt_values, label="Actual", linewidth=3)
        #     ax.plot(hours, pred_values, '--', label=f"Predicted (H{h+1})", linewidth=3)
            
        #     ax.set_title(f"Horizon {h+1}", fontsize=14)
        #     ax.set_xticks(hours)
            
        #     ax.set_xticklabels([f"{h}h" for h in hours], rotation=45, fontsize=11)

        #     #ax.set_xticklabels(hour_labels, rotation=45, fontsize=11)
        #     ax.grid(True, linestyle=':', alpha=0.7)
        #     ax.legend(fontsize=12)

        fig.text(0.04, 0.5, "Power (kW)", va="center", rotation="vertical", fontsize=14, fontweight="bold")
        fig.text(0.5, 0.04, "Forecast Time (Hours)", ha="center", fontsize=14, fontweight="bold")
        
        title_24h = f"{model_name} Wind Power Forecast - 1 Day ({day_start.strftime('%d/%m/%Y %H:%M')} to {day_end.strftime('%d/%m/%Y %H:%M')})"
        fig.suptitle(title_24h, fontsize=16, fontweight="bold", y=0.95)

        plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
        plt.savefig(os.path.join(save_dir, f"{model_name}_24H_HORIZON_PLOT.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[SUCCESS] Saved 24h plot for {model_name}")
    else:
        print(f"[WARNING] No data available for last 24h period")

# === Save MAE CSV ===
mae_df = pd.DataFrame(mae_records)
mae_csv_path = os.path.join(save_dir, "PINN_HORIZON_IDEALGAS_PLOTS_MODELS_MAE_by_WINDOW.csv")
mae_df.to_csv(mae_csv_path, index=False)
print(f"\n[INFO] MAE saved to: {mae_csv_path}")