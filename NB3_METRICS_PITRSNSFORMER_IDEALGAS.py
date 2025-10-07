import numpy as np
import pandas as pd
from scipy.stats import shapiro, kruskal, wilcoxon, kendalltau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tabulate import tabulate
import os
import sys
from datetime import datetime

# === Logging Setup ===
log_path = "/home/admin/BurningGH2/METRICS/PITRANSFORMER_METRICS_IDEALGAS.log"
os.makedirs(os.path.dirname(log_path), exist_ok=True)
sys.stdout = open(log_path, "w")
sys.stderr = sys.stdout

print(f"[INFO] All results logged at: {log_path}\n")

# === Paths ===
y_seq_path = "/home/admin/BurningGH2/PI_TRANSFORMER_WINDPOWER_IDEALGAS/y_test_seq_goal_idealgas.npy"
prediction_files = {
    "PI_TRANSFORMER": "/home/admin/BurningGH2/PI_TRANSFORMER_WINDPOWER_IDEALGAS/PI_Transformer_predictions_inv_h6_idealgas.csv",
    "TRANSFORMER": "/home/admin/BurningGH2/TRANSFORMER_WINDPOWER/Transformer_predictions_inv_h6_goal.csv"
   
}

# === Load ground truth ===
y_seq = np.load(y_seq_path)
power_matrix = y_seq[:, :, 0]
horizons = power_matrix.shape[1]

# === Metric function ===
def get_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    pearson_r = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
    kendall_tau, _ = kendalltau(y_true.flatten(), y_pred.flatten())
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.divide(y_pred, y_true, where=y_true != 0)
        fac2 = np.mean((ratio >= 0.5) & (ratio <= 2))
        nmse = mse / np.var(y_true)
        nmae = mae / np.mean(y_true)
        r2 = r2_score(y_true, y_pred)
        return [mse, rmse, mae, pearson_r, kendall_tau, fac2, nmse, nmae, r2]

# === Compute metrics ===
all_metrics = []
avg_metrics_table = []
model_mae_dict = {}
model_predictions = {}

for model_name, path in prediction_files.items():
    df = pd.read_csv(path)
    metrics_table = []
    mae_samples = []

    print(f"\n[METRICS] {model_name}\n{'='*50}")
    for h in range(horizons):
        y_pred = df[f"Horizon_{h+1}"].values
        y_true = power_matrix[:, h]
        y_pred = y_pred[:len(y_true)]
        y_true = y_true[:len(y_pred)]

        metrics = get_metrics(y_true, y_pred)
        metrics_table.append([f"Horizon {h+1}"] + metrics)
        all_metrics.append([model_name, f"Horizon_{h+1}"] + metrics)
        mae_samples.append(np.abs(y_true - y_pred))

    # Average metrics
    avg = np.mean(np.array([row[1:] for row in metrics_table]), axis=0)
    metrics_table.append(["Average"] + list(avg))
    all_metrics.append([model_name, "Average"] + list(avg))
    avg_metrics_table.append([model_name] + list(avg))

    # Store for statistical tests
    model_mae_dict[model_name] = np.concatenate(mae_samples)
    model_predictions[model_name] = df.iloc[:, 5].values if df.shape[1] > 1 else df.values.flatten()

    print(tabulate(metrics_table, headers=[
        "Horizon", "MSE", "RMSE", "MAE", "Pearson R",
        "Kendall Tau", "FAC2", "NMSE", "NMAE", "R2"
    ], tablefmt='fancy_grid'))

# === Save metrics to CSV ===
metrics_df = pd.DataFrame(all_metrics, columns=[
    "Model", "Horizon", "MSE", "RMSE", "MAE", "Pearson R", 
    "Kendall Tau", "FAC2", "NMSE", "NMAE", "R2"
])
metrics_save_path = "/home/admin/BurningGH2/PITRANSFORMER_METRICS_IDEALGAS.csv"
metrics_df.to_csv(metrics_save_path, index=False)
print(f"\n[INFO] Metrics saved to: {metrics_save_path}\n")

# === Summary of Averages ===
summary_df = pd.DataFrame(avg_metrics_table, columns=[
    "Model", "MSE", "RMSE", "MAE", "Pearson R", 
    "Kendall Tau", "FAC2", "NMSE", "NMAE", "R2"
])
print("\n[SUMMARY] Model-wise Average Metrics:")
print(tabulate(summary_df, headers='keys', tablefmt='fancy_grid'))

# === Save Summary of Averages to Excel ===
excel_save_path = "/home/admin/BurningGH2/PITRANSFORMER_METRICS_IDEALGAS.xlsx"
summary_df.to_excel(excel_save_path, index=False)
print(f"\n[INFO] Average summary saved to Excel at: {excel_save_path}\n")


# === Shapiro-Wilk Normality Test ===
shapiro_results = {}
for model, preds in model_predictions.items():
    np.random.seed(42)
    sample = preds if len(preds) <= 5000 else np.random.choice(preds, 5000, replace=False)
    stat, p = shapiro(sample)
    shapiro_results[model] = {"Statistic": stat, "P-Value": p}
shapiro_df = pd.DataFrame.from_dict(shapiro_results, orient='index')
print("\n[STATISTICS] Shapiro-Wilk Normality Test:")
print(tabulate(shapiro_df, headers='keys', tablefmt='fancy_grid'))

# === Kruskal-Wallis Test (using MAE) ===
kruskal_stat, kruskal_p = kruskal(*model_mae_dict.values())
print(f"\n[STATISTICS] Kruskal-Wallis Test on MAE:\nStatistic = {kruskal_stat:.4f}, p-value = {kruskal_p:.4e}")

# === Wilcoxon Signed-Rank Pairwise Test ===
models = list(model_predictions.keys())
p_values_matrix = pd.DataFrame(index=models, columns=models)

for i in range(len(models)):
    for j in range(i + 1, len(models)):
        m1, m2 = models[i], models[j]
        try:
            diffs = model_predictions[m1] - model_predictions[m2]
            stat, p_value = wilcoxon(diffs)
            p_values_matrix.loc[m1, m2] = p_value
            p_values_matrix.loc[m2, m1] = p_value
        except Exception:
            p_values_matrix.loc[m1, m2] = np.nan
            p_values_matrix.loc[m2, m1] = np.nan
    p_values_matrix.loc[models[i], models[i]] = '-'

# === Format and Display Wilcoxon Table ===
results_table = pd.DataFrame(columns=["REFERENCE MODEL"] + models)
for reference in models:
    row = [reference]
    for model in models:
        val = p_values_matrix.loc[reference, model]
        row.append(f'{val:.3e}' if isinstance(val, float) else '-')
    results_table.loc[len(results_table)] = row

print("\n[STATISTICS] Wilcoxon Signed-Rank Test Pairwise Comparison:")
print(tabulate(results_table, headers='keys', tablefmt='fancy_grid'))

print("\n[LOG COMPLETE]")
