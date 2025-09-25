# import numpy as np
# from scipy.stats import ttest_rel

# # Example F1-scores across folds for two models
# prokdf  = np.array([0.78, 0.81, 0.76, 0.80, 0.79])
# baseline = np.array([0.72, 0.74, 0.70, 0.73, 0.71])

# # Paired t-test
# t_stat, p_value = ttest_rel(prokdf, baseline)

# print("t-statistic:", t_stat)
# print("p-value:", p_value)



import numpy as np
from scipy.stats import ttest_rel

# ========= Fold-wise results =========
# Baseline Model
baseline_acc = [60.74, 62.96, 60.74, 63.70, 71.11, 60.45, 72.39]
baseline_f1  = [65.81, 69.14, 68.26, 67.55, 75.16, 65.36, 76.13]
baseline_auc = [71.22, 69.14, 69.39, 68.80, 84.02, 68.46, 78.64]

# T Fusion Model
t_acc = [77.78, 82.96, 83.70, 81.48, 82.22, 78.36, 76.87]
t_f1  = [78.57, 81.89, 84.29, 80.62, 80.00, 75.63, 78.91]
t_auc = [83.55, 85.90, 90.00, 86.77, 90.55, 86.15, 88.71]

# DT Fusion Model
dt_acc = [81.48, 83.70, 90.37, 82.96, 86.67, 86.57, 85.82]
dt_f1  = [82.01, 84.72, 90.91, 82.96, 85.48, 86.96, 87.42]
dt_auc = [89.38, 91.45, 93.29, 90.23, 92.96, 89.49, 92.87]

# ProKDF Model
prokdf_acc = [84.44, 87.41, 91.11, 85.19, 89.63, 85.82, 85.82]
prokdf_f1  = [85.11, 87.94, 91.55, 85.29, 88.89, 86.33, 87.25]
prokdf_auc = [90.83, 91.99, 93.35, 90.63, 94.20, 90.37, 92.61]

# ========= Paired t-test function =========
def paired_ttest(model1, model2, name1, name2, metric_name):
    stat, p = ttest_rel(model1, model2)
    print(f"\n{metric_name}: {name1} vs {name2}")
    print(f"t-statistic = {stat:.3f}, p-value = {p:.5f}")
    if p < 0.05:
        print("✅ Statistically significant (p < 0.05)")
    else:
        print("❌ Not statistically significant (p >= 0.05)")

# ========= Run tests =========
# Example: ProKDF vs Baseline
paired_ttest(baseline_f1,  prokdf_f1,  "Baseline", "ProKDF", "F1 Score")
paired_ttest(t_f1,  prokdf_f1,  "T Fusion", "ProKDF", "F1 Score")
paired_ttest(dt_f1,  prokdf_f1,  "DT Fusion", "ProKDF", "F1 Score")


paired_ttest(baseline_acc,  prokdf_acc,  "Baseline", "ProKDF", "Accuracy")
paired_ttest(t_acc,  prokdf_acc,  "T Fusion", "ProKDF", "Accuracy")
paired_ttest(dt_acc,  prokdf_acc,  "DT Fusion", "ProKDF", "Accuracy")

# paired_ttest(baseline_acc, prokdf_acc, "Baseline", "ProKDF", "Accuracy")
# paired_ttest(baseline_f1,  prokdf_f1,  "Baseline", "ProKDF", "F1 Score")
# paired_ttest(baseline_auc, prokdf_auc, "Baseline", "ProKDF", "AUC")

# # You can add more comparisons (e.g., ProKDF vs T Fusion, ProKDF vs DT Fusion, etc.)
# paired_ttest(t_acc, prokdf_acc, "T Fusion", "ProKDF", "Accuracy")
# paired_ttest(dt_acc, prokdf_acc, "DT Fusion", "ProKDF", "Accuracy")
