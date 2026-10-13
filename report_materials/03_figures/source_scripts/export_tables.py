import os
import pandas as pd

TABLES_DIR = "/home/subhash/projects/LyASC/report_materials/03_figures/tables"

# 1. Base Model Table 1 Reproduction
t1_df = pd.read_csv('/home/subhash/projects/LyASC/base_model/results/tables/reproduced_table1_FINAL.csv')
t1_csv_path = os.path.join(TABLES_DIR, "table1_reproduction.csv")
t1_tex_path = os.path.join(TABLES_DIR, "table1_reproduction.tex")
t1_df.to_csv(t1_csv_path, index=False)
with open(t1_tex_path, "w") as f:
    f.write(t1_df.to_latex(index=False, caption="Base Model Reproduction Results (CIFAR-10, Depth 110, n=3 seeds)", label="tab:base_table1"))

# 2. Stress Rescue Table 3
t3_data = [
    {"Variant": "0.5x Skip Scale", "Base_Paper_Fail (%)": 37.14, "LyASC_Rescue_Mean (%)": 8.38, "n_seeds": 3, "LyASC_Per_Seed_Errors (%)": "[7.98, 8.84, 8.31]", "Delta_Rescue (pp)": -28.76},
    {"Variant": "2.0x Residual Scale", "Base_Paper_Fail (%)": 37.59, "LyASC_Rescue_Mean (%)": 8.44, "n_seeds": 3, "LyASC_Per_Seed_Errors (%)": "[8.13, 8.81, 8.38]", "Delta_Rescue (pp)": -29.15},
    {"Variant": "3.0x Residual Scale", "Base_Paper_Fail (%)": 49.27, "LyASC_Rescue_Mean (%)": 21.88, "n_seeds": 5, "LyASC_Per_Seed_Errors (%)": "[27.90, 32.73, 19.12, 20.88, 8.75]", "Delta_Rescue (pp)": -27.39}
]
t3_df = pd.DataFrame(t3_data)
t3_csv_path = os.path.join(TABLES_DIR, "table3_stress_rescue.csv")
t3_tex_path = os.path.join(TABLES_DIR, "table3_stress_rescue.tex")
t3_df.to_csv(t3_csv_path, index=False)
with open(t3_tex_path, "w") as f:
    f.write(t3_df.to_latex(index=False, caption="Phase E1 LyASC Stress Rescue vs Base Paper Failures (Depth 110)", label="tab:stress_rescue"))

# 3. Master Experiment Summary
master_data = [
    {"Phase": "Phase A/B", "Variant": "1xSkip (Plain)", "n": 3, "Mean_Error (%)": 6.27, "Spread": "±0.04%", "Comparison_Target": "6.31% (Paper)"},
    {"Phase": "Phase A/B", "Variant": "2rSkip+LN", "n": 3, "Mean_Error (%)": 6.15, "Spread": "±0.09%", "Comparison_Target": "6.02% (Paper Best)"},
    {"Phase": "Phase A/B", "Variant": "wSkip+LN", "n": 3, "Mean_Error (%)": 8.16, "Spread": "±0.13%", "Comparison_Target": "8.66% (Paper)"},
    {"Phase": "Phase C", "Variant": "LyASC noG", "n": 3, "Mean_Error (%)": 9.66, "Spread": "±0.73%", "Comparison_Target": "6.27% (1xSkip)"},
    {"Phase": "Ablation 1", "Variant": "LyASC+LN", "n": 1, "Mean_Error (%)": 7.22, "Spread": "N/A", "Comparison_Target": "7.17% (Phase D)"},
    {"Phase": "Phase D", "Variant": "LyASC+LN (None)", "n": 3, "Mean_Error (%)": 7.17, "Spread": "±0.21%", "Comparison_Target": "Selected Baseline"},
    {"Phase": "Phase D", "Variant": "LyASC+LN (Damped)", "n": 3, "Mean_Error (%)": 7.46, "Spread": "±0.31%", "Comparison_Target": "7.17% (None)"},
    {"Phase": "Phase D", "Variant": "LyASC+LN (Warmup)", "n": 2, "Mean_Error (%)": 7.95, "Spread": "±0.21%", "Comparison_Target": "7.17% (None)"},
    {"Phase": "Phase E1", "Variant": "LyASC Stress 0.5x", "n": 3, "Mean_Error (%)": 8.38, "Spread": "±0.43%", "Comparison_Target": "37.14% (Base Fail)"},
    {"Phase": "Phase E1", "Variant": "LyASC Stress 2.0x", "n": 3, "Mean_Error (%)": 8.44, "Spread": "±0.34%", "Comparison_Target": "37.59% (Base Fail)"},
    {"Phase": "Phase E1", "Variant": "LyASC Stress 3.0x", "n": 5, "Mean_Error (%)": 21.88, "Spread": "±9.15%", "Comparison_Target": "49.27% (Base Fail)"},
    {"Phase": "Phase E2", "Variant": "LyASC noG", "n": 2, "Mean_Error (%)": 9.52, "Spread": "±0.98%", "Comparison_Target": "7.17% (Reused Ph C)"},
    {"Phase": "Phase E2", "Variant": "LyASC+BN", "n": 2, "Mean_Error (%)": 6.34, "Spread": "±0.01%", "Comparison_Target": "7.17% (LyASC+LN)"}
]
master_df = pd.DataFrame(master_data)
master_csv_path = os.path.join(TABLES_DIR, "master_summary.csv")
master_tex_path = os.path.join(TABLES_DIR, "master_summary.tex")
master_df.to_csv(master_csv_path, index=False)
with open(master_tex_path, "w") as f:
    f.write(master_df.to_latex(index=False, caption="Master Experiment Summary Across All Project Phases", label="tab:master_summary"))

print("All figure tables (CSV and LaTeX) exported successfully to tables/!")
