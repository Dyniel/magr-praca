# Generating Thesis Materials from WandB Runs

This project logs comprehensive information to Weights & Biases (WandB), which can be directly used to generate tables, plots, and qualitative results for your thesis.

## 1. Accessing WandB Runs

*   Navigate to your WandB project page (e.g., `SuperpixelGAN_Refactored` or `SuperpixelGAN_Sweeps`).
*   You will see a list of all your runs, including those from `scripts/run_experiments.py` and any hyperparameter sweeps.

## 2. Key Information Logged

For each run, the following crucial information should be available in the WandB dashboard:

*   **Hyperparameters:** The "Overview" tab or a dedicated "Config" section for each run lists all hyperparameters used for that run (from your YAML configs and any overrides). This is essential for documenting your experimental setup.
*   **Metrics over Time (Learning Curves):**
    *   Generator Losses (e.g., `train/Loss_G`, `val/Loss_G`)
    *   Discriminator Losses (e.g., `train/Loss_D`, `val/Loss_D_Adv`, `val/Loss_D_R1`)
    *   Feature Matching Losses (for ProjectedGAN, e.g., `train/Loss_G_FeatMatch`)
    *   Mean Discriminator Logits (e.g., `val/D_Real_Logits_Mean`, `val/D_Fake_Logits_Mean`)
    *   These are available in the "Charts" tab. You can customize plots, smooth lines, and compare multiple runs.
*   **FID Scores:**
    *   Fréchet Inception Distance scores (e.g., `val/FID_Score`) are logged if `enable_fid_calculation=True`. These are critical for quantitative comparison of model quality.
    *   You can view these in the "Charts" tab or create a summary table (see below).
*   **Generated Image Samples:**
    *   Image samples logged during training (e.g., `val/Generated_Samples`) can be found in the "Media" tab. These are vital for qualitative assessment. You can typically view them individually or as a panel, and download them.
*   **System Metrics:** WandB also logs GPU/CPU utilization, memory, etc., which might be relevant for discussing computational aspects.

## 3. Generating Materials for Thesis

### A. Tables

*   **Hyperparameter Tables:**
    *   Use the "Config" section in WandB for each run to manually create tables summarizing the key hyperparameters for different models or ablation studies.
*   **Quantitative Results Table (e.g., FID Scores):**
    1.  In the WandB project view, you can add columns to the runs table to display summary metrics like the minimum `val/FID_Score` achieved by each run.
    2.  Select the runs you want to compare (e.g., all "standard" runs for each architecture, or a model vs. its ablation).
    3.  Customize the columns to show: Run Name, Model Architecture (from config), Key Ablation Flags (from config), and the final or best FID score.
    4.  You can then export this table view as a CSV or take a screenshot.
    *Example Table Structure:*
    | Run Name                          | Architecture   | Ablation Details             | Best FID Score (val) |
    |-----------------------------------|----------------|------------------------------|----------------------|
    | `gan5_gcn_standard`               | `gan5_gcn`     | GCN Enabled                  | XX.XX                |
    | `gan5_gcn_ablation_no_gcn`        | `gan5_gcn`     | GCN Disabled                 | YY.YY                |
    | `stylegan2_standard_no_sp_cond` | `stylegan2`    | No Superpixel Cond.        | AA.AA                |
    | `stylegan2_conditioned_sp`      | `stylegan2`    | Superpixel Cond. Enabled   | BB.BB                |
    | ...                               | ...            | ...                          | ...                  |

### B. Plots (Learning Curves)

*   **Individual Run Analysis:**
    *   Go to a specific run in WandB.
    *   The "Charts" tab will show plots of all logged metrics over training steps/epochs.
    *   Customize these plots (e.g., add smoothing, change x-axis, focus on specific metrics like `val/Loss_G` or `val/FID_Score`).
    *   You can download these plots as PNG images directly from the WandB UI.
*   **Comparing Multiple Runs:**
    1.  From the project page, select multiple runs you want to compare.
    2.  Click "Group" if runs are not already grouped by a common characteristic (like architecture).
    3.  Go to the "Charts" panel. You can plot metrics from all selected runs on the same graph. For example, plot `val/FID_Score` for `stylegan2_standard` vs. `stylegan2_conditioned_sp`.
    4.  Customize and download these comparison plots.

### C. Qualitative Results (Generated Images)

*   **Best Samples:**
    *   For each model/run, go to the "Media" tab in WandB.
    *   Identify the epoch/step that produced the best looking samples (often correlates with the best FID score, but visual inspection is key).
    *   Download these high-quality samples. You can create panels or figures in your thesis to showcase the visual quality.
*   **Evolution of Samples:**
    *   WandB often allows you to scroll through samples logged at different timesteps. This can be used to illustrate how image quality improves over training.

### D. Hyperparameter Importance (from Sweeps)

*   If you run WandB Sweeps:
    *   The sweep results page in WandB provides plots showing parameter importance, parallel coordinate plots, and scatter plots of hyperparameters vs. the target metric (e.g., `val/FID_Score`).
    *   These visualizations can be directly used or adapted to discuss which hyperparameters had the most impact on model performance.

## 4. Using WandB API for Programmatic Access (Advanced)

If you need more customized tables or plots than the WandB UI provides:

*   **Install WandB library:** `pip install wandb`
*   **Use the API:** You can write Python scripts to fetch run data (configs, summary metrics, metric history) and then process it using libraries like Pandas (for tables) and Matplotlib/Seaborn (for plots).

```python
# Example: Fetching FID scores for selected runs
import wandb
api = wandb.Api()

# Replace with your entity, project, and desired run IDs or filters
runs = api.runs("YOUR_ENTITY/YOUR_PROJECT_NAME", filters={
    # "display_name": {"$regex": ".*standard.*"} # Example filter
})

fid_data = []
for run in runs:
    if "val/FID_Score" in run.summary: # Check if FID was logged
        fid_data.append({
            "run_name": run.name,
            "architecture": run.config.get("model", {}).get("architecture"),
            "best_fid": run.summary["val/FID_Score"] # Or min_fid if logged that way
            # Add other relevant config values
        })

import pandas as pd
df_fid = pd.DataFrame(fid_data)
print(df_fid)
# df_fid.to_csv("fid_summary.csv")
```

By leveraging these WandB features, you can efficiently gather and present the results of your experiments for your thesis. Remember to customize plots and tables to fit the specific narrative and requirements of your document.
