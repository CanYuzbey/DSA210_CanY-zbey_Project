
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.loader import load_csv, find_header_row
from src.preprocessing import process_air_data, process_cancer_data, merge_datasets

def main():
    # -----------------------------
    # 1) File lists (root-level CSVs)
    # -----------------------------
    air_files = {
        "California": os.path.join("data", "air", "Air.California.csv"),
        "Florida": os.path.join("data", "air", "Air.Florida.csv"),
        "New York": os.path.join("data", "air", "Air.NewYork.csv"),
        "Washington": os.path.join("data", "air", "Air.Washington.csv"),
    }

    cancer_files = {
        "California": os.path.join("data", "cancer", "Cancer.California.csv"),
        "Florida": os.path.join("data", "cancer", "Cancer.Florida.csv"),
        "New York": os.path.join("data", "cancer", "Cancer.NewYork.csv"),
        "Washington": os.path.join("data", "cancer", "Cancer.Washington.csv"),
    }

    os.makedirs("figures", exist_ok=True)

    # -----------------------------
    # 2) Process Air Data
    # -----------------------------
    air_frames = []
    print("Processing Air Data...")
    for state, path in air_files.items():
        try:
            df = load_csv(path, on_bad_lines="skip")
            processed = process_air_data(df, state)
            if not processed.empty:
                air_frames.append(processed)
        except Exception as e:
            print(f"Skipping {state} air data: {e}")

    if not air_frames:
        print("No air data processed.")
        return
        
    df_air = pd.concat(air_frames, ignore_index=True)

    # -----------------------------
    # 3) Process Cancer Data
    # -----------------------------
    cancer_frames = []
    print("Processing Cancer Data...")
    for state, path in cancer_files.items():
        try:
            # find header row logic extracted to loader
            header_row = find_header_row(path, ["Year", "Observed"])
            if header_row == 0:
                 pass

            df = load_csv(path, skiprows=header_row, on_bad_lines="skip")
            processed = process_cancer_data(df, state)
            if not processed.empty:
                cancer_frames.append(processed)
        except Exception as e:
             print(f"Skipping {state} cancer data: {e}")

    if not cancer_frames:
        print("No cancer data processed.")
        return

    df_cancer = pd.concat(cancer_frames, ignore_index=True)

    # -----------------------------
    # 4) Merge
    # -----------------------------
    print("Merging datasets...")
    df_final = merge_datasets(df_air, df_cancer)
    
    if df_final.empty:
        raise ValueError("Merged dataset is empty. Check overlapping years.")

    print(f"Merged rows: {len(df_final)}")
    # print(df_final.head())


    # -----------------------------
    # 5) Advanced ML Pipeline
    # -----------------------------
    print("Running Advanced ML Pipeline...")
    from src.model_trainer import AdvancedModelTrainer
    
    trainer = AdvancedModelTrainer(df_final)
    metrics_df = trainer.train_and_evaluate()
    
    # Save Metrics
    metrics_df.to_csv("figures/ml_metrics.csv", index=False)
    print("Saved metrics -> figures/ml_metrics.csv")
    
    # Generate Plots
    trainer.plot_model_comparison()
    trainer.plot_feature_importance(model_name="Random Forest")
    trainer.plot_residuals(model_name="Random Forest")
    trainer.plot_residuals(model_name="Linear Regression", output_path="figures/ml_residuals_linreg.png")

if __name__ == "__main__":
    main()