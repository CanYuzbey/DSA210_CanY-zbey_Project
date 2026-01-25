
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

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
    # 5) ML: Linear Regression (AQI + State)
    # -----------------------------
    print("Running Linear Regression...")
    X = df_final[["Avg_AQI", "State"]]
    y = df_final["Cancer_Rate"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pre = ColumnTransformer([
        ("num", "passthrough", ["Avg_AQI"]),
        ("cat", OneHotEncoder(drop="first"), ["State"]),
    ])

    model = Pipeline([
        ("pre", pre),
        ("lr", LinearRegression()),
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    print("\n=== ML RESULTS (Linear Regression: AQI + State) ===")
    print(f"R2   : {r2:.3f}")
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")

    # Save metrics
    metrics_df = pd.DataFrame([{"Model": "LR (Avg_AQI + State)", "R2": r2, "MAE": mae, "RMSE": rmse}])
    metrics_df.to_csv("figures/ml_metrics.csv", index=False)
    print("Saved metrics -> figures/ml_metrics.csv")

    # Plot predicted vs actual
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, pred)
    if not y_test.empty:
        minv = min(float(y_test.min()), float(pred.min()))
        maxv = max(float(y_test.max()), float(pred.max()))
        plt.plot([minv, maxv], [minv, maxv], linestyle="--")
    plt.xlabel("Actual Cancer Rate")
    plt.ylabel("Predicted Cancer Rate")
    plt.title("Predicted vs Actual (LR: AQI + State)")
    plt.tight_layout()
    plt.savefig("figures/ml_pred_vs_actual.png", dpi=200)
    # plt.show()
    print("Saved plot -> figures/ml_pred_vs_actual.png")

if __name__ == "__main__":
    main()