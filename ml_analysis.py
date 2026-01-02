import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ---------------------------------------------------------
# 1. FILE CHECK (Verify if files exist)
# ---------------------------------------------------------
air_files = {
    'Florida': 'Air.Florida.csv',
    'Washington': 'Air.Washington.csv',
    'New York': 'Air.NewYork.csv',
    'California': 'Air.California.csv'
}

cancer_files = {
    'Florida': 'Cancer.Florida.csv',
    'Washington': 'Cancer.Washington.csv',
    'New York': 'Cancer.NewYork.csv',
    'California': 'Cancer.California.csv'
}

print("📂 Checking files...")
for name, f in {**air_files, **cancer_files}.items():
    if not os.path.exists(f):
        print(f"⚠️ WARNING: File '{f}' not found! Please make sure you uploaded it to the Colab folder.")

# ---------------------------------------------------------
# 2. PREPARE AIR QUALITY DATA (Robust Mode)
# ---------------------------------------------------------
print("\n🔄 Processing Air Quality Data...")
air_data_frames = []

for state, filename in air_files.items():
    if os.path.exists(filename):
        try:
            # Read CSV, skipping bad lines that cause errors
            df = pd.read_csv(filename, on_bad_lines='skip')

            # Clean column names (remove whitespaces)
            df.columns = df.columns.str.strip()

            # Fix Date format
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df['Year'] = df['Date'].dt.year
            elif 'Year' in df.columns:
                 pass

            # Convert AQI to numeric (force invalid text to NaN)
            df['AQI'] = pd.to_numeric(df['AQI'], errors='coerce')

            # Drop rows with missing values
            df = df.dropna(subset=['Year', 'AQI'])
            df['Year'] = df['Year'].astype(int)

            # Filter for years 2010 and later
            df = df[df['Year'] >= 2010]

            # Calculate Annual Average
            df_yearly = df.groupby('Year')['AQI'].mean().reset_index()
            df_yearly.rename(columns={'AQI': 'Avg_AQI'}, inplace=True) # Rename for clarity
            df_yearly['State'] = state

            air_data_frames.append(df_yearly)
            print(f"   ✅ {state} Air Data OK. ({len(df_yearly)} years)")
        except Exception as e:
            print(f"   ❌ Error in {state} Air Data: {e}")

if air_data_frames:
    df_air_final = pd.concat(air_data_frames, ignore_index=True)
else:
    print("❌ NO AIR DATA COULD BE LOADED.")

# ---------------------------------------------------------
# 3. PREPARE CANCER DATA (Robust Mode)
# ---------------------------------------------------------
print("\n🔄 Processing Cancer Data...")
cancer_data_frames = []

for state, filename in cancer_files.items():
    if os.path.exists(filename):
        try:
            # Detect header row automatically
            with open(filename, 'r') as f:
                lines = f.readlines()

            header_row = 0
            for i, line in enumerate(lines):
                # Find line with both "Year" and "Observed"
                if "Year" in line and "Observed" in line:
                    header_row = i
                    break

            # Read from the detected header row
            df = pd.read_csv(filename, skiprows=header_row, on_bad_lines='skip')
            df.columns = df.columns.str.strip()

            # Identify the rate column (Observed, Rate, or Incidence)
            target_col = 'Observed'
            if 'Observed' not in df.columns:
                for col in df.columns:
                    if 'Rate' in col or 'Incidence' in col:
                        target_col = col
                        break

            # Convert data to numeric
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
            df[target_col] = pd.to_numeric(df[target_col], errors='coerce')

            # Clean data
            df = df.dropna(subset=['Year', target_col])
            df['Year'] = df['Year'].astype(int)

            # Select and rename columns
            df = df[['Year', target_col]]
            df.rename(columns={target_col: 'Cancer_Rate'}, inplace=True)
            df['State'] = state

            cancer_data_frames.append(df)
            print(f"   ✅ {state} Cancer Data OK. ({len(df)} years)")
        except Exception as e:
            print(f"   ❌ Error in {state} Cancer Data: {e}")

if cancer_data_frames:
    df_cancer_final = pd.concat(cancer_data_frames, ignore_index=True)
else:
    print("❌ NO CANCER DATA COULD BE LOADED.")

# ---------------------------------------------------------
# 4. MERGE AND PLOT (Final Analysis)
# ---------------------------------------------------------
if air_data_frames and cancer_data_frames:
    print("\n🔄 Merging Datasets...")
    # Merge on State and Year
    df_final = pd.merge(df_air_final, df_cancer_final, on=['State', 'Year'], how='inner')

    if len(df_final) > 0:
        print(f"   📊 SUCCESS! Total {len(df_final)} rows ready for analysis.\n")
        print(df_final.head())

        # PLOT
        plt.figure(figsize=(12, 7))
        sns.scatterplot(data=df_final, x='Avg_AQI', y='Cancer_Rate', hue='State', style='State', s=150)

        # Trend Line
        sns.regplot(data=df_final, x='Avg_AQI', y='Cancer_Rate', scatter=False, color='gray', line_kws={'linestyle':'--'})

        plt.title('Air Pollution (AQI) vs Lung Cancer Incidence', fontsize=15)
        plt.xlabel('Average Air Quality Index (AQI)\n(Higher values = More Pollution)', fontsize=12)
        plt.ylabel('Lung Cancer Rate (per 100,000 people)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Calculate Correlation
        corr = df_final['Avg_AQI'].corr(df_final['Cancer_Rate'])
        plt.figtext(0.15, 0.85, f"Correlation: {corr:.2f}", fontsize=12, bbox={"boxstyle":"round", "fc":"white"})

        plt.tight_layout()
        plt.show()

        # FINAL INTERPRETATION
        print("\n" + "="*40)
        print(f"ANALYSIS RESULT (Correlation: {corr:.4f})")
        print("="*40)

        if corr > 0.3:
            print("CONCLUSION: Positive Correlation.")
            print("Explanation: The data suggests that higher air pollution (AQI) is associated with higher lung cancer rates.")
        elif corr < -0.3:
            print("CONCLUSION: Negative Correlation.")
            print("Explanation: The data shows an inverse relationship. This might be due to time-lag effects (cancer takes decades to develop) or other factors.")
        else:
            print("CONCLUSION: No Strong Correlation.")
            print("Explanation: There is no clear linear relationship between AQI and cancer rates in this specific dataset.")

    else:
        print("❌ ERROR: Datasets merged but no common years found! Check if years overlap (e.g. 2010-2022).")
else:
    print("❌ CRITICAL ERROR: Could not create one of the datasets.")
