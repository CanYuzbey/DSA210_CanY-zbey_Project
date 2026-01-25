
import pandas as pd

def process_air_data(df: pd.DataFrame, state: str) -> pd.DataFrame:
    """
    Process air quality data.
    """
    df = df.copy()
    
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Year"] = df["Date"].dt.year
    elif "Year" not in df.columns:
         print(f"Warning: neither 'Date' nor 'Year' found for {state}")
         return pd.DataFrame()

    if "AQI" not in df.columns:
        print(f"Warning: 'AQI' column missing for {state}")
        return pd.DataFrame()

    df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    
    df = df.dropna(subset=["Year", "AQI"])
    df["Year"] = df["Year"].astype(int)
    df = df[df["Year"] >= 2010]
    
    yearly = df.groupby("Year")["AQI"].mean().reset_index()
    yearly.rename(columns={"AQI": "Avg_AQI"}, inplace=True)
    yearly["State"] = state
    
    return yearly

def process_cancer_data(df: pd.DataFrame, state: str) -> pd.DataFrame:
    """
    Process cancer incidence data.
    """
    df = df.copy()
    
    target_col = "Observed"
    if target_col not in df.columns:
        for col in df.columns:
            if ("Rate" in col) or ("Incidence" in col):
                target_col = col
                break
    
    # Check for Year
    if "Year" not in df.columns:
        # Some cancer files might be dirty, validation logic here
        return pd.DataFrame()
    
    if target_col not in df.columns:
        return pd.DataFrame()

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    
    df = df.dropna(subset=["Year", target_col])
    df["Year"] = df["Year"].astype(int)
    
    result = df[["Year", target_col]].copy()
    result.rename(columns={target_col: "Cancer_Rate"}, inplace=True)
    result["State"] = state
    
    return result

def merge_datasets(air_df: pd.DataFrame, cancer_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge air and cancer datasets on State and Year.
    """
    return pd.merge(air_df, cancer_df, on=["State", "Year"], how="inner")
