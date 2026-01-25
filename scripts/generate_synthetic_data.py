
import pandas as pd
import numpy as np
import os

# Ensure directories exist
os.makedirs("data/air", exist_ok=True)
os.makedirs("data/cancer", exist_ok=True)

states = ["California", "Florida", "New York", "Washington"]
years = range(2010, 2023)

# 1. Generate Air Quality Data
# Expected cols: Date (or Year), AQI. 
# Let's use Date to match the original script's primary check.
for state in states:
    print(f"Generating data for {state}...")
    
    # Generate daily data for each year
    dates = pd.date_range(start="2010-01-01", end="2022-12-31", freq="D")
    
    # Base AQI varies by state (approximate real-world relative differences)
    # CA: Higher pollution (wildfires etc) ~ 70
    # NY, FL: Moderate ~ 45-50
    # WA: Good ~ 35
    if state == "California":
        base_aqi = 70
    elif state == "New York":
        base_aqi = 50
    elif state == "Florida":
        base_aqi = 45
    else: # Washington
        base_aqi = 35
        
    # Introduce a slight downward trend (air quality improving over decade)
    # and random seasonal variation
    trend = np.linspace(0, -5, len(dates)) # Improves by 5 units over 13 years
    seasonality = 10 * np.sin(2 * np.pi * dates.dayofyear / 365)
    
    aqi_values = base_aqi + trend + seasonality + np.random.normal(0, 10, size=len(dates))
    aqi_values = np.maximum(aqi_values, 0) # No negative AQI
    
    df_air = pd.DataFrame({
        "Date": dates,
        "AQI": aqi_values,
        "State": state, # Extra col, harmless
        "Site ID": range(len(dates)) # Dummy col
    })
    
    # Save
    path = os.path.join("data", "air", f"Air.{state}.csv")
    df_air.to_csv(path, index=False)
    print(f"  -> Saved {path}")

# 2. Generate Cancer Data
# Expected cols: Year, Observed (or Rate/Incidence)
for state in states:
    
    # Cancer rates generally decreasing? Or correlated with AQI?
    # Let's make a slight correlation for the sake of the ML model finding something
    
    rates = []
    for year in years:
        # Base rate
        base_rate = 50
        if state == "Kentucky": base_rate = 70 # High
        if state == "Utah": base_rate = 30 # Low
        
        # Add trend (decreasing over time)
        trend = (year - 2010) * -0.5
        
        # Random noise
        noise = np.random.normal(0, 2)
        
        rate = base_rate + trend + noise
        rates.append(rate)
        
    df_cancer = pd.DataFrame({
        "Year": years,
        "Observed": rates,
        "State": state # Extra
    })
    
    # Add dummy header rows to mimic some real datasets that need skiprows?
    # The original script looked for "Year" and "Observed" in the line.
    # Let's just write a clean CSV first. The script handles standard CSVs too if header is found.
    
    path = os.path.join("data", "cancer", f"Cancer.{state}.csv")
    df_cancer.to_csv(path, index=False)
    print(f"  -> Saved {path}")
