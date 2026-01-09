# DSA210 – Air Pollution (AQI) and Lung Cancer Incidence  
**Exploratory Data Analysis and Correlation Study**

This project analyzes the relationship between **air pollution**, measured using **annual average Air Quality Index (AQI)**, and **lung cancer incidence rates** across selected U.S. states. The study follows a structured data science workflow including **data preprocessing**, **exploratory data analysis (EDA)**, and **statistical hypothesis testing**.

---

## Research Question

**Is there a statistically significant relationship between annual average air pollution (AQI) and lung cancer incidence rates across U.S. states and years?**

---

## Hypothesis

- **H₀ (Null Hypothesis):** There is no statistically significant linear relationship between annual average AQI and lung cancer incidence rates (ρ = 0).
- **H₁ (Alternative Hypothesis):** There is a statistically significant linear relationship between annual average AQI and lung cancer incidence rates (ρ ≠ 0).

Significance level: **α = 0.05**

---

## Data Sources

- **Air Quality Data:**  
  Environmental Protection Agency (EPA) – Air Quality System (AQS)  
  https://www.epa.gov/aqs

- **Cancer Incidence Data:**  
  Centers for Disease Control and Prevention (CDC) – U.S. Cancer Statistics  
  https://www.cdc.gov/cancer/uscs/

The analysis uses state-level annual summaries derived from these sources.

---

## Data Preprocessing

Before analysis, several preprocessing steps were applied:

- Column names were standardized and dates were parsed into year format
- AQI values and cancer incidence rates were converted to numeric values
- Rows with missing or invalid year/value entries were removed
- Air quality data was aggregated into **annual average AQI**
- Datasets were merged using **State** and **Year** as keys

These steps ensured consistency and reliability before conducting exploratory and statistical analysis.

---

## Exploratory Data Analysis (EDA)

### Distribution of Annual Average AQI

This histogram shows the distribution of annual average AQI values across all states and years. It provides an overview of pollution levels and helps identify variability and potential skewness.

![Distribution of Annual Average AQI](figures/eda_hist_aqi.png)

**Interpretation:**  
Most AQI values fall within moderate ranges, while higher AQI values appear less frequently and are primarily associated with California.

---

### Distribution of Lung Cancer Incidence Rate

This histogram presents the distribution of lung cancer incidence rates (per 100,000 people) across the dataset, highlighting the spread and variability of cancer rates.

![Distribution of Lung Cancer Incidence Rate](figures/eda_hist_cancer.png)

**Interpretation:**  
Cancer incidence rates vary considerably across states and years, with higher rates more common in densely populated regions.

---

### AQI by State (Boxplot)

This boxplot compares the distribution of annual average AQI values between states, highlighting differences in pollution levels and potential outliers.

![AQI by State](figures/eda_box_aqi_by_state.png)

**Interpretation:**  
California shows consistently higher AQI values compared to Washington and New York, indicating relatively poorer air quality during the analyzed period.

---

## Temporal Trends

### Annual Average AQI Over Time

This line plot shows how annual average AQI values change over time for each state, allowing comparison of pollution trends.

![Annual Average AQI Over Time](figures/eda_line_aqi_over_time.png)

**Interpretation:**  
Washington and New York display relatively stable AQI trends, while California maintains higher pollution levels with noticeable year-to-year variation.

---

### Lung Cancer Incidence Over Time

This line plot illustrates lung cancer incidence rates over time for each state.

![Lung Cancer Incidence Over Time](figures/eda_line_cancer_over_time.png)

**Interpretation:**  
All states show a general downward trend in lung cancer incidence, which may reflect long-term improvements in healthcare, reduced smoking rates, or reporting effects.

---

## Relationship Between AQI and Lung Cancer Incidence

### AQI vs Lung Cancer Incidence

This scatter plot examines the relationship between annual average AQI and lung cancer incidence rates across states and years. A linear regression line is included to visualize the overall trend.

![AQI vs Lung Cancer Incidence](figures/result_scatter_aqi_vs_cancer.png)

**Statistical Result:**  
- Pearson correlation coefficient: **r ≈ -0.70**  
- p-value: **p ≈ 0.00038**

**Interpretation:**  
The pooled dataset shows a statistically significant negative linear association between AQI and lung cancer incidence rates. This relationship reflects differences between states rather than a causal effect.

---

## Limitations

This study is based on aggregated state-level data and therefore represents an **ecological analysis**. The findings do not imply causation and do not account for important confounding factors such as:

- Smoking prevalence  
- Occupational exposure  
- Socioeconomic conditions  
- Healthcare access  
- Long latency periods associated with cancer development  

Additionally, the limited number of states and years reduces generalizability.

---

## How to Run the Analysis

This analysis was developed and executed using **Google Colab**, but it can also be run locally.

### Option 1: Run in Google Colab (Recommended)

1. Upload the repository files to a Google Colab session.
2. Upload the CSV data files (`Air.*.csv` and `Cancer.*.csv`) into the Colab working directory.
3. CSV data files are stored under the `data/air/` and `data/cancer/` directories.
4. Open and run the analysis notebook (`analysis.ipynb`) from top to bottom.
5. The script will automatically:
   - Clean and preprocess the data
   - Generate exploratory data analysis (EDA) plots
   - Perform correlation analysis with p-values
   - Save all figures into the `figures/` folder

### Option 2: Run Locally

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-folder>






  




