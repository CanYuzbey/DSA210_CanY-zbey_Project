
import pandas as pd
import os

def load_csv(path: str, **kwargs) -> pd.DataFrame:
    """
    Load a CSV file from the given path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    
    try:
        df = pd.read_csv(path, **kwargs)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        raise IOError(f"Error reading {path}: {e}")

def find_header_row(path: str, keywords: list) -> int:
    """
    Find the row index containing all keywords.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            if all(keyword in line for keyword in keywords):
                return i
    except Exception:
        pass
    return 0
