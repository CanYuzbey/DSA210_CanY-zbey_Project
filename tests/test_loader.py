
import unittest
import pandas as pd
import os
import sys

# Add root to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.loader import load_csv
from src.preprocessing import process_air_data

class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        # Create a dummy CSV file
        self.test_csv = "test_data.csv"
        df = pd.DataFrame({
            "Date": ["2020-01-01", "2021-01-01"],
            "AQI": [50, 60]
        })
        df.to_csv(self.test_csv, index=False)

    def tearDown(self):
        if os.path.exists(self.test_csv):
            os.remove(self.test_csv)

    def test_load_csv(self):
        df = load_csv(self.test_csv)
        self.assertEqual(len(df), 2)
        self.assertIn("AQI", df.columns)

    def test_process_air_data(self):
        df = load_csv(self.test_csv)
        processed = process_air_data(df, "TestState")
        
        self.assertEqual(len(processed), 2)
        self.assertEqual(processed.iloc[0]["Avg_AQI"], 50.0)
        self.assertEqual(processed.iloc[0]["State"], "TestState")

if __name__ == "__main__":
    unittest.main()
