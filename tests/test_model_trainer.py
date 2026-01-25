
import unittest
import pandas as pd
import os
import shutil
from src.model_trainer import AdvancedModelTrainer

class TestAdvancedModelTrainer(unittest.TestCase):
    def setUp(self):
        # Create dummy data
        self.df = pd.DataFrame({
            "Avg_AQI": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "State": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            "Cancer_Rate": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        self.test_fig_dir = "test_figures"
        os.makedirs(self.test_fig_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_fig_dir):
            shutil.rmtree(self.test_fig_dir)

    def test_training_flow(self):
        trainer = AdvancedModelTrainer(self.df)
        metrics = trainer.train_and_evaluate()
        
        # Check if metrics dataframe has rows for all models
        self.assertIn("Linear Regression", metrics["Model"].values)
        self.assertIn("Random Forest", metrics["Model"].values)
        self.assertIn("Gradient Boosting", metrics["Model"].values)
        
        # Check columns
        expected_cols = ["CV_R2", "CV_MAE", "CV_RMSE"]
        for col in expected_cols:
            self.assertIn(col, metrics.columns)

    def test_plotting(self):
        trainer = AdvancedModelTrainer(self.df)
        trainer.train_and_evaluate()
        
        comp_path = os.path.join(self.test_fig_dir, "comp.png")
        feat_path = os.path.join(self.test_fig_dir, "feat.png")
        res_path = os.path.join(self.test_fig_dir, "res.png")
        
        # Test plot generation
        trainer.plot_model_comparison(output_path=comp_path)
        self.assertTrue(os.path.exists(comp_path))
        
        trainer.plot_feature_importance(output_path=feat_path)
        self.assertTrue(os.path.exists(feat_path))
        
        trainer.plot_residuals(output_path=res_path)
        self.assertTrue(os.path.exists(res_path))

if __name__ == "__main__":
    unittest.main()
