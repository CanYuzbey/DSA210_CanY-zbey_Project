
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class AdvancedModelTrainer:
    def __init__(self, df, target_col="Cancer_Rate", feature_cols=None):
        self.df = df
        self.target_col = target_col
        self.feature_cols = feature_cols if feature_cols else ["Avg_AQI", "State"]
        self.models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        self.results = {}
        self.best_model = None
        self._prepare_pipeline()

    def _prepare_pipeline(self):
        # Identify categorical and numerical columns
        categorical_features = ["State"]
        numerical_features = ["Avg_AQI"]

        # Preprocessing for categorical data
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

    def train_and_evaluate(self):
        X = self.df[self.feature_cols]
        y = self.df[self.target_col]
        
        results_list = []

        print(f"Training on {len(X)} samples with {5}-fold Cross-Validation...")

        for name, model in self.models.items():
            pipeline = Pipeline(steps=[('preprocessor', self.preprocessor),
                                       ('regressor', model)])
            
            # K-Fold Cross Validation
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_results = cross_validate(pipeline, X, y, cv=cv, 
                                      scoring=('r2', 'neg_mean_absolute_error', 'neg_mean_squared_error'),
                                      return_train_score=False)
            
            mean_r2 = cv_results['test_r2'].mean()
            mean_mae = -cv_results['test_neg_mean_absolute_error'].mean()
            mean_rmse = np.sqrt(-cv_results['test_neg_mean_squared_error'].mean())

            print(f"  > {name}: R2={mean_r2:.3f}, MAE={mean_mae:.3f}, RMSE={mean_rmse:.3f}")
            
            # Train on full dataset for final artifacts
            pipeline.fit(X, y)
            self.results[name] = {
                "model": pipeline,
                "R2": mean_r2,
                "MAE": mean_mae,
                "RMSE": mean_rmse
            }
            
            results_list.append({
                "Model": name,
                "CV_R2": mean_r2,
                "CV_MAE": mean_mae,
                "CV_RMSE": mean_rmse
            })

        self.metrics_df = pd.DataFrame(results_list)
        return self.metrics_df

    def plot_model_comparison(self, output_path="figures/ml_model_comparison.png"):
        plt.figure(figsize=(10, 6))
        # Melt for seaborn
        melted = self.metrics_df.melt(id_vars="Model", value_vars=["CV_R2"], var_name="Metric", value_name="Score")
        sns.barplot(data=melted, x="Model", y="Score", palette="viridis")
        plt.title("Model Performance Comparison (Cross-Validated R2)")
        plt.ylim(0, 1.0)
        plt.ylabel("R2 Score")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"Saved comparison plot -> {output_path}")

    def plot_feature_importance(self, model_name="Random Forest", output_path="figures/ml_feature_importance.png"):
        if model_name not in self.results:
            print(f"Model {model_name} not available for feature importance.")
            return

        pipeline = self.results[model_name]["model"]
        model = pipeline.named_steps['regressor']
        
        if not hasattr(model, 'feature_importances_'):
            print(f"{model_name} does not provide feature importance.")
            return

        # Get feature names from preprocessor
        feature_names = []
        # Numeric pass through
        feature_names.extend(["Avg_AQI"])
        # Categorical one-hot
        cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat']
        feature_names.extend(cat_encoder.get_feature_names_out(["State"]))

        importances = model.feature_importances_
        
        # Sort
        indices = np.argsort(importances)
        
        plt.figure(figsize=(10, 6))
        plt.title(f"Feature Importances ({model_name})")
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Relative Importance")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"Saved feature importance plot -> {output_path}")

    def plot_residuals(self, model_name="Random Forest", output_path="figures/ml_residuals.png"):
        if model_name not in self.results:
            return

        pipeline = self.results[model_name]["model"]
        X = self.df[self.feature_cols]
        y = self.df[self.target_col]
        
        preds = pipeline.predict(X)
        residuals = y - preds

        plt.figure(figsize=(8, 6))
        plt.scatter(preds, residuals, alpha=0.6)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel("Predicted Cancer Rate")
        plt.ylabel("Residuals (Actual - Predicted)")
        plt.title(f"Residual Plot ({model_name})")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"Saved residual plot -> {output_path}")
