from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd
import mlflow
import mlflow.sklearn
import os

os.environ["MLFLOW_TRACKING_USERNAME"] = "rahultomer218"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "3a408d777ea261bafea97717a9c78551fb61a90f"  # naya token daalo
mlflow.set_tracking_uri("https://dagshub.com/rahultomer218/ghughru-dags.mlflow")

data = load_breast_cancer()
x = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42)
param_grid = {
    "n_estimators": [10, 50, 100],
    "max_depth": [None, 10, 20],
}

mlflow.set_experiment("breast-cancer-rf-grid-search")

with mlflow.start_run():
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(x_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)

    # ✅ Log params and metrics
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_score", grid_search.best_score_)

    # ✅ Fixed: mlflow.data.from_pandas() inside the block
    train_df = x_train.copy()
    train_df["target"] = y_train.values
    train_dataset = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_dataset, "training")

    test_df = x_test.copy()
    test_df["target"] = y_test.values
    test_dataset = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_dataset, "testing")

    # ✅ Log the file
    mlflow.log_artifact(__file__)

    # ✅ Log best model
    mlflow.sklearn.log_model(grid_search.best_estimator_, "best_random_forest_model")

    # ✅ Set tags
    mlflow.set_tag("model_type", "RandomForestClassifier")
    mlflow.set_tag("dataset", "breast_cancer")
    mlflow.set_tag("author", "rahultomer218")

    # ✅ Fixed typo
    print("Best Params:", grid_search.best_params_)