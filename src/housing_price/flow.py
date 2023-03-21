import os

import mlflow
import mlflow.sklearn
from ingest_data import fetch_extract_housing_data, load_housing_data, split_dataset
from train import (
    df_X_y,
    feature_engineering,
    run_decision_tree,
    run_grid_search,
    run_linear_regression,
    run_random_forest,
)

os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000/"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "Housing Price Prediction"


if __name__ == "__main__":
    # experiment_id = mlflow.create_experiment("Housing Price Prediction")

    with mlflow.start_run(run_name="Parent_run", description="Parent") as parent_run:
        mlflow.log_param("parent", "yes")
        with mlflow.start_run(
            run_name="fetch_and_load_data",
            description="Fetch the data from URL and Save it in Local and Load the data",  # noqa:E501
            nested=True,
        ) as child_run1:
            fetch_extract_housing_data()
            housing = load_housing_data()
        with mlflow.start_run(
            run_name="split_data",
            description="Split Dataset to Test and Train",
            nested=True,
        ) as child_run2:
            train, test = split_dataset(housing)
        with mlflow.start_run(
            run_name="Feature Engineering",
            description="Perform Feature Engineering",
            nested=True,
        ) as child_run3:
            X_train, y_train = df_X_y(train, "median_house_value")
            df_train = feature_engineering(X_train, y_train)
            X_trans = df_train.to_numpy()
            mlflow.log_artifact("artifacts/feature_transformer.joblib")
        with mlflow.start_run(
            run_name="Linear Regression",
            description="Running Linear Regresion Model",
            nested=True,
        ) as child_run4:
            metrics_lr = run_linear_regression(X_trans, y_train)
            mlflow.log_artifact("artifacts/lin_reg.joblib")
            for metric in metrics_lr:
                mlflow.log_metric(metric, metrics_lr[metric])
        with mlflow.start_run(
            run_name="Decision Tree",
            description="Running Decision Tree Model",
            nested=True,
        ) as child_run5:
            metrics_dt = run_decision_tree(X_trans, y_train)
            mlflow.log_artifact("artifacts/dt_model.joblib")
            for metric in metrics_dt:
                mlflow.log_metric(metric, metrics_dt[metric])
        with mlflow.start_run(
            run_name="Random Forest",
            description="Running Random Forest Model",
            nested=True,
        ) as child_run6:
            metrics_rf = run_random_forest(X_trans, y_train)
            mlflow.log_artifact("artifacts/rf_model.joblib")
            for metric in metrics_rf:
                mlflow.log_metric(metric, metrics_rf[metric])
        with mlflow.start_run(
            run_name="GridSearch",
            description="Running grid search on random forest",
            nested=True,
        ) as child_run7:
            metrics_grid_search = run_grid_search(X_trans, y_train)
            mlflow.log_artifact("artifacts/final_model.joblib")
            for metric in metrics_grid_search:
                mlflow.log_metric(metric, metrics_grid_search[metric])
