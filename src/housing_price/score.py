import argparse
import logging
import os

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from logger_functions import configure_logger
from sklearn.metrics import mean_absolute_error, mean_squared_error
from train import CombinedAttributesAdder, df_X_y  # noqa:F401

DEFAULT_MODEL_FOLDER = "artifacts"
DEFAULT_DATA_FOLDER = "data/processed"
DEFAULT_OUTPUT_FOLDER = "artifacts/metrics"

logger = logging.getLogger(__name__)
os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000/"


def initialize_parser():
    """
    Parses command-line arguments so they can used in the code.

    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-data-folder",
        help="Specify input data folder",
        default=DEFAULT_DATA_FOLDER,
        type=str,
        required=False,
    )

    parser.add_argument(
        "--input-model-folder",
        help="Specify input model folder",
        default=DEFAULT_MODEL_FOLDER,
        type=str,
        required=False,
    )

    parser.add_argument(
        "--output-folder",
        help="Specify output folder",
        default=DEFAULT_OUTPUT_FOLDER,
        type=str,
        required=False,
    )

    parser.add_argument(
        "--log-level",
        help="Logger level default: %(default)s",
        default="DEBUG",
        choices=["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"],
        required=False,
    )

    parser.add_argument(
        "--log-path", help="Path of the logger file", type=str, required=False
    )

    parser.add_argument(
        "--no-console-log",
        help="Print to console default: %(default)s",
        default=True,
        action="store_false",
    )

    args = parser.parse_args()

    if args.output_folder is not None:
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)

    return args


def driver_score():
    """
    Driver code to run the score script

        1) Reads the train and test datasets
        2) Transform the datasets using saved transformer
        3) Create metrics dataframe for test and train transformed datasets for differerent metrics ('mse', 'rmse', 'mae')

    """  # noqa:E501

    global logger
    experiment_id = mlflow.create_experiment("Scoring of trained model")

    with mlflow.start_run(
        run_name="Parent_run",
        experiment_id=experiment_id,
        description="Scoring of trained model",
    ):
        args = initialize_parser()
        logger = configure_logger(
            logger=logger,
            log_file=args.log_path,
            console=args.no_console_log,
            log_level=args.log_level,
        )
        logger.info("Starting Scoring")

        logger.info("Started reading train and test datasets")
        train_df = pd.read_csv(args.input_data_folder + "/train.csv")
        test_df = pd.read_csv(args.input_data_folder + "/test.csv")

        X_train, y_train = df_X_y(train_df, "median_house_value")
        X_test, y_test = df_X_y(test_df, "median_house_value")

        logger.info("Loading trained feature transformer")
        feature_transformer = joblib.load(
            args.input_model_folder + "/feature_transformer.joblib"
        )

        X_train_trans = feature_transformer.transform(X_train)
        X_test_trans = feature_transformer.transform(X_test)

        metrics_df = pd.DataFrame(columns=["mse", "rmse", "mae"])

        logger.info("Loading trained model")
        final_model = joblib.load(args.input_model_folder + "/final_model.joblib")
        mlflow.log_artifact(args.input_model_folder + "/final_model.joblib")

        y_train_pred = final_model.predict(X_train_trans)
        y_test_pred = final_model.predict(X_test_trans)

        metrics_df.loc["train", "mse"] = mean_squared_error(y_train, y_train_pred)
        metrics_df.loc["train", "rmse"] = np.sqrt(
            mean_squared_error(y_train, y_train_pred)
        )
        metrics_df.loc["train", "mae"] = mean_absolute_error(y_train, y_train_pred)
        metrics_df.loc["test", "mse"] = mean_squared_error(y_test, y_test_pred)
        metrics_df.loc["test", "rmse"] = np.sqrt(
            mean_squared_error(y_test, y_test_pred)
        )
        metrics_df.loc["test", "mae"] = mean_absolute_error(y_test, y_test_pred)
        logger.info(f"Metrics from the trained model: \n{metrics_df}")

        metrics_df.reset_index().to_csv(
            args.output_folder + "/test_metrics.csv", index=False
        )
        mlflow.log_artifact(args.output_folder + "/test_metrics.csv")
        logger.info("Saved the metrics to a file")


if __name__ == "__main__":
    driver_score()
