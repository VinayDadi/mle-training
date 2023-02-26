import argparse
import logging
import os

import joblib
import numpy as np
import pandas as pd
from logger_functions import configure_logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

DEFAULT_INPUT_FOLDER = "data/processed"
DEFAULT_OUTPUT_FOLDER = "artifacts"

logger = logging.getLogger(__name__)


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """
    Custom Transformer to create derived columns `rooms_per_household`, `bedrooms_per_room`, `population_per_household`.

    Parameters
    ----------
    add_bedrooms_per_room: Bool
        Flag whether to add `bedrooms_per_room` to the array.
    cols_to_index_mapper: Dict
        key-value pair where keys are columns and values are it's indexes.

    """  # noqa:E501

    def __init__(
        self,
        add_bedrooms_per_room=True,
        cols_to_index_mapper={
            "rooms": 3,
            "bedrooms": 4,
            "population": 5,
            "households": 6,
        },
    ):
        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.cols_to_index_mapper = cols_to_index_mapper

    def fit(self, X, y=None):
        """
        Fit the transformer.

        Parameters
        ----------
        X: array
            Input array passed
        y: array
            Input target array
        """
        return self

    def transform(self, X):
        """
        Transforms the data.

        Parameters
        ----------
        X: array
            Data to transform.

        Returns
        -------
        Array
            Returns transformed array with added derived columns

        """
        rooms_per_household = (
            X[:, self.cols_to_index_mapper["rooms"]]
            / X[:, self.cols_to_index_mapper["households"]]
        )
        population_per_household = (
            X[:, self.cols_to_index_mapper["population"]]
            / X[:, self.cols_to_index_mapper["households"]]
        )
        if self.add_bedrooms_per_room:
            bedrooms_per_room = (
                X[:, self.cols_to_index_mapper["bedrooms"]]
                / X[:, self.cols_to_index_mapper["rooms"]]
            )
            return np.c_[
                X, rooms_per_household, population_per_household, bedrooms_per_room
            ]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


def df_X_y(df, target):
    """
    Converts the input dataframe into feature dataframe and target array based on target column name.

    Parameters
    ----------
    df: Dataframe
        Dataframe to be divided
    target: str
        Name of the target column in the dataframe

    Returns
    -------
    Dataframe, array
        Returns the divided feature dataframe and target array

    """  # noqa:E501

    logger.info("Splitting the dataset into array and target")
    X = df.drop(target, axis=1)
    y = df[target]
    return X, y


def feature_engineering(X, y, output_folder=DEFAULT_OUTPUT_FOLDER):
    """
    Transforms feature dataframe

        1) Converts categorical columns into dummy variables
        2) Imputes missing values in numerical columns
        3) Adds some derived numerical columns using custom transformer class.
        4) Scales the numerical columns.

    and saves the feature transformer in output_folder.

    Parameters
    ----------
    X: Dataframe
        Feature Dataframe
    y: Array
        Target array
    output_folder: str
        Path to save the feature transformer

    Returns
    -------
    Dataframe
        Returns transformed dataframe
    """

    logger.info("Started feature engineering")
    num_cols = X.select_dtypes("number").columns
    cat_cols = X.select_dtypes("object").columns

    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("attribs_adder", CombinedAttributesAdder()),
            ("std_scaler", StandardScaler()),
        ]
    )
    features_transformer = ColumnTransformer(
        [
            ("cat_enc", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            # ("imputer", SimpleImputer(strategy="median"), num_cols), #deleted
            ("num", num_pipeline, num_cols),  # added
        ]
    )
    arr = features_transformer.fit_transform(X, y)

    cat_cols = list(features_transformer.transformers_[0][1].get_feature_names_out())
    num_cols = list(num_cols) + [
        "rooms_per_household",
        "population_per_household",
        "bedrooms_per_room",
    ]
    total_cols = cat_cols + num_cols
    df = pd.DataFrame(arr, columns=total_cols)
    joblib.dump(features_transformer, output_folder + "/feature_transformer.joblib")
    logger.info("Saved transformer file")
    logger.info("Completed feature engineering")
    return df


def create_metrics_df(models, X, y):
    """
    Creates a metrics dataframe for all some algorithms

    Parameters
    ----------
    models: list of str
        List of models. Accepted strings: "lr", "dt", "rf". Ignores remaining provided strings.
    X: array
        Transformed feature array
    y: array
        target array

    Returns
    -------
    None

    """  # noqa:E501

    logger.info("Started calculating metrics for given models")

    metrics_df = pd.DataFrame(columns=["mse", "rmse", "mae"])

    if "lr" in models:
        lin_reg = LinearRegression()
        lin_reg.fit(X, y)
        y_pred = lin_reg.predict(X)
        metrics_df.loc["lr", "mse"] = round(mean_squared_error(y, y_pred), 3)
        metrics_df.loc["lr", "rmse"] = round(np.sqrt(mean_squared_error(y, y_pred)), 3)
        metrics_df.loc["lr", "mae"] = round(mean_absolute_error(y, y_pred), 3)
    if "dt" in models:
        dt_model = DecisionTreeRegressor(random_state=42)
        dt_model.fit(X, y)
        y_pred = dt_model.predict(X)
        metrics_df.loc["dt", "mse"] = round(mean_squared_error(y, y_pred), 3)
        metrics_df.loc["dt", "rmse"] = round(np.sqrt(mean_squared_error(y, y_pred)), 3)
        metrics_df.loc["dt", "mae"] = round(mean_absolute_error(y, y_pred), 3)
    if "rf" in models:
        rf_model = RandomForestRegressor(random_state=42)
        rf_model.fit(X, y)
        y_pred = rf_model.predict(X)
        metrics_df.loc["rf", "mse"] = round(mean_squared_error(y, y_pred), 3)
        metrics_df.loc["rf", "rmse"] = round(np.sqrt(mean_squared_error(y, y_pred)), 3)
        metrics_df.loc["rf", "mae"] = round(mean_absolute_error(y, y_pred), 3)

    logger.info(f"Calculated metrics for given models: \n{metrics_df}")


def initialize_parser():
    """
    Parses command-line arguments so they can used in the code.

    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-folder",
        help="Specify input folder",
        default=DEFAULT_INPUT_FOLDER,
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


def driver_train():
    """
    Driver code to run the train script

        1) Creates feature engineered dataframe and saves the feature transformer
        2) Run gridsearch on an estimator to get best model parameters and saves that model.

    """  # noqa:E501

    global logger
    args = initialize_parser()
    logger = configure_logger(
        logger=logger,
        log_file=args.log_path,
        console=args.no_console_log,
        log_level=args.log_level,
    )
    logger.info("Started train driver")

    logger.info("Started reading train dataset")
    df = pd.read_csv(args.input_folder + "/train.csv")
    X, y = df_X_y(df, "median_house_value")

    if not os.path.exists(args.output_folder):
        logger.info(f'Directory "{args.output_folder}" not found so creating the same')
        os.makedirs(args.output_folder)

    df = feature_engineering(X, y, args.output_folder)

    X_trans = df.to_numpy()

    create_metrics_df(["lr", "dt", "rf"], X_trans, y)

    logger.info("Started GridSearchCV")
    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]
    rf_model = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(
        rf_model,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(X_trans, y)
    logger.info(f"Used grid: {param_grid}")
    logger.info("GridSearchCV Completed")
    logger.info(f"Best Params for the given grid: {grid_search.best_params_}")
    logger.info(f"Best Score for the given grid: {grid_search.best_score_}")

    feature_importances = grid_search.best_estimator_.feature_importances_
    logger.info(
        f"Feature importances of best param model: \n{sorted(zip(feature_importances, df.columns), reverse=True)}"  # noqa:E501
    )

    final_model = grid_search.best_estimator_
    joblib.dump(final_model, args.output_folder + "/final_model.joblib")
    logger.info("Saved the best estimator model")


if __name__ == "__main__":
    driver_train()
