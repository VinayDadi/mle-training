import argparse
import logging
import os
import os.path as path
import tarfile
import urllib.request
from urllib.error import URLError

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from logger_functions import configure_logger
from sklearn.model_selection import StratifiedShuffleSplit

DEFAULT_OUTPUT_FOLDER = "data/processed"
DEFAULT_DATA_FOLDER = "data"
DEFAULT_ROOT_PATH = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz"  # noqa:E501
# DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
# HOUSING_PATH = os.path.join("data", "raw")
# HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

logger = logging.getLogger(__name__)
os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000/"


def fetch_extract_housing_data(
    housing_url=DEFAULT_ROOT_PATH, housing_path=DEFAULT_DATA_FOLDER
):
    """
    Fetches and extracts data from the given url.

    Parameters
    ----------
    housing_url: str
        URL of raw dataset
    housing_path: str
        Root folder of the extracted dataset

    Returns
    -------
    None
        Extracts the housing dataset

    """
    try:
        logger.info("Started Fetching Data")
        os.makedirs(housing_path + "/raw", exist_ok=True)
        tgz_path = os.path.join(housing_path + "/raw", "housing.tgz")
        urllib.request.urlretrieve(housing_url, tgz_path)
        logger.info("Data Fetch Completed")
    except URLError:
        logger.exception(
            f"Unable to retreive the raw dataset from given url {DEFAULT_ROOT_PATH}"
        )
        exit()
    try:
        logger.info("Started Data Extraction")
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=housing_path + "/raw")
        housing_tgz.close()
        logger.info("Data Extraction Completed")
    except FileNotFoundError:
        logger.exception("Dataset is not found")


def load_housing_data(housing_path=DEFAULT_DATA_FOLDER):
    """
    Loads the dataset.

    Parameters
    ----------
    housing_path: str
        Path of the dataset to load

    Returns
    -------
    Dataframe
        Returns the housing dataframe
    """
    csv_path = path.join(housing_path, "raw", "housing.csv")
    logger.info("Dataset Loading Completed")
    return pd.read_csv(csv_path)


def split_dataset(df):
    """
    Splitting the passed dataframe into test and train by maintaining proportionate ranges of target feature in them.

    Parameters
    ----------
    df: Dataframe
        Housing dataframe with derived columns included

    Returns
    -------
    array(s)
        Returns train and test array respectively
    """  # noqa:E501
    logger.info("Started Data Split")
    housing = df.copy()

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    logger.info("Data Split Completed")
    return strat_train_set, strat_test_set


def initialize_parser():
    """
    Parses command-line arguments so they can used in the code.

    """
    parser = argparse.ArgumentParser()

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

    # if args.output_folder is not None:
    #     if not os.path.exists(args.output_folder):
    #         os.makedirs(args.output_folder)

    return args


def driver_data():
    """
    Driver code to perform below mentioned tasks

        1) Download and extract raw housing dataset
        2) Splitting the dataset into test and train and saves them

    """
    global logger
    experiment_id = mlflow.create_experiment("Data Prep")

    with mlflow.start_run(
        run_name="Parent_run",
        experiment_id=experiment_id,
        description="Fetching Data and splitting the data and saving the split sets",
    ):
        mlflow.log_param("parent", "yes")
        args = initialize_parser()
        # print(args)
        logger = configure_logger(
            logger=logger,
            log_file=args.log_path,
            console=args.no_console_log,
            log_level=args.log_level,
        )
        if not os.path.exists(DEFAULT_DATA_FOLDER + "/raw"):
            logger.info(
                f'Directory "{DEFAULT_DATA_FOLDER + "/raw"}" not found so creating the same'  # noqa:E501
            )
            os.makedirs(DEFAULT_DATA_FOLDER + "/raw")

        if len(os.listdir(DEFAULT_DATA_FOLDER + "/raw")) == 0:
            fetch_extract_housing_data()
            housing = load_housing_data()
        else:
            housing = load_housing_data()

        train_set, test_set = split_dataset(housing)

        if not os.path.exists(args.output_folder):
            logger.info(
                f"Directory '{args.output_folder}' not found so creating the same"
            )
            os.makedirs(args.output_folder)

        train_set.to_csv(args.output_folder + "/train.csv", index=False)
        test_set.to_csv(args.output_folder + "/test.csv", index=False)
        mlflow.log_artifact(args.output_folder + "/train.csv")
        mlflow.log_artifact(args.output_folder + "/test.csv")


if __name__ == "__main__":
    driver_data()
