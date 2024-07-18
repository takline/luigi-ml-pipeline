import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def impute_data(file_path, out_folder="", strategy="median"):
    """
    Impute missing values in a dataset and save the result.

    Parameters:
        file_path (str): Path to the data file.
        out_folder (str): Directory to save the imputed data file.
        strategy (str): Strategy for imputation (default: 'median').

    The function reads a dataset, imputes missing values, and saves the result.
    """
    data = pd.read_csv(file_path)

    imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
    imputer = imputer.fit(data.iloc[:, 1:3])
    data.iloc[:, 1:3] = imputer.transform(data.iloc[:, 1:3])

    file_name = file_path.split("/")[-1]
    file_name = out_folder + file_name.split(".")[0]
    data.to_csv(file_name + "-imputed.csv", index=False)


def encode_data(file_path, out_folder=""):
    """
    Encode categorical data in a dataset and save the result.

    Parameters:
        file_path (str): Path to the data file.
        out_folder (str): Directory to save the encoded data file.

    The function reads a dataset, performs label encoding and one-hot encoding,
    and saves the result.
    """
    data = pd.read_csv(file_path)

    label_encoder = LabelEncoder()
    data["LABEL_ENCODING"] = label_encoder.fit_transform(data.iloc[:, 0])
    data = data[["COUNTRY", "LABEL_ENCODING", "AGE", "SALARY", "PURCHASE"]]

    onehot_encoder = OneHotEncoder()
    onehot = onehot_encoder.fit_transform(data[["LABEL_ENCODING"]]).toarray()
    data["GERMANY"] = onehot[:, 0]
    data["SPAIN"] = onehot[:, 1]
    data["FRANCE"] = onehot[:, 2]
    data = data[
        [
            "COUNTRY",
            "LABEL_ENCODING",
            "GERMANY",
            "SPAIN",
            "FRANCE",
            "AGE",
            "SALARY",
            "PURCHASE",
        ]
    ]

    file_name = file_path.split("/")[-1].split("-")[0]
    file_name = out_folder + file_name.split(".")[0]
    data.to_csv(file_name + "-encoded.csv", index=False)


def load_data(file_path, x_labels, y_labels):
    """
    Load data from a file and separate it into features and labels.

    Parameters:
        file_path (str): Path to the data file.
        x_labels (list): List of column names to use as features.
        y_labels (list): List of column names to use as labels.

    Returns:
        x (DataFrame): Features from the dataset.
        y (DataFrame): Labels from the dataset.
    """
    data = pd.read_csv(file_path)
    x = data[x_labels]
    y = data[y_labels]

    return x, y
