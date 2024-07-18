import luigi

from utils.preproc import impute_data
from utils.preproc import encode_data
from utils.preproc import load_data
from utils.model import train_model

import datetime
import os


class ImputeData(luigi.Task):
    """
    Luigi task for data imputation.

    Attributes:
        date (luigi.DateParameter): Date parameter for the task.
        data (luigi.LocalTarget): Target path for input data.
        out_folder (luigi.LocalTarget): Target path for output data.
        strategy (str): Strategy for imputation (default: 'median').
    """

    date = luigi.DateParameter()
    data = luigi.LocalTarget(path="./data/financials.csv")
    out_folder = luigi.LocalTarget(path="./outcomes/")
    strategy = luigi.Parameter(default="median")

    if not os.path.exists(out_folder.path):
        os.makedirs(out_folder.path)

    def output(self):
        return luigi.LocalTarget(path=f"./logs/impute-{self.date:%Y-%m-%d}.txt")

    def run(self):
        print("Imputing data...")

        impute_data(self.data.path, self.out_folder.path, self.strategy)

        self.output().makedirs()
        with self.output().open("w") as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} - Imputed data.")


class EncodeData(luigi.Task):
    date = luigi.DateParameter()
    data = luigi.LocalTarget(path="./data/financials.csv")
    out_folder = luigi.LocalTarget(path="./outcomes/")
    strategy = luigi.Parameter(default="median")

    def requires(self):
        return ImputeData(self.date, strategy=self.strategy)

    def output(self):
        return luigi.LocalTarget(path=f"./logs/encode-{self.date:%Y-%m-%d}.txt")

    def run(self):
        print("Encoding data...")

        file_name = self.data.path.split("/")[-1].split(".")[0]
        file_name = self.out_folder.path + file_name
        file_name = file_name + "-imputed.csv"
        encode_data(file_name, self.out_folder.path)

        self.output().makedirs()
        with self.output().open("w") as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} - Encoded data.")


class TrainModel(luigi.Task):
    date = luigi.DateParameter()
    data = luigi.LocalTarget(path="./data/financials.csv")
    out_folder = luigi.LocalTarget(path="./outcomes/")
    strategy = luigi.Parameter(default="median")

    def requires(self):
        return EncodeData(self.date, strategy=self.strategy)

    def output(self):
        return luigi.LocalTarget(path=f"./logs/train-{self.date:%Y-%m-%d}.txt")

    def run(self):
        print("Training model...")

        file_name = self.data.path.split("/")[-1].split(".")[0]
        file_name = self.out_folder.path + file_name
        file_name = file_name + "-encoded.csv"
        x, y = load_data(
            file_name,
            x_labels=["GERMANY", "SPAIN", "FRANCE", "AGE", "SALARY"],
            y_labels=["PURCHASE"],
        )

        train_model(x, y, self.out_folder.path)

        self.output().makedirs()
        with self.output().open("w") as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} - Trained model.")


if __name__ == "__main__":
    luigi.build(
        [TrainModel(date=datetime.datetime.now())], workers=5, local_scheduler=True
    )
