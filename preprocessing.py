from this import d
import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import MinMaxScaler


class Preprocessor:

    def __init__(self, train_path) -> None:
        self.min_max_scaler = MinMaxScaler()
        self.train_path = train_path
        self.train_df = self.load_dataset(train_path)

    def load_dataset(self, filepath):
        """
        Loading in dataset
        """
        return pd.read_csv(filepath)

    def add_time_of_day(self, df):
        """
        Adding time of day
        """
        date_time = pd.to_datetime(df["start_time"])
        df["time_of_day_sin"] = np.sin(date_time.dt.hour * ((2 * np.pi) / 24))
        df["time_of_day_cos"] = np.cos(date_time.dt.hour * ((2 * np.pi) / 24))

    def add_time_of_week(self, df):
        """
        Adding time of week
        """
        date_time = pd.to_datetime(df["start_time"])
        df["time_of_week_sin"] = np.sin(date_time.dt.day_of_week *
                                        ((2 * np.pi) / 7))
        df["time_of_week_cos"] = np.cos(date_time.dt.day_of_week *
                                        ((2 * np.pi) / 7))

    def add_time_of_year(self, df):
        """
        Adding time of year
        """
        date_time = pd.to_datetime(df["start_time"])
        df["time_of_year_sin"] = np.sin(date_time.dt.month *
                                        ((2 * np.pi) / 12))
        df["time_of_year_cos"] = np.cos(date_time.dt.month *
                                        ((2 * np.pi) / 12))

    def add_features(self, df):
        """
        Adding features
        """
        self.add_time_of_day(df)
        self.add_time_of_week(df)
        self.add_time_of_year(df)

    def preprocessing(self, df):
        """
        Preprocessing the dataset specified
        """
        # Clamping 1% of the target values (top 0.5% and lower 0.5%)
        y = df["y"]
        clamped = winsorize(y, limits=[0.005, 0.005])
        df["y"] = np.array(clamped)

        # Normalizing using a scaler
        scale_features = df[[
            "hydro", "micro", "thermal", "wind", "river", "total", "y",
            "sys_reg", "flow"
        ]]
        df[[
            "hydro", "micro", "thermal", "wind", "river", "total", "y",
            "sys_reg", "flow"
        ]] = self.min_max_scaler.fit_transform(scale_features)

        self.add_features(df)

        return df


if __name__ == "__main__":
    pp = Preprocessor(train_path="datasets/no1_train.csv")
    print(pp.preprocessing(pp.train_df))
