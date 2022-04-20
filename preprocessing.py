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
        return df

    def add_time_of_week(self, df):
        """
        Adding time of week
        """
        date_time = pd.to_datetime(df["start_time"])
        df["time_of_week_sin"] = np.sin(date_time.dt.day_of_week *
                                        ((2 * np.pi) / 7))
        df["time_of_week_cos"] = np.cos(date_time.dt.day_of_week *
                                        ((2 * np.pi) / 7))
        return df

    def add_time_of_year(self, df):
        """
        Adding time of year
        """
        date_time = pd.to_datetime(df["start_time"])
        df["time_of_year_sin"] = np.sin(date_time.dt.month *
                                        ((2 * np.pi) / 12))
        df["time_of_year_cos"] = np.cos(date_time.dt.month *
                                        ((2 * np.pi) / 12))
        return df

    def add_y_24h(self, df):
        """
        Adding the target value (y) 24 hours ago
        """
        one_day_timesteps = (1 * 24 * 60) // 5
        df["y_24h"] = df["y"].shift(one_day_timesteps)
        return df

    def add_y_yesterday(self, df):
        """
        Adding the mean y for the previous day
        """
        # Converting to datetime
        df['start_time'] = pd.to_datetime(df['start_time'])

        # Finding the y average for each day, shifting that by one and merging it with the dataframe
        df = pd.merge_asof(
            df,
            df.resample('D', on="start_time")["y"].mean().shift(1),
            right_index=True,
            left_on="start_time",
        )
        # Renaming columns
        df = df.rename(columns={"y_x": "y", "y_y": "y_yesterday"})
        return df

    def add_y_prev(self, df):
        """
        Adding the previous recorded y (5-minutes ago)
        """
        df["prev_y"] = df["y"].shift(1)
        return df

    def add_features(self, df):
        """
        Adding features
        """
        df = self.add_time_of_day(df)
        df = self.add_time_of_week(df)
        df = self.add_time_of_year(df)
        df = self.add_y_24h(df)
        df = self.add_y_yesterday(df)
        df = self.add_y_prev(df)
        return df

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

        # Adding features after scaling
        df = self.add_features(df)

        return df


if __name__ == "__main__":
    pp = Preprocessor(train_path="datasets/no1_train.csv")
    print(pp.preprocessing(pp.train_df))
