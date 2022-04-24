import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import MinMaxScaler


class Preprocessor:

    def __init__(self, train_path, val_path) -> None:
        self.min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.train_path = train_path
        self.train_df = self.load_dataset(train_path)
        self.val_path = val_path
        self.val_df = self.load_dataset(val_path)

    def load_dataset(self, filepath):
        """
        Loading in dataset
        """
        return pd.read_csv(filepath)

    def add_time_of_hour(self, df):
        """
        Adding time of hour (minutes)
        """
        date_time = pd.to_datetime(df["start_time"])
        df["time_of_hour_sin"] = np.sin(date_time.dt.minute *
                                        ((2 * np.pi) / 60))
        df["time_of_hour_cos"] = np.cos(date_time.dt.minute *
                                        ((2 * np.pi) / 60))
        return df

    def add_time_of_day(self, df):
        """
        Adding time of day (hours)
        """
        date_time = pd.to_datetime(df["start_time"])
        df["time_of_day_sin"] = np.sin(date_time.dt.hour * ((2 * np.pi) / 24))
        df["time_of_day_cos"] = np.cos(date_time.dt.hour * ((2 * np.pi) / 24))
        return df

    def add_time_of_week(self, df):
        """
        Adding time of week (days)
        """
        date_time = pd.to_datetime(df["start_time"])
        df["time_of_week_sin"] = np.sin(date_time.dt.day_of_week *
                                        ((2 * np.pi) / 7))
        df["time_of_week_cos"] = np.cos(date_time.dt.day_of_week *
                                        ((2 * np.pi) / 7))
        return df

    def add_time_of_year(self, df):
        """
        Adding time of year (month)
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
        df["y_prev"] = df["y"].shift(1)
        return df

    def add_features(self, df):
        """
        Adding features
        """
        df = self.add_time_of_hour(df)
        df = self.add_time_of_day(df)
        df = self.add_time_of_week(df)
        df = self.add_time_of_year(df)
        df = self.add_y_24h(df)
        # df = self.add_y_yesterday(df)
        # Adding prev_y last so its always the last column
        df = self.add_y_prev(df)
        return df

    def preprocessing_df(self, df, train_df):
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
        if train_df:
            df[[
                "hydro", "micro", "thermal", "wind", "river", "total", "y",
                "sys_reg", "flow"
            ]] = self.min_max_scaler.fit_transform(scale_features)
        else:
            df[[
                "hydro", "micro", "thermal", "wind", "river", "total", "y",
                "sys_reg", "flow"
            ]] = self.min_max_scaler.transform(scale_features)

        # Adding features after scaling
        df = self.add_features(df)

        # Removing rows that contain NaN
        df = df.dropna()

        return df

    def preprocess(self):
        train_df = self.preprocessing_df(self.train_df, train_df=True)
        val_df = self.preprocessing_df(self.val_df, train_df=False)
        return train_df, val_df

    def df_to_x(self, df, seq_len):
        np_df = df.to_numpy()
        x = []
        for i in range(len(np_df) - seq_len + 1):
            row = np_df[i:i + seq_len]
            x.append(row)
        return np.array(x)

    def df_to_y(self, df, seq_len):
        np_df = df.to_numpy()
        y = np_df[seq_len - 1:]
        return np.array(y)


if __name__ == "__main__":
    pp = Preprocessor(train_path="datasets/no1_train.csv",
                      val_path="datasets/no1_validation.csv")
    train_df, val_df = pp.preprocess()