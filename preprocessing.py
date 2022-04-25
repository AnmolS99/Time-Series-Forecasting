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
        df["min_hour_sin"] = np.sin(date_time.dt.minute * ((2 * np.pi) / 60))
        df["min_hour_cos"] = np.cos(date_time.dt.minute * ((2 * np.pi) / 60))
        return df, ["min_hour_sin", "min_hour_cos"]

    def add_time_of_day_min(self, df):
        """
        Adding time of day (minute)
        """
        date_time = pd.to_datetime(df["start_time"])
        min_in_day = 60 * 24
        df["min_day_sin"] = np.sin(
            ((date_time.dt.hour * 60) + date_time.dt.minute) *
            ((2 * np.pi) / min_in_day))
        df["min_day_cos"] = np.cos(
            ((date_time.dt.hour * 60) + date_time.dt.minute) *
            ((2 * np.pi) / min_in_day))
        return df, ["min_day_sin", "min_day_sin"]

    def add_time_of_day_hour(self, df):
        """
        Adding time of day (hour)
        """
        date_time = pd.to_datetime(df["start_time"])
        df["hour_day_sin"] = np.sin(date_time.dt.hour * ((2 * np.pi) / 24))
        df["hour_day_cos"] = np.cos(date_time.dt.hour * ((2 * np.pi) / 24))
        return df, ["hour_day_sin", "hour_day_cos"]

    def add_time_of_week_sin_cos(self, df):
        """
        Adding time of week (days)
        """
        date_time = pd.to_datetime(df["start_time"])
        df["day_week_sin"] = np.sin(date_time.dt.day_of_week *
                                    ((2 * np.pi) / 7))
        df["day_week_cos"] = np.cos(date_time.dt.day_of_week *
                                    ((2 * np.pi) / 7))
        return df, ["day_week_sin", "day_week_cos"]

    def add_time_of_week_one_hot(self, df):
        """
        Adding time of week (days)
        """
        date_time = pd.to_datetime(df["start_time"])
        df["day_week"] = "day_" + date_time.dt.day_of_week.astype(str)

        possible_categories = [f"day_{i}" for i in range(7)]
        # Getting one-hot encoding of month_year
        one_hot_day = pd.get_dummies(df.day_week, prefix="", prefix_sep="")
        one_hot_day = one_hot_day.reindex(columns=possible_categories,
                                          fill_value=0)

        # List of the feature names
        feats = list(one_hot_day.columns)
        # Adding one-hot columns
        df = df.join(one_hot_day)
        # Dropping month_year
        df = df.drop(columns=["day_week"])
        return df, feats

    def add_time_of_year_sin_cos(self, df):
        """
        Adding time of year (month)
        """
        date_time = pd.to_datetime(df["start_time"])
        df["month_year_sin"] = np.sin(date_time.dt.month * ((2 * np.pi) / 12))
        df["month_year_cos"] = np.cos(date_time.dt.month * ((2 * np.pi) / 12))
        return df, ["month_year_sin", "month_year_cos"]

    def add_time_of_year_one_hot(self, df):
        """
        Adding time of year (months)
        """
        date_time = pd.to_datetime(df["start_time"])
        df["month_year"] = "month_" + date_time.dt.month.astype(str)

        possible_categories = [f"month_{i}" for i in range(1, 13)]
        # Getting one-hot encoding of month_year
        one_hot_month = pd.get_dummies(df.month_year, prefix="", prefix_sep="")
        one_hot_month = one_hot_month.reindex(columns=possible_categories,
                                              fill_value=0)

        # List of the feature names
        feats = list(one_hot_month.columns)
        # Adding one-hot columns
        df = df.join(one_hot_month)
        # Dropping month_year
        df = df.drop(columns=["month_year"])

        return df, feats

    def add_y_24h(self, df):
        """
        Adding the target value (y) 24 hours ago
        """
        one_day_timesteps = (1 * 24 * 60) // 5
        df["y_24h"] = df["y"].shift(one_day_timesteps)
        return df, ["y_24h"]

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
        return df, ["y_yesterday"]

    def add_y_prev(self, df):
        """
        Adding the previous recorded y (5-minutes ago)
        """
        df["y_prev"] = df["y"].shift(1)
        return df, ["y_prev"]

    def add_features(self, df):
        """
        Adding features
        """
        new_feat = []

        # df, feat_list = self.add_time_of_hour(df)
        # new_feat += feat_list

        df, feat_list = self.add_time_of_day_min(df)
        new_feat += feat_list

        # df, feat_list = self.add_time_of_week_sin_cos(df)
        # new_feat += feat_list

        df, feat_list = self.add_time_of_week_one_hot(df)
        new_feat += feat_list

        # df, feat_list = self.add_time_of_year_sin_cos(df)
        # new_feat += feat_list

        df, feat_list = self.add_time_of_year_one_hot(df)
        new_feat += feat_list

        df, feat_list = self.add_y_24h(df)
        new_feat += feat_list

        # df, feat_list = self.add_y_yesterday(df)
        # new_feat += feat_list

        # Adding prev_y last so its always the last column
        df, feat_list = self.add_y_prev(df)
        new_feat += feat_list

        return df, new_feat

    def preprocessing_df(self, df, train_df):
        """
        Preprocessing the dataset specified
        """
        # Clamping 1% of the target values (top 0.5% and lower 0.5%)
        y = df["y"]
        clamped = winsorize(y, limits=[0.005, 0.005])
        df["y"] = np.array(clamped)

        # Declaring original features that are to be scaled
        original_feat = [
            "hydro", "micro", "thermal", "wind", "river", "total", "y",
            "sys_reg", "flow"
        ]
        # Normalizing using a scaler
        scale_features = df[original_feat]
        if train_df:
            df[original_feat] = self.min_max_scaler.fit_transform(
                scale_features)
        else:
            df[original_feat] = self.min_max_scaler.transform(scale_features)

        # Adding features after scaling
        df, new_feat = self.add_features(df)

        X_feat = original_feat + new_feat
        X_feat.remove("y")

        # Removing rows that contain NaN
        df = df.dropna()

        return df, X_feat

    def preprocess(self):
        train_df, X_feat = self.preprocessing_df(self.train_df, train_df=True)
        val_df, _ = self.preprocessing_df(self.val_df, train_df=False)
        return train_df, val_df, X_feat

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
    train_df, val_df, X_feat = pp.preprocess()