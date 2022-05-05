import random
import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import MinMaxScaler
from scipy import interpolate


class Preprocessor:
    """
    Preprocessing object, performing the following actions:
        - Loading data
        - Preprocessing
        - Feature engineering
        - Converting pd DataFrames to np arrays
    """

    def __init__(self, train_path, val_path, alt_forecasting=False) -> None:
        # MinMaxScaler, normalizing data to the range of -1 to 1. The range matches with
        # the sine and cosine features.
        self.min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.train_path = train_path
        self.train_df = self.load_dataset(train_path)
        self.val_path = val_path
        self.val_df = self.load_dataset(val_path)
        self.alt_forecasting = alt_forecasting

    def load_dataset(self, filepath):
        """
        Loading in dataset
        """
        # Reading the CSV-file
        df = pd.read_csv(filepath)

        # Flipping the sign of "flow" to fix error in data
        df["flow"] = -df["flow"]

        return df

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
        return df, ["min_day_sin", "min_day_cos"]

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
        Adding time of week (days) as one-hot encodings
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
        Adding time of year (months) as one-hot encodings
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

    def add_struct_imbalance(self, df):
        """
        Adding structural imbalance as feature
        """
        # Calculating the load/net power
        df["load"] = df["total"] + df["flow"]
        x_load = df["load"].index

        # Converting "start_time" to a datetime object
        date_time = pd.to_datetime(df["start_time"])

        # Finding the load values between hours
        load_midpoints_y_df = df[date_time.dt.minute == 30]["load"]
        # Getting the x-values for the load values between hours
        load_midpoints_x = load_midpoints_y_df.index
        # Converting the load values between hours to numpy array
        load_midpoints_y = df[date_time.dt.minute == 30]["load"].to_numpy()

        # Getting the coefficients of the approximation spline given by the load y- and x-values
        tck = interpolate.splrep(load_midpoints_x, load_midpoints_y)
        # Getting all y/load-values for all points given the coefficients for the approx. spline
        y_interp = interpolate.splev(x_load, tck)

        # Calculating structual imbalance
        df["struct_imb"] = df["load"] - y_interp

        # Adding y with structural imbalance as a feature
        df["y_with_struct_imb"] = df["y"]

        # Removing structural imbalance from y, if altered forecasting
        if self.alt_forecasting:
            df["y"] = df["y"] - df["struct_imb"]

        return df

    def add_features(self, df):
        """
        Adding features
        """
        # List of new features
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
        Preprocessing the dataset
        """
        # Adding structural imbalance as a feature. If altered_forecasting is True, remove struct_imb from y.
        df = self.add_struct_imbalance(df)

        # Clamping 1% of the target values, both y and y_with_struct_imb (top 0.5% and lower 0.5%)
        y = df["y"]
        y_with_struct_imb = df["y_with_struct_imb"]

        y_clamped = winsorize(y, limits=[0.005, 0.005])
        y_with_struct_imb_clamped = winsorize(y_with_struct_imb,
                                              limits=[0.005, 0.005])

        df["y"] = np.array(y_clamped)
        df["y_with_struct_imb"] = np.array(y_with_struct_imb_clamped)

        # Declaring original features that are to be scaled
        original_feat = [
            "hydro", "micro", "thermal", "wind", "river", "total", "y",
            "y_with_struct_imb", "sys_reg", "flow", "struct_imb"
        ]

        # Normalizing using a MinMax-scaler
        scale_features = df[original_feat]
        # Fitting the scaler on train df, and transforming/normalizing
        if train_df:
            df[original_feat] = self.min_max_scaler.fit_transform(
                scale_features)
        # Transforming/normalizing on valid set
        else:
            df[original_feat] = self.min_max_scaler.transform(scale_features)

        # Adding new features after scaling
        df, new_feat = self.add_features(df)

        # Removing target from list of features to use in X
        X_feat = original_feat + new_feat
        X_feat.remove("y")
        X_feat.remove("y_with_struct_imb")

        # Removing rows that contain NaN
        df = df.dropna()

        return df, X_feat

    def preprocess(self):
        """
        Preprocessing train and valid dataframes
        """
        train_df, X_feat = self.preprocessing_df(self.train_df, train_df=True)
        val_df, X_feat = self.preprocessing_df(self.val_df, train_df=False)
        return train_df, val_df, X_feat

    def df_to_x(self, df, seq_len, noise_percent=0):
        """
        Converting x-DataFrame to np array of sequences:
        [[t1, t2, t3, t4, t5],
        [t2, t3, t4, t5, t6],
        [t3, t4, t5, t6, t7]]
        Also adding a percentage of noise to the very last prev_y
        """
        np_df = df.to_numpy().copy()
        x = []
        for i in range(len(np_df) - seq_len + 1):
            row = np_df[i:i + seq_len]
            x.append(row)
        x = np.array(x)
        for i in range(len(x)):
            if random.random() < noise_percent:
                x[i, -1, -1] = random.uniform(-1, 1)
        return np.array(x)

    def df_to_y_prev_true(self, df, seq_len):
        """
        Similar to df_to_x, but only returns a np array of actual y_prev values (with struct imb)
        """
        np_df = df.to_numpy()
        y_prev_true = []
        for i in range(len(np_df) - seq_len + 1):
            row = np_df[i:i + seq_len]
            y_prev_true.append(row)
        return np.array(y_prev_true)

    def df_to_y(self, df, seq_len):
        """
        Converts DataFrame to target-values related to the sequences in df_to_x
        """
        np_df = df.to_numpy()
        y = np_df[seq_len - 1:]
        return np.array(y)