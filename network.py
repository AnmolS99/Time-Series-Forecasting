import math
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

from preprocessing import Preprocessor


class RNN:

    def __init__(self, seq_len, num_feat) -> None:
        self.seq_len = seq_len
        self.num_feat = num_feat
        self.model = self.create_rnn()

    def create_rnn(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer((self.seq_len, self.num_feat)))
        model.add(tf.keras.layers.LSTM(64, return_sequences=True))
        model.add(tf.keras.layers.Dropout(0.20))
        model.add(tf.keras.layers.LSTM(64, return_sequences=False))
        model.add(tf.keras.layers.Dropout(0.20))
        model.add(tf.keras.layers.Dense(32, activation="relu"))
        model.add(tf.keras.layers.Dense(1, activation="linear"))
        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))
        return model

    def train_model(self,
                    train_X,
                    train_Y,
                    validation_data,
                    epochs=3,
                    batch_size=32):

        self.model.fit(train_X,
                       train_Y,
                       validation_data=validation_data,
                       epochs=epochs,
                       batch_size=batch_size)
        self.model.save(
            f"models/model_seq{self.seq_len}_epochs{epochs}_batch{batch_size}")

    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def predict_ahead(self, x, y, start, steps_ahead):
        """
        NB! Assumes that y_prev is the last column
        """
        preds = []

        # First time predicting
        seq = x[[start]]
        pred = self.model.predict(seq)
        preds.append(pred.flatten())
        for i in range(start + 1, start + steps_ahead):
            seq = x[[i]]

            # Replacing y_prevs of next sequence rows, with the previous predictions
            for j in range(1, min(self.seq_len, len(preds)) + 1):
                y_prev = preds[-j]
                seq[:, -j, -1] = y_prev

            pred = self.model.predict(seq)

            preds.append(pred.flatten())

        history = y[start:start + self.seq_len]
        target = y[start + self.seq_len - 1:start + self.seq_len + steps_ahead]
        last_history = float(history[-1])
        preds = [last_history] + preds

        x_history = np.arange(start, start + self.seq_len)
        x_target = np.arange(start + self.seq_len - 1,
                             start + self.seq_len + steps_ahead)
        x_preds = np.arange(start + self.seq_len - 1,
                            start + self.seq_len + steps_ahead)

        return x_history, history, x_target, target, x_preds, preds

    def predict_multiple_series(self, x, y, start, steps_ahead, num_series):
        """
        NB! Assumes that y_prev is the last column
        """

        fig, ax = plt.subplots(nrows=2, ncols=math.ceil(num_series / 2))
        for i in range(num_series):
            x_history, history, x_target, target, x_preds, preds = self.predict_ahead(
                x, y, start + (i * steps_ahead), steps_ahead)
            row = i // math.ceil(num_series / 2)
            col = i % math.ceil(num_series / 2)
            ax[row, col].plot(x_history, history)
            ax[row, col].plot(x_target, target)
            ax[row, col].plot(x_preds, preds)
            ax[row, col].set_title(f"Series {i}")

        # ax[0, math.ceil(num_series / 2) - 1].legend()
        fig.legend(["history", "target", "predictions"], loc="upper right")

        plt.show()


if __name__ == "__main__":
    pp = Preprocessor(train_path="datasets/no1_train.csv",
                      val_path="datasets/no1_validation.csv")
    train_df, val_df = pp.preprocess()
    seq_len = 144
    X_feat = [
        "hydro", "micro", "thermal", "wind", "river", "total", "sys_reg",
        "flow", "time_of_hour_sin", "time_of_hour_cos", "time_of_day_sin",
        "time_of_day_cos", "time_of_week_sin", "time_of_week_cos",
        "time_of_year_sin", "time_of_year_cos", "y_24h", "y_prev"
    ]
    num_feat = len(X_feat)
    rnn = RNN(seq_len=seq_len, num_feat=num_feat)

    train_X = pp.df_to_x(train_df[X_feat], seq_len=seq_len)
    train_Y = pp.df_to_y(train_df["y"], seq_len=seq_len)
    val_X = pp.df_to_x(val_df[X_feat], seq_len=seq_len)
    val_Y = pp.df_to_y(val_df["y"], seq_len=seq_len)

    # Maybe change batch_size to make training go faster
    rnn.train_model(train_X,
                    train_Y,
                    validation_data=(val_X, val_Y),
                    epochs=10,
                    batch_size=32)

    rnn.load_model("models/model_seq72_epochs1_batch32")

    # [
    #     [y_prev1, y_prev2, y_prev3, y_prev4, y_prev5] 'y5 aka y_prev6'
    #     [t2, t3, t4, t5, 't6'] 't7'
    #     [t3, t4, t5, 't6', 't7'] 't8'
    # ]

    rnn.predict_multiple_series(x=val_X,
                                y=val_Y,
                                start=0,
                                steps_ahead=24,
                                num_series=6)
