from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
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
        model.add(tf.keras.layers.LSTM(64))
        model.add(tf.keras.layers.Dense(8, activation="relu"))
        model.add(tf.keras.layers.Dense(1, activation="linear"))
        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))
        return model

    def predict_ahead(self, x, y, start, steps_ahead):
        """
        NB! Assumes that y_prev is the last column
        """
        preds = []

        # First time predicting
        seq = x[[start]]
        pred = self.model.predict(seq)
        preds.append(pred.flatten())
        for i in range(start + 1, steps_ahead):
            seq = x[[i]]

            # Replacing y_prevs of next sequence rows, with the previous predictions
            for j in range(1, min(self.seq_len, len(preds)) + 1):
                y_prev = preds[-j]
                seq[:, -j, -1] = y_prev

            pred = self.model.predict(seq)

            preds.append(pred.flatten())

        y = y[start:start + self.seq_len + steps_ahead]
        preds = preds
        x_y = np.linspace(start, start + self.seq_len + steps_ahead,
                          self.seq_len + steps_ahead)
        x_preds = np.linspace(start + self.seq_len,
                              start + self.seq_len + steps_ahead, steps_ahead)
        plt.plot(x_y, y, label="y")
        plt.plot(x_preds, preds, label="predictions")
        plt.show()


if __name__ == "__main__":
    pp = Preprocessor(train_path="datasets/no1_train.csv",
                      val_path="datasets/no1_validation.csv")
    train_df, val_df = pp.preprocess()
    seq_len = 72
    X_feat = [
        "hydro", "micro", "thermal", "wind", "river", "total", "sys_reg",
        "flow", "time_of_day_sin", "time_of_day_cos", "time_of_week_sin",
        "time_of_week_cos", "time_of_year_sin", "time_of_year_cos", "y_24h",
        "y_yesterday", "y_prev"
    ]
    num_feat = len(X_feat)
    rnn = RNN(seq_len=seq_len, num_feat=num_feat)

    train_X = pp.df_to_x(train_df[X_feat], seq_len=seq_len)
    train_Y = pp.df_to_y(train_df["y"], seq_len=seq_len)
    val_X = pp.df_to_x(val_df[X_feat], seq_len=seq_len)
    val_Y = pp.df_to_y(val_df["y"], seq_len=seq_len)

    # Maybe change batch_size to make training go faster
    rnn.model.fit(train_X,
                  train_Y,
                  validation_data=(val_X, val_Y),
                  epochs=3,
                  batch_size=256)

    rnn.predict_ahead(x=val_X, y=val_Y, start=0, steps_ahead=24)
