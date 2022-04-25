import math
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import random

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
                    epochs=10,
                    batch_size=32):

        # Creating model name
        model_name = f"model_seq{self.seq_len}_epochs{epochs}_batch{batch_size}"

        # Maybe try fitting  with shuffle=True
        history = self.model.fit(train_X,
                                 train_Y,
                                 validation_data=validation_data,
                                 epochs=epochs,
                                 batch_size=batch_size)

        # Plotting and saving the loss
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("Model loss")
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['loss', 'val_loss'], loc='upper right')
        plt.savefig(f"losses/{model_name}")
        plt.show()

        self.model.save(f"models/{model_name}")

    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def predict_ahead(self, x, y, start, steps_ahead, show_graphs=False):
        """
        NB! Assumes that y_prev is the last column
        """
        preds = []

        # First time predicting
        seq = x[[start]]
        history = seq[0, :, -1]
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

        last_history = float(history[-1])
        target = y[start:start + steps_ahead]
        target = [last_history] + list(target)
        target = np.array(target, dtype=object)

        preds = [last_history] + preds
        preds = np.array(preds, dtype=object)

        x_history = np.arange(start, start + self.seq_len)
        x_target = np.arange(start + self.seq_len - 1,
                             start + self.seq_len + steps_ahead)
        x_preds = np.arange(start + self.seq_len - 1,
                            start + self.seq_len + steps_ahead)

        if show_graphs:
            plt.plot(x_history, history, label="history")
            plt.plot(x_target, target, label="target")
            plt.plot(x_preds, preds, label="predictions")
            plt.legend()
            plt.show()

        return x_history, history, x_target, target, x_preds, preds

    def predict_multiple_series(self,
                                x,
                                y,
                                start,
                                steps_ahead,
                                num_series,
                                random_series=True):
        """
        NB! Assumes that y_prev is the last column
        """

        fig, ax = plt.subplots(nrows=2, ncols=math.ceil(num_series / 2))

        if random_series:
            x_length = len(x)
            series_list = [
                random.randint(0, x_length - steps_ahead - 1)
                for _ in range(num_series)
            ]
        else:
            series_list = [
                start + (i * steps_ahead) for i in range(num_series)
            ]
        for i in range(num_series):
            x_history, history, x_target, target, x_preds, preds = self.predict_ahead(
                x, y, series_list[i], steps_ahead)
            row = i // math.ceil(num_series / 2)
            col = i % math.ceil(num_series / 2)
            ax[row, col].plot(x_history, history)
            ax[row, col].plot(x_target, target)
            ax[row, col].plot(x_preds, preds)
            ax[row, col].set_title(f"Series {i}")

        fig.legend(["history", "target", "predictions"], loc="upper right")

        plt.show()


if __name__ == "__main__":
    pp = Preprocessor(train_path="datasets/no1_train.csv",
                      val_path="datasets/no1_validation.csv")
    train_df, val_df, X_feat = pp.preprocess()
    seq_len = 72
    rnn = RNN(seq_len=seq_len, num_feat=len(X_feat))

    train_X = pp.df_to_x(train_df[X_feat], seq_len=seq_len)
    train_Y = pp.df_to_y(train_df["y"], seq_len=seq_len)
    val_X = pp.df_to_x(val_df[X_feat], seq_len=seq_len)
    val_Y = pp.df_to_y(val_df["y"], seq_len=seq_len)

    # rnn.train_model(train_X,
    #                 train_Y,
    #                 validation_data=(val_X, val_Y),
    #                 epochs=7,
    #                 batch_size=32)

    rnn.load_model(f"models/model_seq{seq_len}_epochs7_batch32")

    # rnn.predict_ahead(x=val_X,
    #                   y=val_Y,
    #                   start=0,
    #                   steps_ahead=1,
    #                   show_graphs=True)

    # [
    #     [y_prev1, y_prev2, y_prev3, y_prev4, y_prev5] 'y_prev6'
    #     [y_prev2, y_prev3, y_prev4, y_prev5, 'y_prev6'] 'y_prev7'
    #     [y_prev3, y_prev4, y_prev5, 'y_prev6', 'y_prev7'] 'y_prev8'
    #     [y_prev4, y_prev5, 'y_prev6', 'y_prev7', 'random_noise'] 'y_prev9'
    # ]

    rnn.predict_multiple_series(x=val_X,
                                y=val_Y,
                                start=0,
                                steps_ahead=24,
                                num_series=6,
                                random_series=True)
