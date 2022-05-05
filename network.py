import math
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import random


class RNN:
    """
    Reccurent Neural Network object, containing the model, and operations related to the model
    """

    def __init__(self, seq_len, num_feat) -> None:
        self.seq_len = seq_len
        self.num_feat = num_feat
        self.model = self.create_rnn()

    def create_rnn(self):
        """
        Creating the RNN
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer((self.seq_len, self.num_feat)))
        model.add(tf.keras.layers.LSTM(128, return_sequences=True))
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
        """
        Training the model (fit) and saving it and the corresponding loss graph
        """
        # Creating model name
        model_name = f"model_seq{self.seq_len}_epochs{epochs}_batch{batch_size}"

        # Fitting the model, and getting the training history
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

        # Saving the model
        self.model.save(f"models/{model_name}")

    def load_model(self, model_path):
        """
        Loading an existing model from path
        """
        self.model = tf.keras.models.load_model(model_path)

    def predict_ahead(self,
                      x,
                      y_prev_true,
                      y,
                      y_with_struct_imb,
                      start,
                      steps_ahead,
                      show_graphs=False):
        """
        Using RNN-model to predict a number of steps ahead
        """
        preds = []

        # First time predicting
        seq = x[[start]]

        # Getting the history (y_with_struct_imb for first sequence sent into the model)
        history = y_prev_true[start]

        # Predicting the y for the first sequence
        pred = self.model.predict(seq)
        preds.append(pred.flatten())

        # Predicting further into the future
        for i in range(start + 1, start + steps_ahead):
            seq = x[[i]]

            # Replacing y_prevs of next sequence rows, with the previous predictions
            for j in range(1, min(self.seq_len, len(preds)) + 1):
                y_prev = preds[-j]
                seq[:, -j, -1] = y_prev

            # Predicting the sequence after changing y_prev from actual value to predicted value
            pred = self.model.predict(seq)

            preds.append(pred.flatten())

        # Getting the target values
        last_history = float(history[-1])
        target = y_with_struct_imb[start:start + steps_ahead]
        target = [last_history] + list(target)
        target = np.array(target, dtype=object)

        # Calculating the structural imbalance
        struct_imb = y_with_struct_imb[start:start +
                                       steps_ahead] - y[start:start +
                                                        steps_ahead]
        preds = np.array(preds).flatten()
        # Adding the strucural imbalance back to our predictions
        preds = np.add(preds, struct_imb)
        preds = np.concatenate(([last_history], preds))
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

        # Returning history, target and predictions
        return x_history, history, x_target, target, x_preds, preds

    def predict_multiple_series(self,
                                x,
                                y_prev_true,
                                y,
                                y_with_struct_imb,
                                start,
                                steps_ahead,
                                num_series,
                                random_series=True,
                                alt_forecasting=False):
        """
        Predicting multiple times using predict_ahead()
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
        # Predicting multiple times and plotting the results
        for i in range(num_series):
            x_history, history, x_target, target, x_preds, preds = self.predict_ahead(
                x, y_prev_true, y, y_with_struct_imb, series_list[i],
                steps_ahead, False)
            row = i // math.ceil(num_series / 2)
            col = i % math.ceil(num_series / 2)
            ax[row, col].plot(x_history, history)
            ax[row, col].plot(x_target, target)
            ax[row, col].plot(x_preds, preds)
            ax[row, col].set_title(f"Series {i}")

        # Adding legend
        fig.legend(["history", "target", "predictions"], loc="upper right")

        # Showing the plots
        plt.show()

    def predict_one_ahead(self, x, y, y_with_struct_imb, start, steps_ahead):
        """
        Using RNN-model to predict a number one steps ahead multiple times
        """
        preds = []

        # First time predicting
        seq = x[[start]]

        # Predicting the y for the first sequence
        pred = self.model.predict(seq)
        preds.append(pred.flatten())

        # Predicting further into the future
        for i in range(start + 1, start + steps_ahead):
            seq = x[[i]]

            # Predicting the sequence after changing y_prev from actual value to predicted value
            pred = self.model.predict(seq)

            preds.append(pred.flatten())

        # Getting the target values
        target = y_with_struct_imb[start:start + steps_ahead]
        target = list(target)
        target = np.array(target, dtype=object)

        # Calculating the structural imbalance
        struct_imb = y_with_struct_imb[start:start +
                                       steps_ahead] - y[start:start +
                                                        steps_ahead]
        preds = np.array(preds).flatten()
        # Adding the strucural imbalance back to our predictions
        preds = np.add(preds, struct_imb)
        preds = np.array(preds, dtype=object)

        plt.plot(target, label="target")
        plt.plot(preds, label="predictions")
        plt.legend()
        plt.show()
