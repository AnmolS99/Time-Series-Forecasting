from network import RNN
from preprocessing import Preprocessor


def main(seq_len=144,
         epochs=10,
         batch_size=32,
         alt_forecasting=False,
         train_model=False):
    """
    Main function for running this python script.
    """

    # Creating Preprocessor, specifing the train and valid paths
    pp = Preprocessor(train_path="datasets/no1_train.csv",
                      val_path="datasets/no1_validation.csv",
                      alt_forecasting=alt_forecasting)

    # Preprocessing train and valid
    train_df, val_df, X_feat = pp.preprocess()
    rnn = RNN(seq_len=seq_len, num_feat=len(X_feat))

    train_X = pp.df_to_x(train_df[X_feat], seq_len=seq_len, noise_percent=0.25)
    train_Y = pp.df_to_y(train_df["y"], seq_len=seq_len)

    val_X = pp.df_to_x(val_df[X_feat], seq_len=seq_len)
    val_Y_prev_true = pp.df_to_y_prev_true(val_df["y_with_struct_imb"],
                                           seq_len=seq_len)
    val_Y = pp.df_to_y(val_df["y"], seq_len=seq_len)
    val_Y_with_struct_imb = pp.df_to_y(val_df["y_with_struct_imb"],
                                       seq_len=seq_len)

    # Training the model
    if train_model:
        rnn.train_model(train_X,
                        train_Y,
                        validation_data=(val_X, val_Y),
                        epochs=epochs,
                        batch_size=batch_size)

    # Loading the model
    else:
        model_path = f"models/model_seq{seq_len}_epochs{epochs}_batch{batch_size}"
        if alt_forecasting:
            model_path += "_alt_fore"

        rnn.load_model(model_path)

    rnn.predict_one_ahead(x=val_X,
                          y=val_Y,
                          y_with_struct_imb=val_Y_with_struct_imb,
                          start=0,
                          steps_ahead=144)

    # Predicting multiple series
    rnn.predict_multiple_series(x=val_X,
                                y_prev_true=val_Y_prev_true,
                                y=val_Y,
                                y_with_struct_imb=val_Y_with_struct_imb,
                                start=0,
                                steps_ahead=24,
                                num_series=6,
                                random_series=True,
                                alt_forecasting=alt_forecasting)


if __name__ == '__main__':

    main(seq_len=144,
         epochs=10,
         batch_size=32,
         alt_forecasting=False,
         train_model=False)
