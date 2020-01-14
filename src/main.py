from keras.callbacks import ModelCheckpoint, TensorBoard

from src.test import *
import pandas as pd


def main():
    def run():
        # create folders if they does not exist
        upsert_folders()

        # load the CSV file from disk (dataset) if it already exists (without downloading)
        if os.path.isfile(ticker_data_filename):
            ticker = pd.read_csv(ticker_data_filename)

        print("load the data")
        data = load_data(TICKER, N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                         feature_columns=FEATURE_COLUMNS, shuffle=False)

        print("construct the model")
        model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                             dropout=DROPOUT, optimizer=OPTIMIZER)

        # some tensorflow callbacks
        print("Create checkpoints")
        checkpointer = ModelCheckpoint(os.path.join("results", model_name), save_best_only=True, verbose=1)
        # tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))

        print("Fit the model to the data")
        history = model.fit(data["X_train"], data["y_train"],
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
                            validation_data=(data["X_test"], data["y_test"]),
                            callbacks=[checkpointer],
                            verbose=1)
        model.save(os.path.join("results", model_name) + ".h5")

        print("evaluate the model")
        mse, mae = model.evaluate(data["X_test"], data["y_test"])
        print("MSE: " + str(mse))
        print("MAE: " + str(mae))

        # # calculate the mean absolute error (inverse scaling)
        # mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform(mae.reshape(1, -1))[0][0]
        # print("Mean Absolute Error:", mean_absolute_error)

        print("Predict future price")
        future_price = predict(model, data)
        print(f"Future price after {LOOKUP_STEP} days is {future_price:.2f}$")
        print("Accuracy Score:", get_accuracy(model, data))
        title = TICKER + f" Price prediction for {LOOKUP_STEP} days"
        plot_graph(title, model, data)

    run()


if __name__ == '__main__':
    main()
