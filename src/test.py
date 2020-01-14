import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from src.parameters import *
from src.stock_prediction import *


def upsert_folders():
    # create these folders if they does not exist
    if not os.path.isdir("results"):
        os.mkdir("results")

    if not os.path.isdir("logs"):
        os.mkdir("logs")

    if not os.path.isdir("data"):
        os.mkdir("data")


def plot_graph(title, model, data):
    y_test = data["y_test"]
    X_test = data["X_test"]
    y_pred = model.predict(X_test)
    y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
    plt.plot(y_test[-200:], c='b')
    plt.plot(y_pred[-200:], c='r')

    plt.title(title, fontsize=16)
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()


def get_accuracy(model, data):
    y_test = data["y_test"]
    X_test = data["X_test"]
    y_pred = model.predict(X_test)
    y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
    y_test = list(
        map(lambda current, future: int(float(future) > float(current)), y_test[:-LOOKUP_STEP], y_test[LOOKUP_STEP:]))
    y_pred = list(
        map(lambda current, future: int(float(future) > float(current)), y_pred[:-LOOKUP_STEP], y_pred[LOOKUP_STEP:]))

    return accuracy_score(y_test, y_pred)


def predict(model, data, classification=False):
    # retrieve the last sequence from data
    last_sequence = data["last_sequence"][:N_STEPS]
    # retrieve the column scalers
    column_scaler = data["column_scaler"]
    # reshape the last sequence
    last_sequence = last_sequence.reshape((last_sequence.shape[1], last_sequence.shape[0]))
    # expand dimension
    last_sequence = np.expand_dims(last_sequence, axis=0)
    # get the prediction (scaled from 0 to 1)
    prediction = model.predict(last_sequence)
    # get the price (by inverting the scaling)
    predicted_price = column_scaler["adjclose"].inverse_transform(prediction)[0][0]
    return predicted_price
