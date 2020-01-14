import os

import time
from keras.layers import LSTM

# Window size or the sequence length
N_STEPS = 3
# Lookup step, 1 is the next day
LOOKUP_STEP = 1

# test ratio size, 0.2 is 20%
TEST_SIZE = 0.2
# features to use
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
# date now
date_now = time.strftime("%Y-%m-%d")

### model parameters
N_LAYERS = 7
# LSTM cell
CELL = LSTM
# 256 LSTM neurons
UNITS = 512
# 40% dropout
DROPOUT = 0.50

### training parameters
# mean squared error loss
LOSS = "mse"
OPTIMIZER = "adam"
ACTIVATION = 'relu'
BATCH_SIZE = 32
EPOCHS = 7

# Apple stock market
TICKER = "SPY"
ticker_data_filename = os.path.join("data", f"{TICKER}_{date_now}.csv")
# model name to save
model_name = f"{date_now}_{TICKER}-{LOSS}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
