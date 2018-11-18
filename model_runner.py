from glob import glob
import numpy as np
import keras

from data_loader import data_loader
from model_builder import BasicLSTM

file_list = glob("./*.csv")
dl = data_loader(data_len=4, NaN_val=-0.01)
trainX, trainY = dl.prepare_data(file_list[:-3])
model = BasicLSTM(h_size=64, last_activation="relu", return_state=False, num_classes=10)
model.compile(loss="mse", optimizer="SGD")
model.fit(dataX, dataY, epochs=100, batch_size=64, verbose=True)
# Test
testX, testY = dl.prepare_data(file_list[-3:])
pred = model.predict(testX)

