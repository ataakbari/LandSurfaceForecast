import numpy as np
import keras


class BasicLSTM(keras.Model):

	def __init__(self, h_size=64, last_activation="relu", return_state=False, num_classes=10):
		super(BasicLSTM, self).__init__(name='lstm')
		self.h_size = h_size
		self.return_state = return_state
		self.num_classes = num_classes
		self.last_activation = last_activation

		self.lstm1 = keras.layers.LSTM(h_size, return_sequences=True, return_state=self.return_state)
		self.lstm2 = keras.layers.LSTM(h_size, return_sequences=True, return_state=self.return_state)
		self.dense = keras.layers.Dense(self.num_classes, activation=self.last_activation)


	def call(self, input_shape):
		inputs = Input(shape=input_shape)
		if self.return_state:
			x, h, c = self.lstm1(inputs)
			x, _, _ = self.lstm2(x, initial_state=[h, c])
			return self.dense(x)
		else:
			x = self.lstm1(inputs)
			x = self.lstm2(x)
			return self.dense(x)

