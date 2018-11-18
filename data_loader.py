import numpy as np
import pandas as pd


class data_loader(object):
	def __init__(self, data_len=4, NaN_val=-0.01):
		self.data_len = data_len
		self.NaN_val = NaN_val


	def read_data(self, file_path="./"):
		df = pd.read_csv(file_path)
		return np.array(df.iloc[:, 1:])

	def load_data(self, file_list=[]):
		file_list = [f+"/" if not f.endswith("/") for f in file_list]

		data = np.zeros((1, 10))
		for fl in file_list:
			d = self.read_data(fl)
			data = np.concatenate((data, d), axis=0)
		return np.delete(data, 0, 0)

	def recursive_data(self, data):
		'''
		Many to One config.
		'''
		dataX, dataY = [], []
		for i in range(len(data) - look_back - 1):
			a = data[i:(i+self.data_len), ::]
			b = data[i+self.data_len, ::]

			dataX.append(a)
			dataY.append(b)
		return np.array(dataX), np.array(dataY)

	def norm_data(self, data):
		min_d, max_d = np.nanmin(data, axis=0), np.nanmax(data, axis=0)
		return (x - min_d) / (max_d - min_d)

	def prepare_data(self, file_list=[]):

		# load data.
		data = self.load_data(file_list=file_list)
		# set NaNs to a constant.
		data[np.isnan(data)] = self.NaN_val
		# normalize data
		data = self.norm_data(data)
		# recur the data.
		dataX, dataY = self.recursive_data(data)
		return dataX, dataY
		
