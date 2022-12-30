
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, GRU, Embedding, SimpleRNN, TimeDistributed, Dense
from tensorflow.keras import Model, models
import pickle


class Vanilla_RNN_model(Model):
	"""
	Creates an instance of a simple RNN model
	"""
	def __init__(self, vocabulary_size, embedding_weights, y_num_classes, 
				embedding_size=300, max_seq_len=100, weights_trainable=True):
		super(Vanilla_RNN_model, self).__init__()
		self.model = Sequential()
		self.model.add(Embedding(input_dim = vocabulary_size,
					output_dim = embedding_size,
					input_length = max_seq_len,
					weights = [embedding_weights],
					trainable = weights_trainable))
		# Adding an RNN layer that has 64 RNN cells
		self.model.add(SimpleRNN(64,
						return_sequences = True)) # return whole sequences
		# if False, returns single output of the end of the sequence
		self.model.add(TimeDistributed(Dense(y_num_classes,
										activation = "softmax")))

	def call(self, input_tensor):
		X = self.model.call(input_tensor)
		return X

	def test_predict(self, X_test):
		Y = self.model.predict(X_test)
		return Y

###############################################################################

class LSTM_model(Model):
	def __init__(self, vocabulary_size, embedding_weights, y_num_classes, 
				embedding_size=300, max_seq_len=100, weights_trainable=True):
		super(LSTM_model, self).__init__()
		
		self.model = Sequential()
		self.model.add(Embedding(input_dim = vocabulary_size,
			output_dim = embedding_size, input_length = max_seq_len,
			weights = [embedding_weights], trainable = weights_trainable))
		self.model.add(LSTM(64, return_sequences = True))
		self.model.add(TimeDistributed(Dense(y_num_classes, activation="softmax")))


	def call(self, input_tensor):
		X = self.model.call(input_tensor)
		return X

	def test_predict(self, X_test):
		Y = self.model.predict(X_test)
		return Y

	def get_summary(self):
		return self.model.summary()


###############################################################################


class GRU_model(Model):
	def __init__(self, vocabulary_size, embedding_weights, y_num_classes, 
				embedding_size=300, max_seq_len=100, weights_trainable=True):
		super(GRU_model, self).__init__()
		
		self.model = Sequential()
		self.model.add(Embedding(input_dim = vocabulary_size,
			output_dim = embedding_size, input_length = max_seq_len,
			weights = [embedding_weights], trainable = weights_trainable))
		self.model.add(GRU(64, return_sequences = True))
		self.model.add(TimeDistributed(Dense(y_num_classes, activation="softmax")))


	def call(self, input_tensor):
		X = self.model.call(input_tensor)
		return X

	def test_predict(self, X_test):
		Y = self.model.predict(X_test)
		return Y

	def get_summary(self):
		return self.model.summary()


###############################################################################

class Bidirec_LSTM(Model):
	def __init__(self, vocabulary_size, embedding_weights, y_num_classes, 
				embedding_size=300, max_seq_len=100, weights_trainable=True):
		super(Bidirec_LSTM, self).__init__()
		self.model = Sequential()
		self.model.add(Embedding(input_dim = vocabulary_size,
				output_dim = embedding_size, input_length = max_seq_len,
				weights = [embedding_weights], trainable = weights_trainable))
		self.model.add(Bidirectional(LSTM(64, return_sequences=True)))
		self.model.add(TimeDistributed(Dense(y_num_classes, activation="softmax")))

	def call(self, input_tensor):
		X = self.model.call(input_tensor)
		return X

	def test_predict(self, X_test):
		Y = self.model.predict(X_test)
		return Y

	def get_summary(self):
		return self.model.summary()


###############################################################################
