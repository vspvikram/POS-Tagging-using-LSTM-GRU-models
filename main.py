#!/usr/bin/python

'''
	File is used to train and test various RNN models on the text data.
	Usage: python main.py train_txt_file train_tgs_file test_txt_file
'''
import sys
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split

import pickle
from utils import preprocess_data, preprocess_sent, load_data, create_embeddings
from models import Vanilla_RNN_model, LSTM_model, Bidirec_LSTM, GRU_model

###############################################################################


def process_new_data(txt_file, max_seq_len, word_tokenizer, tag_tokenizer):
	X = load_data(txt_file)
	X_padded, word_tokenizer, tag_tokenizer = preprocess_data(X, 
		max_seq_len, wordTokenizer=word_tokenizer, tagTokenizer=tag_tokenizer)
	return (X_padded, X)


###############################################################################


if __name__ == "__main__":

	txt_file = sys.argv[1]
	tags_file = sys.argv[2]
	test_txt_file = sys.argv[3]
	# test_tag_file = sys.argv[4]
	max_seq_len = 100
	embedding_size = 300
	TEST_SIZE = 0.15
	VALID_SIZE = 0.15

	X,Y = load_data(txt_file, tags_file)

	X_padded, Y_embeddings, word_tokenizer, tag_tokenizer = preprocess_data(X, max_seq_len, Y)

	# unique word to id dictionary
	word2id = word_tokenizer.word_index
	id2word = word_tokenizer.index_word

	tag2id = tag_tokenizer.word_index
	id2tag = tag_tokenizer.index_word
	y_num_classes=(len(tag_tokenizer.word_index)+1)

	embedding_weights, X_word2vec = create_embeddings(X_padded, 
		embedding_size=embedding_size, word_tokenizer=word_tokenizer)

	X_old, Y_old = X, Y
	X, Y = X_padded, Y_embeddings

	# splitting the data into training, validation and test set
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, 
		random_state=42)
	# X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, 
		# test_size=VALID_SIZE, random_state=42)

	# Printing the size of train, test data
	print("Training Data:\n" + "-"*100)
	print("Train X size: {0}".format(X_train.shape))
	print("Train Y size: {0}".format(Y_train.shape))
	print("\n"+"-"*100+"\nValidation Set data:")
	print("Validation X size: {0}".format(X_test.shape))
	print("Validation Y size: {0}".format(Y_test.shape))
	print("\n"+"-"*100+"\nTest Set data:")
	print("Test X size: {0}".format(X_test.shape))
	print("Test Y size: {0}".format(Y_test.shape))

	vocabulary_size = len(word_tokenizer.word_index) + 1

#------------------------------------------------------------------------------------------
	
	#                    (RNN MODEL RUN)

	# rnn_model = Vanilla_RNN_model(vocabulary_size=vocabulary_size, 
	# 	embedding_weights=embedding_weights, y_num_classes=y_num_classes, 
	# 			embedding_size=embedding_size, max_seq_len=max_seq_len)
	# rnn_model.compile(loss = "categorical_crossentropy",
	# 		optimizer = "adam",
	# 		metrics = ['acc'])
	# rnn_model.build(input_shape = X_train.shape)

	# print(rnn_model.summary())

	# rnn_training = rnn_model.fit(X_train, Y_train, batch_size=128,
	# 	epochs=5, validation_data = (X_test, Y_test))

	# rnn_model.save("RNN/Vanilla_RNN_model")
	# with open('RNN/Vanilla_RNN_model_history', 'wb') as filePi:
	# 	pickle.dump(rnn_training.history, filePi)
	# final_model = rnn_model

#------------------------------------------------------------------------------------------
	
	#                     (LSTM MODEL RUN)

	# LSTM_model = LSTM_model(vocabulary_size=vocabulary_size, 
	# 	embedding_weights=embedding_weights, y_num_classes=y_num_classes, 
	# 			embedding_size=embedding_size, max_seq_len=max_seq_len)
	# LSTM_model.compile(loss = "categorical_crossentropy",
	# 		optimizer = "adam",
	# 		metrics = ['acc'])
	# LSTM_model.build(input_shape = X_train.shape)

	# print(LSTM_model.get_summary())

	# LSTM_training = LSTM_model.fit(X_train, Y_train, batch_size=128,
	# 	epochs=15, validation_data = (X_test, Y_test))

	# LSTM_model.save("RNN/LSTM_model")
	# with open('RNN/LSTM_model_history', 'wb') as filePi:
	# 	pickle.dump(LSTM_training.history, filePi)

	# final_model = LSTM_model

#------------------------------------------------------------------------------------------
	
	
	#                     (Bidirectional LSTM MODEL RUN)

	BiLSTM_model = Bidirec_LSTM(vocabulary_size=vocabulary_size, 
		embedding_weights=embedding_weights, y_num_classes=y_num_classes, 
				embedding_size=embedding_size, max_seq_len=max_seq_len)
	BiLSTM_model.compile(loss = "categorical_crossentropy",
			optimizer = "adam",
			metrics = ['acc'])
	BiLSTM_model.build(input_shape = X_train.shape)

	print(BiLSTM_model.get_summary())

	BiLSTM_training = BiLSTM_model.fit(X_train, Y_train, batch_size=128,
		epochs=20, validation_data = (X_test, Y_test))

	BiLSTM_model.save("RNN/BiLSTM_model")
	with open('RNN/BiLSTM_model_history', 'wb') as filePi:
		pickle.dump(BiLSTM_training.history, filePi)

	final_model = BiLSTM_model

#------------------------------------------------------------------------------------------

	with open('RNN/word2id', 'wb') as word2idFile:
		pickle.dump(word2id, word2idFile)

	with open('RNN/tag2id', 'wb') as tag2idFile:
		pickle.dump(tag2id, tag2idFile)
#------------------------------------------------------------------------------------------


	# with open('RNN/word2id', 'rb') as word2idFile:
	# 	word2id = pickle.dump(word2id, word2idFile)

	# with open('RNN/tag2id', 'rb') as tag2idFile:
	# 	pickle.dump(tag2id, tag2idFile)

	# final_model = models.load_model("RNN/Vanilla_RNN_model")

	out_test_tag_file = "RNN/ptb.22_LSTM.tgs"

	X_new, X_new_origm = process_new_data(test_txt_file, 
		max_seq_len, word_tokenizer, tag_tokenizer)
	predictions = final_model.test_predict(X_new)
	print("\n"+"-"*100+"\nShape of X test predictions: {0}".format(predictions.shape))
	# print("Shape of X test actual Y values: {0}".format(len(Y_new)))

	Y_new_back = [np.argmax(z, axis=1) for z in predictions]
	print("\n"+"-"*100+"\nSample for Y converted back: {0}".format(Y_new_back[0]))

	id2tag[0] = ""
	Y_new_back = [[id2tag[id_] for id_ in sent] for sent in Y_new_back]
	print("\n"+"-"*100+"\nSample for Y tags converted back: {0}".format(Y_new_back[0]))

	# print("-"*100+"\nSample for Y original tags: {0}".format(Y_new_orig[0]))

	with open("{0}".format(out_test_tag_file), 'w') as outTagFile:
		for i in range(len(Y_new_back)):
			length = len(X_new_origm[i])
			sent = Y_new_back[i]
			if length < 100:
				sent = sent[100-length:]
			sent = [tag.upper() for tag in sent]
			sent = " ".join(sent)
			outTagFile.write("{0}\n".format(sent))






