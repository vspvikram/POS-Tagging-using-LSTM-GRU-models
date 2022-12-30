import sys, re
import numpy as np
from gensim.models import Word2Vec

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences


def preprocess_sent(sent, tags = None):
	# returns tokenized sentence and tags
	sent = re.split(r"\s+", sent.rstrip())
	sent = [word.lower() for word in sent]

	if tags != None:
		tags = re.split(r"\s+", tags.rstrip())
		tags = [tag for tag in tags]
		return (sent, tags)
	else:
		return sent


###############################################################################

def load_data(txt_file, tags_file=None):
	X = []
	Y = []
	if tags_file != None:
		with open(txt_file) as txtFile, open(tags_file) as tgsFile:
			for line, tags in zip(txtFile, tgsFile):
				X_sent, Y_sent = preprocess_sent(line, tags)
				X.append(X_sent)
				Y.append(Y_sent)
		return (X, Y)
	else:
		with open(txt_file) as txtFile:
			for line in txtFile:
				X_sent = preprocess_sent(line)
				X.append(X_sent)
				# Y.append(Y_sent)
		return X

###############################################################################

def preprocess_data(X, max_seq_len, Y = None, wordTokenizer=None, tagTokenizer=None):

	# X, Y = load_data(txt_file, tags_file)
	# word tokenizer
	if wordTokenizer==None:
		word_tokenizer = Tokenizer(lower=True,
			oov_token='OOV')
		word_tokenizer.fit_on_texts(X)
	else:
		word_tokenizer = wordTokenizer

	#encoding X using tokenizer
	X_encoded = word_tokenizer.texts_to_sequences(X)

	# padding the data both text and tags
	X_padded = pad_sequences(X_encoded, maxlen = max_seq_len,
		padding="pre", truncating="post")
	# print("Type of X_padded: {0}".format(type(X_padded)))
	# print("Example of X_padded: {0}".format(X_padded[0]))

	# tags tokenizer
	if tagTokenizer==None and Y != None:
		tag_tokenizer = Tokenizer()
		tag_tokenizer.fit_on_texts(Y)
	else:
		tag_tokenizer = tagTokenizer

	if Y != None:
		Y_encoded = tag_tokenizer.texts_to_sequences(Y)
		Y_padded = pad_sequences(Y_encoded, maxlen = max_seq_len,
			padding="pre", truncating="post")
		Y_embeddings = to_categorical(Y_padded)
		print("Shape for Y_embeddings[0] : (%s, %s)"%(Y_embeddings.shape, Y_embeddings[0].shape))
		print("Shape for Y_embeddings[10] : (%s, %s)"%(Y_embeddings.shape, Y_embeddings[10].shape))
		return (X_padded, Y_embeddings, word_tokenizer, tag_tokenizer)
	else:
		return (X_padded, word_tokenizer, tag_tokenizer)



###############################################################################

def create_embeddings(X_padded, embedding_size, word_tokenizer):
	model = Word2Vec(sentences=X_padded.tolist(), min_count=1, 
		window=5, vector_size=embedding_size)

	# Initializing embedding matrix weights
	vocabulary_size = len(word_tokenizer.word_index) + 1
	embedding_weights = np.zeros((vocabulary_size, embedding_size))
	word2id = word_tokenizer.word_index

	# copying the embedding weigths from word2vec model
	print("-"*100 + "\nHere is Model.wv")
	# print(model.wv)
	for word, index in word2id.items():
		try:
			embedding_weights[index, :] = model.wv[word2id[word]]
		except KeyError:
			pass
	
	print("Shape of embedding matrix: {0}".format(embedding_weights.shape))
	# X_embeddings = np.array([np.array([model.wv[word] for word in sent]) for sent in X_padded])
	
	
	# one-hot encoding for Y target tags
	# Y_embeddings = to_categorical(Y_padded)
	# Y_embeddings = np.array([to_categorical(sent,
		# num_classes=y_num_classes) for sent in Y_padded])  #.reshape(Y_embeddings.shape, Y_embeddings)

	# print("-"*100 + "\nY_num_classes shape: %s"%(y_num_classes))
	# print("Y_encoded shape: (%s,%s)"%(len(Y_encoded), len(Y_encoded[0])))
	# print("Shape for Y_embeddings[0] : (%s, %s)"%(Y_embeddings.shape, Y_embeddings[0].shape))
	# print("Shape for Y_embeddings[10] : (%s, %s)"%(Y_embeddings.shape, Y_embeddings[10].shape))
	# print(Y_embeddings[0][91:100])
	print("embedding matrix[0]: %s"%(embedding_weights[1]))
	# print("X_embedding shape: ({0}, {1})".format(X_embeddings.shape, X_embeddings[0].shape))

	return (embedding_weights, model)