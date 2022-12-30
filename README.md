# POS-Tagging-using-LSTM-GRU-models
This GitHub repository contains a machine learning model implemented in Python that uses recurrent neural networks (RNNs) to perform part-of-speech (POS) tagging. POS tagging is a common task in natural language processing (NLP) that involves assigning a grammatical category to each word in a sentence. While POS tagging is widely seen as a solved problem in the NLP community, it can still be a useful exercise for learning about NLP techniques. 

## Usage
The training and test files are present in the data folder. The training set includes two files: a text file and a file containing grammatical tags for each word in the text file. The test set also has two similar files, but we will only use the text file to predict the POS tags and compare them to the ground truth.

To train the model the following code can be run:
```bash
python main.py data/train.txt data/train.tgs data/test.txt
```
Here, `train.txt` is the text file for the training data. `train.tgs` is the tag file that contains tags for each word in `train.txt`. `text.txt` is the file that contains test text for which we will predict the POS tags.

The default model used is a Bidirectional LSTM model. However, the user can choose to use a different model, such as a `Simple RNN`, `Simple LSTM`, `Bidirectional LSTM`, or `GRU`, by uncommenting the corresponding code in the main.py file.
