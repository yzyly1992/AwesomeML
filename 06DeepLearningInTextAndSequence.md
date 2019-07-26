### Deep Learning In Text and Sequence

Intro: in this chapter, we will use deep learing model to deal with text, time sequence, and commen sequence data. Recurrent neural network (循环神经网络) and 1D convnet (一维卷积神经网络) are the two  deep learning models to deal with the sequence problem.

Application:
	- Document Catergory and time sequence catergory, like identity article's topic or the author of the book
	- Comparasion of time sequence, like evaluating the relative of two documents and two stock situatioin
	- Sequence to sequence, like translation to foreign language
	- Affection analysis, catergory contents of twitter and movie comments to positive and negtive
	- Time sequence prediction, like weather prediction according to historical data

6.1 Process Text Data
Vectorize(向量化): Translate Text Data into tensor.
Tokenization(分词): Parse text into unit/token(标记) 
	- Divid text to words, turn word to vector
	- Divid text to character, turn character to vector
	- Extract n-gram of words or characters, turn every n-gram to a vector
	- n-gram is a set of multiple continuous words or characters
One-hot Encoding: methods to relate vector and token
Token Embedding / Word Embedding: methods to relate vector and token
Text -> Token -> Marked Encoded Vector (Numpy Tensor)
n-gram and bags: 2-grams / 3-grams / bag-of-2-grams / bag-of-3-grams / bag-of-words

6.1.1 One-hot Encoding of Words and Characters
``` python
## one-hot code of word
import numpy as np
samples = ['The cat sat on the mat.', 'The dog ate my homework.']

token_index = ()
for sample in samples:
	for word in sample.split():
		if word not in token_index:
			token_index[word] = len(token_index) + 1

max_length = 10

results = np.zeros(shape = (len(samples),
				max_length,
				max(token_index.value()) + 1 ))
for i, sample in enumerate(samples):
	for j, word in list(enumerate(sample.split()))[:max_length]:
		index = token_index.get(word)
		results[i, j, index] = 1.

## one-hot code of character
import string

samples = ['The cat sat on the mat.', 'The dog ate my homework.']
characters = string.printable
token_index = dict(zip(range(1, len(characters) + 1), characters))

max_length = 50
results = np.zeros((len(samples), max_length, max(token_index.keys()) + 1))
for i, sample in enumerate(sample):
	index = token_index.get(character)
	results[i, j, index] = 1.

## use keras to turn word to one-hot code
from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples)

sequences = tokenizer.texts_to_sequences(samples)

one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
```
``` python
## one-hot hashing trick, hash collison
samples = ['The cat sat on the mat.', 'The dog ate my homework.']
dimensionality = 1000
max_length = 10

results = np.zeros(len(samples), max_length, dimensionality))
for i, sample in enumerate(samples):
	for j, word in list(enumerate(sample.split()))) [:max_length]:
	index = abs(hash(word)) % dimensionality
	results[i, j, index] = 1.
```


