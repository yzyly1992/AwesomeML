3.1.1 Layer: Basic Element
Intro: Layer is a data operation model that translate single or multiple tensor to single or multiple tensor. Most of layer has its status which is Weight. Weight is a tensor or multi tensors learned from Random Ramp Descend, and it contains the knowledge of this network.

Simple vector data / 2D Tensor -- Densely connected layer / fully connected layer / dense layer / Dense
3D Tensor -- Recurrent Layer: LSTM
4D Tensor -- Convet network / Conv2D
Layer compatibility
```
from keras import layers

layers = layers.Dense(32, input_shape=(784,))
```
3.1.2 Model: Network of layers
Two-branck
Multihead
Inception
Hypothesis Space
3.1.3 Lost Funtion and Optimizer
Bi-catogory Problem: Binary crossentropy
Multi cato: categorical crossentropy
Regression Prob: mean-squared error
Sequential Prob: connectionist temporal classification

3.2 Keras
Two Models: Functional API & Sequential Model
``` python
## Use funtional API define the same model
input_tensor = layers.Input(shape=(784,))
x = layers.Dense(32, activation='relu')(input_tensor)
output_tensor = layers.Dens(10, acivation='softmax')(x)

smodel = models.Model(inputs= input_tensor, outputs=output_tensor)

## Set up lost funtion and optimizer
from keras import optimizers
model.compile(optimizer=optimizers.RMSprop(1r=0.001),
		loss='mse',
		metrics=['accuracy'])

model.fit(input_tensor, target_tensor, batch_size=128, epochs=10)

3.4 Movie Comments Category: Bi-catergory
``` python
from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_work=10000)

word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join(
	[reverse_word_index.get(i - 3, '?') for i in train_data[0]])

## Prepare Data
``` python
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
	results = np.zeros((len(sequences), dimension))
	for i, sequence in enumerate(sequences):
		results[i, sequences] = 1 .
	return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
```
Define a Model
```
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

## 16 are number of hidden unit, each unit means one dimension of space.
## 16 hidden units compose the weight metrix W. 
## W dot input equals projecting input data into the 16-D space.
## output = relu(dot(W, input) + b)
## 16 also means the freedom of network interior learning.
```

Activation Funtion
intro: if we do not have relu, Dense layer only can learn linear relationship. activation funtion(relu) can ensure learning the unlinear relation and effectiveness of multiple layers.

For rate model, crossentropy usually is the best. Bi-caterg best use binary_crossentropy

``` python
model.compile(optimizer='rmsprop',
	      loss='binary_crossentropy',
	      metrics=['accuracy'])
##custmize your optimizer and loss function
from keras import optimizers

model.compile(optimizer=optimizers.RMSprop(1r=0.001),
		loss='binary_crossentropy',
		metrics=['accuracy'])
```
Train Model
```
model.compile(optimizer='rmsprop',
		loss='binary_crossentropy',
		metrics=['acc']

history = model.fit(partial_x_train,
			partial_y_train,
			epochs=20,
			batch_size=512,
			calidatioin_data=(x_val, y_val)
model.predict(x_test)
```
3.5.1 Multiclass classification
``` python
from keras.datasets import reuters

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

## prepare for data
import numpy as np

def vectorize_sequences(sequences, dimension = 10000):
	results = np.zeros((len(sequences), dimension))
	for i, sequence in enumerate(sequences):
		results[i, sequence] = 1.
	return results

## one way to vectorize the data
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

## use one-hot code to vectorize / categorical encoding
def to_one_hot(labels, dimension=46):
	results = np.zeros((len(labels), dimension))
	for i, label in enumerate(labels):
		results[i, label] = 1.
	return results

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

## or use keras to_categorical funtion
one_hot_train_labels = to_categorical(train_labels)
one_hot_train_labels = to_categorical(test_labels)

## define model
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
			loss='categorical_crossentropy',
			metrics=['accuracy'])

## leave 1000 samples as validation dataset
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

## train the network model
history = model.fit(partial_x_train,
		    partial_y_train,
		    epochs=20,
		    batch_size=512,
		    validation_data=(x_val, y_val))

## draw train loss and validation loss
import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

## draw train accuracy and validation accuracy
plt.clf()

acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validatioin acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
```
Summary:
-  if you have N category, the last Dense layer should be N Dimensions
- single label, multiple categories, the last layer should use softmax activation function, which can output the possibility in N categories
- loss funtion always use crossentropy method
- label methods: one-hot code ...
- middle layer should not be too small

3.6 Real estate predicion: Regression Problem
```python
## boston real estate prediction
from keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

## train_data.shape ## (404, 13) ## test_data.shape ## (102, 13)
## each sample has 13 feature number like: criminal rate, average room, reachability of highway...
## target is the average house price

## prepare data - standardlize the data
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

## build network
## since lack of samples, we will use a smaller network including two hidden layer.
## less samples will cause over fitting, small network could reduce overfitting
from keras import models
from keras import layers

def build_model():
	model = models.Sequential()
	model.add(layers.Dense(64, activation='relu',
			input_shap=(train_data.shape[1],)))
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dense(1))
	model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
	return model

## last layer only has one unit and no activation function, is a linear layer. It is a standard setting for standard regression
## adding activation function would limit the output range. i.e. if we add sigmoid activation function, network only can predict 0-1.
## last layer is a pure linear relation, can learn to predict all range of value
## MSE - mean squared error
## MAE - mean absolute error
```

3.6.4 K fold validation
Intro: divide data into K part (4 or 5), K numbers of same model, each model train at K-1 partial, and evaluate at the last partial. The final evaluation equals the everage of K number of results.

``` python
import numpy as np

k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []

for i in range(k):
	print('processing fold #', i)
	val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
	val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
	partial_train_data = np.concatenate(
				[train_data[:i * num_val_samples],
				train_data[(i + 1) * num_val_samples:]],
				axis=0)
	model = build_model()
	model.fit(partial_train_data, partial_train_targets,
			epochs=num_epochs, batch_size=1, verbose=0)
	val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
	all_scores.append(val_mae)
## all_scores ## np.mean(all_scores)
```
Save the validation result for each group
``` python
num_epochs = 500
all_mae_histories = []
for i in range(k):
	print('processing fold #', i)
	val_data = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
	partial_train_data = np.concatentate(
		[train_data[:i * num_val_samples],
		train_data[(i + 1) * num_val_samples:]],
		axis = 0)

	partial_train_targets = np.concatenate(
		[train_targets[:i * num_val_samples],
		train_targets[(i + 1) * num_val_samples:]],
		axis=0)

	model = build_model()
	history = model.fit(partial_train_data, partial_train_targets,
				validation_data=(val_data, val_targets),
				epochs=num_epochs, batch_size=1, verbose=0)
	mae_history = history.history['val_mean_absolute_error']
	all_mae_histories.append(mae_history)
```

``` python
import matplotlib.pyplot as plt

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()
```
```
## training final model
model = build_model()
model.fit(train_data, train_targets, epochs=80, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
```
Chapter Summary
	- if the feature data has different range, we need preprocess these data, scale or standardlize them
	- loss function and evaluation indicator are different between regression problems and categories problems.
	- If we have not much data, k fold is useful for evaluation.
	 
