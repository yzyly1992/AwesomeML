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


