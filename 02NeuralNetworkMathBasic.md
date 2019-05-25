### Chapter 2 Math Basic of Neural Network

2.1 Intro to Neural Network
Exampel: MNIST; Class, Samplem Label,

```python
from keras.datasets import mnist

##training set and test set
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

## check shape of the data set
## train_images.shape ## (60000, 28, 28)
## len(test_labels) ## 10000
## test_labels ## array{[7, 2, 1, ..., 4, 5, 6], dtype-uint8}

from keras import models
from keras import layers

## Layer is kind of data distillation or filter
## Dense layer is a fully connected model
## 10 ways Softmax returns 10 numbers which total value is 1.0
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

## Then we need three parameters to compile the model: loss function, optimizer, metric
network.compile(optimizer='rmsprop',
		loss='categorical_crossentropy',
		metrics=['accuracy'])

## Preprocess the data
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.reshape('float32') / 255

## catergary compile the labels
from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

## use fit to train the model
network.fit(train_images, train_labels, epochs=5, batch_size= 128)

## then test the model on the test data set
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

## end of the example
```

2.2 Data expression in Neural Network
Unit: Tensor / Numpy number array
Tensor's dimension calls axis;
0D Tensor: only one number we call it scalar / 0D Tensor (ndim == 0) / Biao liang
```python
import numpy as np
x = np.array(12)
x.ndim
## 0
```
1D Tensor: Vector, ndim == 1
```python
x = np.array([12, 3, 6, 14, 7])
x.ndim
## 1
```
2D Tensor: Matrix has row and column, ndim == 2
``` python
x = np.array([[5, 78, 2, 34, 0],
	      [6, 79, 3, 35, 1],
	      [7, 34, 4, 37, 2]])
x.ndim
## 2
```
3D Tensor: More like a cube; ndim == 3
```python
x = np.array([[[5, 34, 6],
	       [6, 36, 9],
	       [2, 38, 1]],
	      [[6, 35, 2],
	       [0, 48, 1]
	       [8, 30, 4]],
	      [[5, 29, 0],
	       [2, 40, 7],
	       [6, 30, 9]]])
x.ndim
## 3
```
4D Tensor, 5D Tensor....

Data Type: dtype could be float 32, uint8, float 64

```python
## use Matplotlib to show the number image of 3d tensor
digit = train_images[4]

import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
```

2.2.6 Manipulate Tensor in Numpy
Tensor Slicing
```python
my_slice = train_images[10:100]
print(my_slice.shape) ## (90, 28, 28)

## my_slice = train_images[10:100, :, :]
## my_slice = train_images[10:100, 0:28, 0:28]
## Same as exmaple

## 14 by 14 Right coner of image
my_slice = train_images[:, 14:, 14:]

## 14 by 14 Center of images
my_slice = train_images[:, 7:-7, 7:-7]
```

2.2.7 Data Batch
OD is also called samples axis / batch axis / batch dimension
```python
batch = train_images[:128]
## next batch
batch = train_images[128:256]
## n batch
batch = train_images[128 * n: 128 * (n + 1)]
```

2.2.8 Tensor in the real world
2D Tensor: Vector (samples, features)
3D tensor: Time Sequential Data (samples, timesteps, features)
4D Tensor: Images (samples, height, width, channels) or (samples, channels, height, width)
5D Tensor: Videos (samples, frames, height, width, channels) or (samples, frames, channels, height, width)

2.2.9 Tensor Data

2.3 Tensor Operation
`keras.layers.Dense(512, activation='relu')`
`output = relu(dot(W, input) + b)`
2.3.1 Element-wise Operation
relu and plus are all element-wise operation
```
def naive_relu(x):
	assert len(x.shape) == 2

	x = x.copy()
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			x[i, j] = max(x[i,j], 0)
	return x
```
Plus operation
```
def naive_add(x, y):
	assert len(x.shape) == 2
	assert len(y.shape) == 2

	x = x.copy()
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			x[i, j] += y[i, j]
	return x
```

2.3.2 Broadcast
Add axis to smaller Tensor until same as bigger one. Then repeat in new axis to make it has the same shape with the bigger one.
2.3.3 Tensor Product / Dot / Dian Ji Operation
```
import numpy as np
z = np.dot(x, y)
## math express: z = x.y

## See what they do as a python function
def naive_vector_dot(x, y):
	assert len(x.shape) == 1
	assert len(y.shape) == 1
	assert x.shape[0] == y.shape[0]

	z = 0.
	for i in range(x.shape[0]):
		z += x[i] * y[i]
	return z

## or

def naive_matrix_vector_dot(x, y):
	z = np.zeros(x.shape[0])
	for i in range(x.shape[0]):
		z[i] = naive_vector_dot(x[i, :], y)
	return z

## Uneven dot operation
def naive_matrix_dot(x,y):
	assert len(x.shape) == 2
	assert len(y.shape) == 2
	assert x.shape[1] == y.shape[0]

	z = np.zeros((x.shape[0], y.shape[1]))
	for i in range(x.shape[0]):
		for j in range(y.shape[1]):
			row_x = x[i, :]
			column_y = y[: j]
			z[i, j] = naive_vector_dot(row_x, column_y)
	return z

## math express: x.y = z
(a, b, c, d) . (d,) -> (a, b ,c)
(a, b, c, d) . (d, e) -> (a, b, c, e)

2.3.4 Tensor Reshaping
print(x.shape) 
(3, 2)
x = x.reshape((6, 1))
x = x.reshape((2, 3))
## Transposition
x = np.zeros((300,20))
x = np.transpose(x)
pirnt(x.shape) ## (20, 300)
```
2.3.5 Tensor Operation in Geometry Explanation

2.4 Neural Network Engine: SGD Optimization
Output = relu(dot(W, input) + b)
W & b: weight / trainable parameter -- kernel and bias
started with Random Initialization
Training loop
Use Differentiable to calculate the loss gradient
Derivative(Dao Shu)
f'(x) -- Xie lv / Gradient
```
y_pred = dot(W, x)
loss_value = loss(y_pred, y)
loss_value = f(W)
We can reduce the f(W) if we move the opposite direction of f'(W)
```

2.4.3 Random Gradient Reduction
Mini-batch Stochastic Gradient Descent (SGD) stochastic = random 

SGD, Adagrad, RMSProp with momentum call optimization method or optimizer 
Momentum / Dong Liang can avoid local minimum to find global minimum

2.4.4 Chain Rule
f(W1, W2, W3) = a(W1, b(W2, c(W3)))
(f(g(x)))' = f'(g(x)) * g'(x)
Backpropagation / Reverse-mode Differentiation
Symbolic Differentiation -- TensorFlow


