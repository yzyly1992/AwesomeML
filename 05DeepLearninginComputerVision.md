G## Deep Learning in Computer Vision

5.1 Conv Neural Network Intro
``` python
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

## lets check Convnet structure
model.summary()
```
``` python
## add category model
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
```

``` python
## Train Convnet at MNIST Image set
from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer='rmsprop',
		loss='categorical_crossentropy',
		metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)
## test_loss, test_acc = model.evaluate(test_images, test_labels)
```

5.1.1 Convnet Calculation
	- Translation Invariant
	- Spatial Hierachies of Patterns
	- Feature Map, Output Feature Map, Response Map
	- Strided Convolution, Concolution Kernel

5.1.2 MaxPooling2D
	- MaxPooling reduce half size of the feature map
	
5.2 From small dataset to train a Convnet
5.2.1 5.2.2

``` python
import os, shutil

ortiginal_dataset_dir = '/Users/fchollet/Downloads/kaggle_original_data'

base_dir = '/Users/fchollet/Downloads/cats_and_dogs_small'
os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir, 'cats')
os.makdir(validation_cats_dir)

validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)

test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000)
for fname in fnames:
	src = os.path.join(original_dataset_dir, fname)
	dst = os.path.join(train_cats_dir, fname)
	shutil.copyfile(src, dst)
	
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)
for fname in fnames:
	src = os.path.join(original_dataset_dir, fname)
	dst = os.path.join(validation_cats_dir, fname)
	shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)
for fname in fnames:
	src = os.path.join(original_dataset_dir, fname)
	dst = os.path.join(test_cats_dir, fname)
	shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000)
for fname in fnames:
	src = os.path.join(original_dataset_dir, fname)
	dst = os.path.join(train_dogs_dir, fname)
	shutil.copyfile(src, dst)
	
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)
for fname in fnames:
	src = os.path.join(original_dataset_dir, fname)
	dst = os.path.join(validation_dogs_dir, fname)
	shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)
for fname in fnames:
	src = os.path.join(original_dataset_dir, fname)
	dst = os.path.join(test_dogs_dir, fname)
	shutil.copyfile(src, dst)
```

5.2.3 Build Network

``` python
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

## Setting to train
from keras import optimizers

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=le-4), metrics=['acc'])
```

