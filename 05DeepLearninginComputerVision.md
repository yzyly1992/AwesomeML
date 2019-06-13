## Deep Learning in Computer Vision

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

5.2.4 Data Preprocess
	- this process is going to read the image file, and tranlate to float tensor, then scale to 0 to 1 range. Keras has tool can automaticly finish these processes.

``` python
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
	train_dir,
	target_size=(150, 150),
	batch_size=20,
	class_mode='binary')

## use fit_generator to fit the model
history = model.fit_generator(
	train_generator,
	steps_per_epoch=100,
	epochs=30,
	validation_data=validation_generator,
	validation_steps=50)

## save the model after training
model.save('cats_and_dogs_small_1.h5')

## draw the lost and acc curve
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
```
5.2.5 Use data augmentation
	- Overfitting because of lack of study samples
	- we could use ImageDataGenerator to transform and changing the images in multiple times
``` python
## use ImageDataGenerator to set the data augmentation
datagen = ImageDataGenerator(
	rotation_range=40,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True,
	fill_mode='nearest')

## randomly show some augmented images
from keras.preprocessing import image

fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]
img_path = fnames[3]
img = image.load_img(img_path, target_size=(150, 150))
x = image.img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size=1):
	plt.figure(i)
	imgplot = plt.imshow(image.array_to_img(batch[0]))
	i += 1
	if i % 4 == 0:
		break

plt.show()
```

``` python
## define a new convnet including dropout
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
	input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
	optimizer=optimizers.RMSprop(lr=le-4),
	metrics=['acc'])

## use data augmentation to train
train_datagen = ImageDataGenerator(
	rescale=1./255,
	rotation_range=40,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True,)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
	train_dir,
	target_size=(150, 150),
	batch_size=32,
	class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
	validation_dir,
	target_size = (150, 150),
	batch_size = 32,
	class_mode = 'binary')

history = model.fit_generator(
	train_generator,
	steps_per_epoch= 100,
	epochs = 100,
	validation_data = validation_generator,
	validation_steps = 50)

model.save('cats_and_dogs_small_2.h5')
```

5.3 Pretrained Network
	- VGG, ResNet, Inception, Inception-ResNet, Xception
	- Feature Extraction
	- Fine-Tuning
5.3.1 Feature Extraction
``` python 
## VGG16 Example
from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
		include_top=False,
		input_shape=(150, 150, 3))
## weights indicate the initial weight check points
## include_top indicate wether the model needs include the dense connected classification
## default classification is ImageNet 1000 classifications. We are using our own classification, so we dont need include_top
## input_shape is the input shape of tensor.
```

``` python
## Fast Feature Extraction without data augmentation
import os
import numpy as np
from keras.preprocessing.image import ImangeDataGenerator

base_dir = '/Users/fchollet/Downloads/cats_and_dogs_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

def extract_features(directory, sample_count):
	feartures = np.zeros(shape=(sample_count, 4, 4, 512))
	labels = np.zeros(shape=(sample_count))
	generator = datagen.flow_from_directory(
		directory,
		target_size=(150,150),
		batch_size=batch_size,
		class_mode='binary')
	i = 0
	for inputs_batch, labels_batch in generator:
		features_batch = conv_base.predict(inputs_batch)
		features(i * batch_size : (i + 1) * batch_size) = features_batch
		labels(i * batch_size : (i + 1) * batch_size) = labels_batch
		i += 1
		if i * batch_size >= sample_count:
			break
	return features, labels

train_features, train_labels = txtract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_feature, test_labels = extract_features(test_dir, 1000)

## we extract feature as (samples, 4, 4, 512), we need translate to (samples, 8192) then input into the dense connection classification layer
train_features = np.reshape(train_features, (200, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features,  ( 1000, 4 * 4 * 512))

## Define and train dense classification
from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add(layers.Dense(256, activation = 'relu', input_dim = 4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(1r=2e-5),
			loss='binary_crossentropy',
			metrics=['acc])

history = model.fit(train_features, train_labels,
			epochs=30,
			batch_size=20,
			validation_data=(validation_features, validation_labels))
## it will be very fast since we only have two dense layers

## lets see the loss and acc curves
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation acc')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
```

Use data augmentation feature extraction
``` python
from keras import models
from keras import layers

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation= 'relu'))
model.add(layers.Dense(1, activation= 'sigmoid'))

## need freeze some layers: turn the network trainable to False
len(model.trainable_weights))
conv_base.trainable = False

## use frozen conv to train the model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

train_datagen= ImageDataGenerator(
	rescale=1./255,
	rotation_range=40,
	width_shift_range=0.2,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True,
	fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
	train_dir,
	target_size = (150, 150),
	batch_size=20,
	class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
	validation_dir,
	target_size=(150, 150),
	batch_size=20,
	class_mode='binary')

model.compile(loss='binary_crossentropy',
		optimizer=optimizers.RMSprop(1r=2e-5),
		metrics=['acc'])

history = model.fit_generator(
	train_generator,
	steps_per_epoch=100,
	epochs=30,
	validation_data=validation_generator,
	validation_steps=50)

```

Fine Tune Steps
	- add custom network on the base network
	- freeze the base net
	- train the added part
	- unfroze some layers in base net
	- train these unfrozed layers and new added layers together

``` python
## lets fine tune the last three conv layers
## freeze the layers
conv_base.trainable= True

set_trainable = False
for layer in conv_base.layers:
	if layer.name == 'block5_conv1':
		set_trainable = True
	if set_trainable:
		layer.trainable = True
	else:
		layer.trainable = False

## start fine tune
model.compile(loss = 'binary_crossentropy',
		optimizer = optimizers.RMSprop(1r=1e-5),
		metrics= ['acc']

history = model.fit_generator(
	train_generator,
	steps_per_epoch=100,
	epochs=100,
	validation_data= validation_generator,
	validation_steps= 50)

## make curve smooth
def smooth_curve(points, factor=0.8):
	smoothed_points = []
	for point in points:
		if smoothed_points:
			previous = smoothed_points[-1]
			smoothed_points.append(prevous * factor + point * (1 - factor))
		else:
			smoothed_points.append(point)
		return smoothed_points

plt.plot(epochs,
	smooth_curve(acc), 'bo', label='Smoothed training acc')
plt.plot(epochs,
	smooth_curve(val_acc), 'b', label= 'Smoothed validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs,
	smooth_curve(loss), 'bo', label='Smoothed training loss')
plt.plot(epochs,
	smooth_curve(val_loss), 'bo', label='Smoothed validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

## then evaluate the model on the test data set
test_generator = test_datagen.flow_directory(
	test_dir,
	target_size=(150,150),
	batch_size=20,
	class_mode='binary')

test_loss, test_acc = model.evaluate_generator(test_generator, steps = 50)
print('test acc: ', test_acc)

Summary
	- Convnetwork is the best model for computer vision tasks, even on the small set of data
	- Data augmentation is the method to deal with overfitting on the small data set
	- Using feature extraction, can easily re-apply new data to the existing convnetwork model
	- Fine tuning as a compementation of feature extraction, can improve the performence of the model

5.4 Convnetwork Visualization
	- Middle activation output of visuable convnet
	- Filter of visible convnet
	- type activation heat map of visible convnet
5.4.1 visualize the middle activation
	- show the output feature map(activation) of every conv layer pooling layer
## from keras.models import load_model
model = load_model('cats_and_dogs_small_2.h5')
model.summary()

## preprocessing single image
img_path = '/User/david/Downloads/cats_and_dogs_small/test/cats/cat.2700.jpg'

from keras.preprocessing import image
import numpy as np

img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255. ## remeber to preprocessing the data by scaling

print(img_tensor.shape)

## show the image
import matplotlib.pyplot as plt

plt.imshow(img_tensor[0])
plt.show()

## use one import tensor and one output tensor
from keras import models
layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs = leayer_outputs)

## run model in prediction mode
atcivations = activation_model.predict(img_tensor)
## return a list including eight Numpy metrics, each metric corresponse to a activation layer

## first_layer_activation = activation[0]
## print(first_layer_activation.shape)
## (1, 148, 148, 32)

## visualize the 4th channel
import matplotlib.pyplot as plt

plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
 
## visualize the 7th channel
plt.matshow(first_layer_activation[0,:,:, 7], cmap='viridis')

## visualize all the middle activation channels
layer_names=[]
for layer in model.layers[:8]:
	layer_names.append(layer.name)

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
	n_features = layer_acivation.shape[-1]
	
	size = layer_activation.shape[1]

	n_cols = n_features
	display_grid = np.zeros((size * n_cols, images_per_row * size))

	for col in range(n_cols):
		for row in range(images_per_row):
			channel_image = layer_activation[0,:,:, col * images_per_row + row]
			
			channel_image -= channel_image.mean()
			channel_image /= channel_image.std()
			channel_image *= 64
			channel_image += 128
			channel_image = np.clip(channel_image, 0, 255).astype('uint8')
			display_grid[col * size : (col + 1) * size,
				row * size : (row + 1) * size] = channel _image
			scale = 1. / size
			plt.figure(figsize = (scale * display_grid.shape[1],
						scale * display_grid.shape[0]))
			plt.title(layer_name)
			plt.grid(False)
			plt.imshow(display_grid, aspect= 'auto', cmap = 'viridis')
			
```
Summary
	- First layer is the collection of all kinds of boundary dectors, activation almost maintain most of the original information of the image
	- With more layers, activation becomes more abstract and hard to understand. They start to represent higher hierarchies of concepts, like cat ear and cat eyes. With deeper the layer is, the less visual information but catorgory information
	- Sparsity of activation become more with the layer deeper. In first layer, all the filter are being activated by input image. At the end of the layers, more and more blank filters. In other words, Input image can not find these filter's coding mode

	- With the deeper layer goes, the extraced features get more abstract. Higher the layer is, less the input information is, more target information is. We call it Information Distillation Pipeline. Similar as human brain function.

5.4.2 the filter of visible convnetwork
## define loss tensor for filter visualization
from keras.applications import VGG16
from keras import backend as K

model = VGG16(weights='imagenet',
		include_top=False)

layer_name = 'block3_conv1'
filter_index = 0

layer_output = model.get_layer(layer_name).output
loss = K.mean(layer_output[:, :, :, filter_index])

## the gradients of loss comparing to input
grads = K.gradients(loss, model.input)[0]

## standardlize the gradients
grads /= (K.sqrt(K.mean(K.square(grads))) + le-5)

## give Numpy input, get Numpy output
iterate = K.function([model.input], [loss, grads])

import numpy as np
loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])

## Max the loss by using random gradient decrease
input_img_data = np.random((1, 150, 150, 3)) * 20 + 128.
step = 1.
for i in range(40):
	loss_value, grads_value = iterate([input_img_data])

	input_img_data += grads_value * step


## turn tensor to effective function
def deprocess_image(x):
	x -= x.mean()
	x /= (x.std() + le-5)
	x *= 0.1
	
	x += 0.5
	x = np.clip(x, 0, 1)

	x *= 255
	x = np.clip(x, 0, 255).astype('uint8')
	return x

## funtion to visualize the filter
def generate_pattern(layer_name, filter_index, size=150):
	layer_output = model.get_layer(layer_name).output
	loss = K.mean(layer_output[:, :, :, filter_index])

	grads = K.gradients(loss, model.input)[0]
	grads /= (K.sqrt(K.mean(K.square(grads))) + le-5)
	iterate = K.function([model.input], [loss, grads])
	input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

	step = 1.
	for i in range(40):
		loss_value, grads_value = iterate([inpu_img_data])
		input_img_data += grads_value * step

	img =input_img_data[0]
	return deprocess_image(img)


## generate a network of all filter response modes in one layer
layer_name = 'block1_conv1'
size = 64
margin = 5

results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))

for i in range(8):
	for j in range(8):
		filter_img = generate_pattern(layer_name, i + (j * 8), size = size)
		horizontal_start = i * szie + i * margin
		horizontal_end = horizontal_start + size
		vertical_start = j * size + j * margin
		vertical_end = vertical_start + size
		results[horizontal_start: horizontal_end,
			vertical_start: vertical_end, :] = filter_img

plt.figure(fisize= (20, 20))
plt.imshow(results)
```
Summary
	- the filter of fiest layer (block1_conv1) response to edge, color, and direction
	- block2_conv1 corresponse to edge color groups and simple pattern
	- higher layers filter response to natural image pattern like: feature, eyes, leaf, and so on.

5.4.3 visible activation heat map
CAM, class activation map
Grad-CAM: visual expanations from deep networks via gradient-based localization

## load pre-trained VGG16 network
from keras.applications.vgg16 import VGG16

model = VGG16(weights='imagenet')

## pre-process an image by VGG16
from keras.preprocessing import image 
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

img_path ='...'
img = image.load_img(img_path, targe_size = (224, 224))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', u'African_elephant', 0.92546833), (u'n01871265', u'tusker', 0.070257246), (u'n02504013', u'Indian_elephant', 0.0042589349)]

## apply Grad-CAM method
african_elephant_output = model.output[:, 386]
last_conv_layer = model.get_layer('block5_conv3')

grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

pooled_grads_value, conv_layer_output_value = iterate([x])

for i in range(512):
	conv_layer_output_value[:,:,i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)

## Use OpenCV to generate an overlay image
import cv2
img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite('...', superimposed_img)
```

Chapter Summary
	- Convnet is the best tool for visual catergorization
	- Convnet represent visual world through learning the modes and layers
	- Convnet is not dark box, but easy to visualize
	- Now we can train our own convnet model to solve image catergorize problem
	- You know how to deal with overfitting by data augmentation
	- Pre-trained network to feature extraction and fine tuning
	- Visualize the filter/activation learned from network

