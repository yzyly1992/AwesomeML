4.1.1 Supervised Learning
	- Category Problem
	- Regression Problem
	- Sequence Generation: Giving an image, predict the image description text. It also could be represented as a serious of category problems
	- Syntax Tree Prediction: Given a sentence, predict its structure grammer tree
	- Object Detection: Given an image, draw a boundary around target object. It also could be represented as category problem.
	- Image segmentation: given a image, draw a pixel mask on certain object

4.1.2 Non-Supervised Learning
	- Def: Find interesting transformation of input data without specific purpose and target. It can help data visualization, data compress, data noise-reduce, or better understand data relationship.
	- Dimensionality
	- Clustering

4.1.3 Self-Supervised Learning
	- Autoencoder
	- Temporally Supervised Learning

4.1.4 Reinforce Learning
	- Game, AutoDrive, Robot, Resource Management, Education
	- Agent recieve the environment info, and study the maximum reward action

4.2 Evaluate Machine Learning Model
4.2.1 Train Set, Val Set, Test Set
	- Why we have validation set is because we need adjust parameter according to the feedback of val set test result.
	- if we use the test set to adjust the parameter, the model will overfitting quickly on the test set.
	- We call it information leak
	- 1. Simplely hold-out validation
``` python
num_validation_samples = 10000

## usually we will randomlize the data set
np.random.shuffle(data)

validation_data = data[:num_validation_samples]
data = data[num_validation_samples:]

training_data =  data[:]

model = get_model()
model.train(training_data)
validation_score = model.evaluate(validation_data)

## then we can adjust model, train it again, evaluate it and adjust ...
## once finishing adjust parameter, we will retrain all the train set and val set, and test it on the test set.
model = get_model()
model.train(np.concatenate([training_data, validation_data]))
test_score = model.evaluate(test_data)	
```
	- 2. K-fold validation
``` python
k = 2
num_validation_samples = len(data) // k

np.random.shuffle(data)

validation_scores = []
for fold in range(k):
	validation_data = data[num_validation_samples * fold:
	num_validation_samples * (fold + 1)]
	training_data = data[:num_validation_samples * fold] + 
	data[num_validation_samples * (fold + 1):]

	model = get_model()
	model.train(training_data)
	validation_score = model.evaluate(validation_data)
	validation_scores.append(validation_scores)

validation_score = np.average(validation_scores)

model = get_model()
model.train(data)
test_score = model.evaluate(test_data)
```

	- Iterated K-fold validation with shuffling
	- Before divide data into K part, we shuffle all the data first
	- The epochs will be P x K ( P is the repeat time)
	- There will be lot more calculation, but very popular and effective on Kaggle

4.2.2 Attentions on Evaluating Model
	- Data Representativeness: usually we need shuffle the data, or it could be bad results because the order of the data
	- The arrow of time: make sure the data is in order, or it will have temporal leak.

	- Redundancy in your data: make sure there is no cross  between training set and validation set

4.3 data preprocessing, feature engineer study
4.3.1 Preprocessing the neural network data
	-1. Vectorization -- turn all kinds of data into tensor. We can use one-hot code turn data into float32 tensor
	- 2. Standardlize data: Everage is 0, standard difference is 1.
	- 3. Heterogeneous data / Homogenous data
```
x -= x.mean(axis=0)
x /= x.std(axis=0)
```
	- 4. Absent data: we usually set absent data to 0, if 0 is not a meaningful value.

4.3.2 Feature Engineering
	- Def: before inputing your data into the model, according to your knowledge or human experience to tranform the data so that can improve the effectiveness of the network model

4.4 Overfit and Underfit
	- Optimization: adjust parameter to get best performence.
	- Generalization: Evaluate trained model on unkown data
	- The more data is, better the generalization is. Or you need to limit the parameter to get better gemeralization. 
	- We call this process Regularization

4.4.1 Reduce the network size
	- The more parameter, the larger memorization capacity the model has. It can easily learn the dictionary relation perfectly. However, this relation does not have generalizaton ability.
	- You should have enough parameter to avoid underfit and lack of memorization capacity, and you also should not have too much parameter to avoid not generalization enough.
	- Test on the movie comment categorize example
``` python
## original model
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```
``` python
## smaller capacity model
model = models.Sequential()
model.add(layers.Dense(4, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
## smaller capacity model get overfit late than original model. The performence reduce speed after overfit is slower than the original model
```

``` python
## bigger capacity model
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
## overfit really bad, val loss very unstable
```
4.4.2 Adding Weight and Regularization
	- Occam's razor: If a thing has two explain, the most possible one is the simpler one
	- Same principle: simpler model is not easy to overfit than complex one
	- Method 1: Weight regularization: force weight only can pick smaller value, and make weight location more regular
	- We call it weigh regularization
	- The method is to add related cost into loss function when have bigger weight
	- L1 Regularization: linear relation between cost in loss function and abs of weight value
	- L2 Regularization: linear relation between cost in loss function and squ of weight value / Weigth Decay

``` python
## add L2 regularization in to model
from keras import regularizers

model = models.Sequential()
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
			activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
			activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
## l2(0.001) means this layers weight metric, every parameter will make whole network lose 0.001 * weight_coefficient_value
## l2 regularization could make model not easy to get overfit
```
``` python
## could also use l1, or l1 and l2
regularizers.l1(0.001)
regularizers.l1_l2(l1=0.001, l2=0.001)
````

4.4.3 Adding Dropout Regularization
	- Def: Use dropout for a layer, it will reset certain  feature to 0. Dropout rate usually is 0.2~0.5
	- When test, there is no unit to drop, this layer output should reduce same as the dropout rate. And there would be more units being actived.
``` python
## at test
layer_output *= 0.5

## at training
layer_output *=np.random.randint(0, high=2, size=layer_output.shape)
layer_output /=0.5
```
``` python
## adding dropout to IMDB
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
## it can effectively improve the performence
```
Summary: Methods of Preventing Overfitting
	- Get more data
	- Reduce Network Capacity
	- Add Weight Regularization
	- Add Dropout

4.5 General Process of Machine Learning
4.5.1 Define Problems, Collect Data Set
4.5.2 Choose the Success Indication: Precisioin? Recall Rate? ...
4.5.3 Decide Evaluation Methods: Left Validation Set, K-fold Validation, Repeat K-fold
4.5.4 Prepare Data: Translat to tensor, standardlize, feature engineer
4.5.5 Develop Better Model: Last layer of activation function, loss function, optimizer
4.5.6 Enlarge Model Size: Develop Overfit Model
4.5.7 Model regularization and Parameter Adjustment: L1 / L2, add or reduce layer, add dropout, repeat feature engineer


