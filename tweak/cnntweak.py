# Baseline MLP for MNIST dataset
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
# define baseline model
def baseline_model(neuron):
	# create model
	model = Sequential()
	model.add(Dense(neuron, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
# build the model
neuron = 5
model = baseline_model(neuron)

accuracy = 0.0

def buildModel():
	# Fit the model
	model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=200, verbose=0)
	# Final evaluation of the model
	scores = model.evaluate(X_test, y_test, verbose=0)
	accuracy = scores[1]*100
	print("Accuracy: %.2f%%" % (scores[1]*100))
	return accuracy

buildModel()
count = 0
best_acc = accuracy
best_neuron = 0

def resetWeights():
	print("Reseting weights")
	w = model.get_weights()
	w = [[j*0 for j in i] for i in w]
	model.set_weights(w)


while accuracy < 99 and count < 4:
	print("Updating Model")
	model = baseline_model(neuron*2)
	neuron = neuron * 2
	count = count + 1
	accuracy = buildModel()
	if best_acc < accuracy:
		best_acc = accuracy
		best_neuron = neuron
	print()
	resetWeights()
 
print("***********")
# resetWeights()
print(best_neuron)
model = baseline_model(best_neuron)
buildModel()
model.save('mnist_model_update.h5')
print("Model Saved")

file1 = open("result.txt","w")
file1.write(str(best_acc))
file1.close()