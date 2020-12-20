import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras import optimizers
from keras.layers import Activation, Conv2D, MaxPooling2D

datadir = "./images/"
categories = ["French_bulldog", "Chihuahua", "Golden_retriever", "Maltese_dog", "Miniature_Dachshund", "Saint_Bernard", "Shiba", "Shih_Tzu", "Toypoodle", "Yorkshire_terrier"]
class_num = len(categories)

X = []
Y = []

for index, category in enumerate(categories):
	path = os.path.join(datadir, category)
	#隠しファイルは読まないため
	data_list = [data for data in os.listdir(path) if not data.startswith('.')]
	for data in data_list:
		data = path + "/" + data
		img = cv2.imread(data)
		img = cv2.resize(img, (64, 64))
		X.append(img)
		Y.append(index)

X = np.array(X)
Y = np.array(Y)

rand_index = np.random.permutation(np.arange(len(X)))
X = X[rand_index]
Y = Y[rand_index]

X_train, X_test, y_train, y_test = train_test_split(X, Y)

X_train = X_train.astype("float") / 255
X_test  = X_test.astype("float")  / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# モデルの定義
model = Sequential()
model.add(Conv2D(input_shape=(64, 64, 3), filters=32,kernel_size=(3, 3),
				 strides=(1, 1), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(3, 3),
				 strides=(1, 1), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(3, 3),
				 strides=(1, 1), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation("sigmoid"))
model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dense(class_num))
model.add(Activation('softmax'))

model.compile(loss="categorical_crossentropy",
optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=["accuracy"])

history = model.fit(X_train, y_train, batch_size=16, epochs=100, validation_data=(X_test, y_test))

model.save('dog_model.h5')

#acc, val_accのプロット
plt.plot(history.history["acc"], label="acc", ls="-", marker="o")
plt.plot(history.history["val_acc"], label="val_acc", ls="-", marker="x")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(loc="best")
plt.show()

score = model.evaluate(X_test, y_test, verbose=0)
print("evaluate loss: {0[0]}\nevaluate acc: {0[1]}".format(score))

#model = load_model('dog_model.h5')
