import numpy as np
import pandas as pd
import tensorflow as tf
# from keras.layers import BatchNormalization
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Convolution2D, Flatten, Dense, Dropout, MaxPool2D
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.models import Sequential

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

trainData = pd.read_csv("training.csv")
testData = pd.read_csv("test.csv")
valData = pd.read_csv("IdLookupTable.csv")

trainData.fillna(method="ffill", inplace=True)
images = []

for i in range(len(trainData)):
    img = trainData["Image"][i].split(" ")
    img = ["0" if x == "" else x for x in img]
    images.append(img)

imageList = np.array(images, dtype="float")
XTrain = imageList.reshape(-1, 96, 96, 1)

training = trainData.drop("Image", axis=1)
YTrain = []

for i in range(len(training)):
    y = training.iloc[i, :]
    YTrain.append(y)

YTrain = np.array(YTrain, dtype="float")

test_images = []

for i in range(len(testData)):
    test_img = testData["Image"][i].split(" ")
    test_img = ["0" if x == "" else x for x in test_img]
    test_images.append(test_img)

test_imageList = np.array(test_images, dtype="float")
XTest = test_imageList.reshape(-1, 96, 96, 1)

model = Sequential()

model.add(Convolution2D(32, (3, 3), padding='same', use_bias=False, input_shape=(96, 96, 1)))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())

model.add(Convolution2D(32, (3, 3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(64, (3, 3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())

model.add(Convolution2D(64, (3, 3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(96, (3, 3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())

model.add(Convolution2D(96, (3, 3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(128, (3, 3), padding='same', use_bias=False))
# model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())

model.add(Convolution2D(128, (3, 3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(256, (3, 3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())

model.add(Convolution2D(256, (3, 3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(512, (3, 3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())

model.add(Convolution2D(512, (3, 3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(30))

model.compile(loss="mean_squared_error", optimizer="Adam", metrics=["accuracy"])
model.fit(XTrain, YTrain, batch_size=256, epochs=50, validation_split=0.1, verbose=1)

model.save("facepoints_cuda.h5")

prediction = model.predict(XTest)
valDataList = list(valData["FeatureName"])
imageId = list(valData["ImageId"] - 1)
predictionList = list(prediction)
rowId = list(valData["RowId"])

feature = []
for f in valDataList:
    feature.append(valDataList.index(f))

predicted = []

for x, y in zip(imageId, feature):
    predicted.append(predictionList[x][y])

rowId = pd.Series(rowId, name="rowId")
loc = pd.Series(predicted, name="Location")
result = pd.concat([rowId, loc], axis=1)
result.to_csv("result.csv", index=False)
