import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import model_from_json
import numpy
import os

# Image size
img_rows, img_cols = 28, 28
usingMLP = False

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# Compile the model
loaded_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# Check if the model is ok

from keras.datasets import mnist

num_classes = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()

if usingMLP:
    x_test = x_test.reshape(10000, 784)
    input_shape = (784,)
else:
    # Adjuse channel position basing on the backend
    # Every CNN layer has to know where the number of channels is
    # It can be channel_first or channel_last
    if K.image_data_format() == 'channels_first':
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

y_test = keras.utils.to_categorical(y_test, num_classes)

score = loaded_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# Return the array of NN predictions
def getStasFromImageNumber(img):
  if usingMLP:
      img = img.reshape(1, 784)
  else:
      # Adjuse channel position basing on the backend
      # Every CNN layer has to know where the number of channels is
      # It can be channel_first or channel_last
      if K.image_data_format() == 'channels_first':
        img = img.reshape(1, 1, img_rows, img_cols)
      else:
        img = img.reshape(1, img_rows, img_cols, 1)
  
  img = img.astype('float32')
  img /= 255

  return loaded_model.predict(img)

# Converts the predictions array into a number
def catToNum(arr):
  high = 0
  highIndex = 0 # The number
  i = 0
  for num in arr[0]:
      if num > high:
          high = num
          highIndex = i
      i += 1
  return highIndex