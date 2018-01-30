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

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# Evaluate loaded model on test data
loaded_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# Return the array of NN predictions
def getStasFromImageNumber(img):
  if K.image_data_format() == 'channels_first':
    img = img.reshape(1, 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
  else:
    img = img.reshape(1, img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

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