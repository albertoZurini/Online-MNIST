from flask import Flask, request, send_from_directory, Response
from flask_cors import CORS, cross_origin

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

import cv2
import numpy as np
import json

import neuralNetwork.image as image
import neuralNetwork.predict as nn

app = Flask(__name__, static_url_path='')
CORS(app)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
  img = request.form['img']
  img = str(img)

  img = img.split(',')[1] # Extract base64
  img = image.getImg(img) # Converts into RGB

  img = cv2.resize(img, (28, 28)) # Resize the image for the neural network
  img = 255 - img # Black = white and vice versa

  prediction = nn.getStasFromImageNumber(img) # Predict the number
  num = nn.catToNum(prediction) # Extract the number

  ret = '{"val": '+str(num)+', "stats": '+json.dumps(prediction[0].tolist())+'}' # The output is JSON
  resp = Response(response=ret,
                    status=200,
                    mimetype="application/json")
  return resp

@app.route('/<path:path>')
def retFile(path):
  return send_from_directory('static', path)

@app.route('/')
def root():
  return app.send_static_file('index.html')

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000, debug=False)