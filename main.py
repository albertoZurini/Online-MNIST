from flask import Flask, request, send_from_directory, Response
from flask_cors import CORS, cross_origin

import cv2
import numpy as np
import json

from neuralNetwork.image import *
from neuralNetwork.predict import *

app = Flask(__name__, static_url_path='')
CORS(app)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
  img = request.form['img']
  img = str(img)

  img = img.split(',')[1] # Extract base64
  img = getImg(img) # Converts into RGB

  img = cv2.resize(img, (28, 28)) # Resize the image for the neural network

  prediction = getStasFromImageNumber(img) # Predict the number
  num = catToNum(prediction) # Extract the number

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
  app.run(debug=False)