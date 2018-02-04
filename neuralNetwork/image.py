import io
import cv2
import base64 
import numpy as np
from PIL import Image

# Take in base64 string and return PIL image
def stringToImage(base64_string):
  imgdata = base64.b64decode(base64_string)
  return Image.open(io.BytesIO(imgdata))

# convert PIL Image to a gray scale image( technically a numpy array ) that's compatible with opencv
def toGray(image):
  return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)

def getImg(img):
  img = stringToImage(img)
  img = toGray(img)
  return img