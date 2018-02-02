# What is this

Online MNIST is a Web App which, with the help of CNNs, can predict the number you wrote on a canvas

# How to install

Install the conda environment. Note that the envyronment uses tensorflow-gpu, so if you want to install this environment you will have to install CUDA & CuDNN.

conda env create -f webDL.yml

If you don't have a compatible GPU install the following packages via pip or similar:

- Flask
- flask_cors
- opencv
- numpy
- json
- Keras
- Tensorflow
- PILLOW (PIL)
- base64
- StringIO (io)

# How to run

After you have installed all required libraries, enable the environment and run the main script

source activate webDL

python main.py

Then open your browser and go to http://127.0.0.1:5000
