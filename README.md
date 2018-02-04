# What is this

Online MNIST is a Web App which, with the help of CNNs, can predict the number you wrote on a canvas

# How to install

Install the conda environment. If you have a compatible GPU, install the GPU environment else the CPU. If for any reasons the GPU env doesn't work, try installing CUDA and CUDnn, following the tutorial on https://www.tensorflow.org/install/. 
To check if the GPU is correctly being used, run `nvidia-smi` on a terminal and check if Python is listed and all the RAM of the GPU is used.

`conda env create -f webDL-CPU.yml`

`conda env create -f webDL-GPU.yml`

If you don't have Anaconda, install the following packages via pip or similar:

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

After you have installed the environment, enable it:

`source activate webDL`

and run the main script:

`python main.py`

Then open your browser and go to http://127.0.0.1:5000
