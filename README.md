# tb_challenge_demo
This is a python based dummy example for CODA TB challenge. It is meant to showcase how participants can:
  - Train their model using any ML framework.
  - Convert trained model in ONNX format model.
  - Inference on ONNX format model.

### Installation
Install required modelues based on the choice of ML framework. For this dummy example install below modelues:
  - Tensorflow
  - Keras
  - tf2onnx
  - onnxruntime

### Training and Inference
In this dummy example we are using Keras to train our model. Inference should be done on ONNX format model. So,
  - Convert any framework trained model into ONNX format. ( Using tf2onnx in this case)
  - Prepare a docker image that will include:
    - ONNX format trained model under directory name 'model' (as suggested in the example)
    - Inference script that will output prediction.csv (save it under '/output' directory location)