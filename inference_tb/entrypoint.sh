#!/bin/sh

ln -sf model/tb_recognition.onnx /output
python inference.py
