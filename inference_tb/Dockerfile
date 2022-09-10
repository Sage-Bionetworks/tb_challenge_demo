FROM jupyter/scipy-notebook
FROM python:3.9-slim-buster

RUN pip install joblib
RUN pip install Pandas
RUN pip install onnxruntime

RUN mkdir model

COPY model ./model/
COPY inference.py ./inference.py


ENTRYPOINT ["python3", "./inference.py"]

USER root
RUN mkdir -p /output
RUN mkdir -p /input