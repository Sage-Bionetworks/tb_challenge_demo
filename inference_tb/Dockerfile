FROM jupyter/scipy-notebook
FROM python:3.9-slim-buster

RUN pip install joblib
RUN pip install Pandas
RUN pip install onnxruntime

RUN mkdir model

COPY model ./model/
COPY inference.py ./inference.py
COPY entrypoint.sh ./entrypoint.sh

ENTRYPOINT ["sh", "./entrypoint.sh"]