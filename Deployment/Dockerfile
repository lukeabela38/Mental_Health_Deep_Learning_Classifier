FROM python:3.7-alpine
FROM tensorflow/tensorflow:latest

RUN apt install -y curl
RUN apt install -y vim

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade setuptools

WORKDIR /workspace

COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt
RUN apt update

COPY . .

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

EXPOSE 8000
EXPOSE 8080

CMD ["python3", "app.py"]
