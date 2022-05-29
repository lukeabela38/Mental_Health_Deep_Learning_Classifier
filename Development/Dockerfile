FROM python:3.7-alpine
FROM tensorflow/tensorflow:latest

RUN apt install -y curl
RUN apt install -y vim

WORKDIR /workspace

COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt
RUN apt update

COPY development_files .

EXPOSE 8888

CMD ["python3", "main.py"]
