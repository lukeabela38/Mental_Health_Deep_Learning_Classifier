FROM tensorflow/tensorflow:latest

RUN apt-get -y update && apt-get install -y \
    python3-dev \
    python3-pip \
python3-setuptools

RUN apt install -y curl
RUN apt install -y vim

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade setuptools

RUN python3 -m pip install numpy
RUN python3 -m pip install tqdm
RUN python3 -m pip install nltk
RUN python3 -m pip install transformers

RUN apt update
