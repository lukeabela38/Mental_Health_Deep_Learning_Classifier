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
RUN python3 -m pip install nltk
RUN python3 -m pip install pandas
RUN python3 -m pip install gradio
RUN python3 -m pip install flask==2.0.3
RUN python3 -m pip install waitress==2.0.0
RUN apt update

EXPOSE 8001

CMD ["python3", "app.py"]