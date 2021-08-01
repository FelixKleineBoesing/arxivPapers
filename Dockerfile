FROM python:3.7

RUN mkdir /app
WORKDIR /app

COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt

RUN pip3 install -r requirements_dev.txt
RUN pip3 install -r requirements.txt

RUN pip3 install .
RUN pip3 uninstall -r requirements_dev.txt