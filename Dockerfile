FROM python:3.7

RUN mkdir /app
WORKDIR /app

COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt