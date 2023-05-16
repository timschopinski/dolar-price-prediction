FROM --platform=linux/amd64 python:3.10-slim
MAINTAINER Tim Schopinski

ENV PYTHONUNBUFFERED 1

RUN pip install --upgrade pip

COPY ./requirements.txt /requirements.txt
COPY ./scripts /scripts

RUN pip3 install -r requirements.txt
RUN mkdir /app
WORKDIR /app
COPY ./app /app

RUN adduser user
RUN chmod -R +x /scripts

ENV PATH="/scripts:/py/bin:$PATH"

USER user
