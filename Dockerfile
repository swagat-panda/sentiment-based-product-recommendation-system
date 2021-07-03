FROM python:3.7-slim
EXPOSE 8000

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_md

CMD gunicorn -b 0.0.0.0:8000 server:app