FROM python:3.8-slim-bullseye

RUN apt-get update && \
    apt-get install -y libturbojpeg0=1:2.0.6-4

RUN mkdir /abs

COPY . /abs

RUN python -m pip install -r /abs/requirements.txt --ignore-installed --no-warn-script-location --upgrade

WORKDIR /abs

ENV PYTHONPATH "${PYTHONPATH}:/abs"
ENV PYTHONUNBUFFERED=1

CMD ["python", "/abs/src/main.py", "/config.yaml"]