FROM python:3.8-slim-bullseye

ARG ABS_M_VERSION="v0_0"
ARG ABS_Z_VERSION="v0_0"

RUN apt-get update && \
    apt-get install -y git libturbojpeg0=1:2.0.6-4

RUN git clone https://github.com/geospaitial-lab/abs --depth 1 && \
    python -m pip install -r /abs/requirements.txt --ignore-installed --no-warn-script-location --upgrade

WORKDIR /abs

ENV PYTHONPATH "${PYTHONPATH}:/abs"
ENV PYTHONUNBUFFERED=1

RUN huggingface-cli download geospaitial-lab/abs_m "models/abs_m_${ABS_M_VERSION}.onnx" --local-dir data && \
    huggingface-cli download geospaitial-lab/abs_z "models/abs_z_${ABS_Z_VERSION}.onnx" --local-dir data

RUN mv data/models/abs_m_${ABS_M_VERSION}.onnx data/models/abs_m.onnx && \
    mv data/models/abs_z_${ABS_Z_VERSION}.onnx data/models/abs_z.onnx

CMD ["python", "/abs/src/main.py", "/config.yaml"]