# everything except the gpu: python stack, host toolchain, arm cross compiler
# and qemu. tensorrt and torch pull their own cuda wheels from pypi, so the
# image needs no cuda base, just --gpus all and a driver on the host:
#
#   docker build -t edgeinfer .
#   docker run --rm -it --gpus all -v $PWD:/work edgeinfer make smoke-export c-engine test
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        build-essential make curl ca-certificates git \
        gcc-arm-none-eabi libnewlib-arm-none-eabi qemu-system-arm \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY requirements.txt requirements-backends.txt ./
# ubuntu 24.04 ships python 3.12 and marks it externally managed.
# the backends file pulls requirements.txt in, so this is the full stack
RUN pip3 install --break-system-packages -r requirements-backends.txt

COPY . .
CMD ["make", "help"]
