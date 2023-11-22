FROM nvidia/cuda:11.0.3-runtime-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

RUN apt-get update -y
RUN apt-get --fix-missing install -y wget build-essential checkinstall  libreadline-gplv2-dev  libncursesw5-dev  libssl-dev  libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev libgl1-mesa-dev libsndfile1 libglib2.0-0
RUN apt-get install -y \
    git \
    curl \
    python3.7 \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117

WORKDIR /home/ofa
ADD  requirements.txt .

RUN pip3 install -r requirements.txt