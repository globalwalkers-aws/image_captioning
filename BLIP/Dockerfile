# This is the DockerFile used for ITSP REID Phase-3 system #
# This environment make use of Ubuntu20.04 with Cuda 11.1  #

FROM nvidia/cuda:11.3.1-base-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install essentials
RUN apt-get update
RUN apt update -y && apt upgrade -y && \
    apt-get --fix-missing install -y wget build-essential checkinstall  libreadline-gplv2-dev  libncursesw5-dev  libssl-dev  libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev

# install python
ARG PYTHON_VERSION
RUN apt-get install ${PYTHON_VERSION} -y
RUN apt -y install python3-pip


# install libraries for opencv
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# install libraries for git and curl
RUN apt-get install git curl -y

# install torch
ARG TORCH_VERSION
ARG TORCHVISION_VERSION
RUN pip install torch==${TORCH_VERSION} -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torchvision==${TORCHVISION_VERSION} 
# install system libraries from requirements.txt
COPY requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache \ 
    python3 -m pip install -r requirements.txt



