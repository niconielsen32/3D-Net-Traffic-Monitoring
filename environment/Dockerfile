# hash:sha256:e5f1d4b1e5161e6090f60750cd49defe0a30ae64906a74173aa72412e1faaa94
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-cuda11.7.0-cudnn8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        jupyter==1.0.0 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    filterpy==1.4.5 \
    ipython==8.12.0 \
    matplotlib==3.7.1 \
    numpy==1.24.2 \
    opencv-python==4.7.0.72 \
    pandas==2.0.0 \
    psutil==5.9.5 \
    pyyaml==6.0 \
    seaborn==0.12.2 \
    tqdm==4.65.0

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install -U --no-cache-dir \
    torch==2.0.0 \
    torchvision==0.15.1

COPY postInstall /
RUN /postInstall
