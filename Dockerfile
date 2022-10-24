FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y libglib2.0-0

RUN apt-get update \
    && apt-get install -y \
    build-essential \
    curl \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-dev \
    python3.8 \
    python3-pip \
    && ln -s /usr/bin/python3.8 /usr/local/bin/python

WORKDIR mvs

COPY . ./

RUN pip3 install -r requirements.txt

CMD ["python",  "eval_rcmvsnet_tanks.py", "--split", "intermediate", "--loadckpt", "./pretrain/model_000014_cas.ckpt", "--plydir", "./tanks_submission", "--outdir", "./tanks_exp", "--testpath", "tankandtemples"]
