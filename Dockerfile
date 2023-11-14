FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

RUN cd /opt 
WORKDIR /opt 

RUN apt-get update && apt-get install -y wget git ffmpeg libsm6 libxext6 gcc ninja-build 

COPY . /opt/performancereidTrainLITE

COPY ./Anaconda3-2023.09-0-Linux-x86_64.sh /opt/Anaconda3-2023.09-0-Linux-x86_64.sh

# RUN wget https://repo.anaconda.com/archive/Anaconda3-2023.07-2-Linux-x86_64.sh 

RUN bash Anaconda3-2023.09-0-Linux-x86_64.sh -b -p ./anaconda3

RUN bash -c "source /opt/anaconda3/etc/profile.d/conda.sh && conda create --name reid -y python=3.7 && conda activate reid && cd /opt/performancereidTrainLITE && pip install -r requirements.txt && conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y && python setup.py develop"


# git clone https://github.com/KaiyangZhou/deep-person-reid.git &&
#    cd deep-person-reid/ &&
