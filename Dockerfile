FROM ubuntu:20.04
ENV DEBIAN_FRONTEND noninteractive
RUN apt update
RUN DEBIAN_FRONTEND="noninteractive" apt-get install vim wget nano curl git git-lfs ca-certificates -y
WORKDIR /root/
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh
RUN chmod +x Miniconda3-py38_4.10.3-Linux-x86_64.sh
RUN /root/Miniconda3-py38_4.10.3-Linux-x86_64.sh -b && eval "$(/root/miniconda3/bin/conda shell.bash hook)" && /root/miniconda3/bin/conda clean -afy
RUN /root/miniconda3/bin/conda init
RUN /root/miniconda3/bin/conda run -n base pip install tensorflow==2.6.2
#TUTORIAL
RUN apt-get upgrade -y
RUN apt-get install -y build-essential cmake unzip pkg-config
RUN apt-get install -y libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev
RUN apt-get install -y libjpeg-dev libpng-dev libtiff-dev
RUN apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
RUN apt-get install -y libxvidcore-dev libx264-dev
RUN apt-get install -y libgtk-3-dev
RUN apt-get install -y libopenblas-dev libatlas-base-dev liblapack-dev gfortran
RUN apt-get install -y libhdf5-serial-dev graphviz
RUN apt-get install -y python3-dev python3-tk python-imaging-tk
RUN apt-get install -y linux-image-generic linux-image-extra-virtual
RUN apt-get install -y linux-source linux-headers-generic
RUN apt-get update
RUN /root/miniconda3/bin/conda run -n base  conda install -c conda-forge cudatoolkit==11.2.2
RUN /root/miniconda3/bin/conda run -n base conda install -c conda-forge cudnn==8.1.0.77
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute
WORKDIR /
COPY requirements.txt /requirements.txt
RUN /root/miniconda3/bin/conda run -n base pip install -r requirements.txt
WORKDIR /
RUN mkdir workspace
WORKDIR /workspace
COPY mrcnn mrcnn
COPY setup.py setup.py
RUN /root/miniconda3/bin/conda run -n base python setup.py install
RUN /root/miniconda3/bin/conda run -n base pip install -U scikit-image==0.16.2
RUN /root/miniconda3/bin/conda run -n base conda install -c conda-forge imgaug


# Build: docker build -t maskrcnn:new .

# docker run -it --gpus all -v $PWD/:/workspace maskrcnn:new bash
