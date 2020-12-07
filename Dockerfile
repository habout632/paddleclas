# FROM defines the base image
#FROM nvidia/cuda:10.1-devel
FROM nvcr.io/nvidia/tensorrt:20.03-py3
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-setuptools
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y
#RUN apt install libgl1-mesa-glx -y
RUN ln -s -f /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
ENV LISTEN_PORT 6666
COPY requirements.txt /app/requirements.txt
RUN  pip3 install -i https://mirror.baidu.com/pypi/simple --upgrade pip
RUN  pip3 install -i https://mirror.baidu.com/pypi/simple -r /app/requirements.txt
#RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple -r /app/requirements.txt
#RUN pip3 install -i https://mirror.baidu.com/pypi/simple -r /app/requirements.txt
COPY ./tools /app/tools
COPY ./inference /app/inference
ENV PYTHONPATH=/
WORKDIR /app

#RUN make
#RUN apt-get update && apt-get install -y --no-install-recommends \
#        cuda-samples-$CUDA_PKG_VERSION && \
#    rm -rf /var/lib/apt/lists/*

# set the working directory
#WORKDIR /usr/local/cuda/samples/1_Utilities/deviceQuery
# CMD defines the default command to be run in the container
# CMD is overridden by supplying a command + arguments to
# `docker run`, e.g. `nvcc --version` or `bash`
#CMD ./deviceQuery
