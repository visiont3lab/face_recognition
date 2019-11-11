# ------- Build Command docker build -t img-starter -f d-starter .
# Simply, because you cannot have multiple base images in a Dockerfile
# https://runnable.com/blog/9-common-dockerfile-mistakes

#-------- Starter developing image

FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ENV PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# ------- Installation Web Server dependendencies

RUN apt-get update && apt-get install -y \
    git \
    cmake \
    vim \
    net-tools  \
    wget \
    htop \
    unzip \
    libsm6  \
    libxext6 \
    libxrender-dev \
    libgtk2.0-dev

ENV HOME=/root

# ------- Installation Python dependendencies

RUN apt-get update && apt-get install -y python3-pip python3-dev libmysqlclient-dev

RUN /bin/bash -c  "pip3 install numpy \
                        scipy  \
			tornado \
 			imutils \
			pyinotify \
			mysqlclient \
                        pandas \
                        matplotlib \
                        scikit-learn \
                        Pillow cython  \
                        opencv-python \
                        opencv-contrib-python \
                        tensorflow-gpu==1.14 \
			face_recognition"

#------------------ DLIB GPU SETUP 
RUN /bin/bash -c  "pip3 uninstall -y dlib"
RUN apt-get install -y libopenblas-dev liblapack-dev
RUN /bin/bash -c  "mkdir -p /root/home/lib && cd /root/home/lib && git clone https://github.com/davisking/dlib.git"
RUN /bin/bash -c  "cd /root/home/lib/dlib &&  python3 setup.py install  --set CUDA_HOST_COMPILER=/usr/bin/gcc-7 --clean"

# xhost +local:docker &&  docker run --runtime=nvidia  --rm -it --name deep_learning_face_recognition  --env="DISPLAY=$DISPLAY" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --device=/dev/video0  -v $HOME/visiont3lab-github/face_recognition:/root/home/ws deep-learning:face_recognition  /bin/bash -c "cd /root/home/ws/algorithm/ && python3 custom_face_recognition.py"
# We need to compile in runtime mode:  cd /root/home/lib &&  python3 setup.py install  --set CUDA_HOST_COMPILER=/usr/bin/gcc --clean
 
