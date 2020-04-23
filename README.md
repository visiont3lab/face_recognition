# Face Recognition

## Introduction

This repository is an example of how to use the [well known Face Recognition Github repository](https://github.com/ageitgey/face_recognition).
In particular this repository contains:

* "algorithm/custom_face_recognition.py" : This python script perform real time face recognition. The face that the algorithm is going to recognize are contained inside the folder "common/faces"
* "common/utils/takeAPicture.py" : It is a simple script that uses opencv to take picture of a person. The collected picture has to be then placed inside the folder "common/faces/<person_name>" where
<person_name> is the folder associated to the new person that we want to recognize.
* "common/facerec.py" : It is a class wrapping all the function you need for both face detection and recognition.
* "common/centroidTracker.py": It is a class allowing to keep track of a person when no detection occurs (occlusion)

## Requirements

* [Install docker](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-18-04)
* [Install nvidia-docker2](https://github.com/NVIDIA/nvidia-docker)


## Setup

```
cd $HOME &&\
git clone https://github.com/visiont3lab/face_recognition.git && \
echo "export FACE_RECOGNITION=$HOME/face_recognition" >> $HOME/.bashrc && source $HOME/.bashrc
```

## Run
We have developed a docker container allowing to run a real time demo for face recognition. It will recognize the people contained inside the folder common/faces.
For this reason if you want to make the algorithm able to recognize yourself firstly you need to collect pictures. You can do this by running

1. Collect Face Images

    ```
    xhost +local:docker && \
        docker run --gpus all --rm  \
            -it --name deep_learning_face_recognition  \
            --env="DISPLAY=$DISPLAY" --env="QT_X11_NO_MITSHM=1" \
            --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --device=/dev/video0  \
            -v $FACE_RECOGNITION:/root/home/ws \
            visiont3lab/deep-learning:all \
            /bin/bash -c "cd /root/home/ws/common/utils && python3 takeAPicture.py"
    ```
    To take a picture simply press "p" . Around 20 images will be sufficient. Once you have the images create a folder inside common/faces with the name of the person and place inside it the images that are automatically saved into common/faces. Then you are ready to go.

2. Run Face detection and recognition

    ```
    xhost +local:docker && \
        docker run --gpus all  --rm  \
            -it --name deep_learning_face_recognition  \
            --env="DISPLAY=$DISPLAY" --env="QT_X11_NO_MITSHM=1" \
            --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --device=/dev/video0  \
            -v $FACE_RECOGNITION:/root/home/ws \
            visiont3lab/deep-learning:all \
            /bin/bash -c "cd /root/home/ws/algorithm/ && python3 custom_face_recognition.py"
    ```
    The upper command will start the face recognition algorithm. Feel free to remove or add new images inside the folder common/faces

## References
[Face Recognition Github repository](https://github.com/ageitgey/face_recognition) <br>
[Face Recognition PyImageSearch](https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/)
