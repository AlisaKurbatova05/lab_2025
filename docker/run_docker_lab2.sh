#!/bin/bash

xhost +local:

docker run -it --rm \
    --env "QT_X11_NO_MITSHM=1" \
    --env DISPLAY=${DISPLAY} \
    --env LIBGL_ALWAYS_SOFTWARE=1 \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    --volume $(pwd)/../src:/workspace/lab_2025/src \
    --device /dev/dri:/dev/dri \
    --runtime=nvidia \
    --gpus all \
    --rm \
    --name lab2_docker \
    lab_2