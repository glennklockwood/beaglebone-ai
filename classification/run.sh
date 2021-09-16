#!/usr/bin/env bash

USB_CAMERA=/dev/video1

ti-mct-heap-check -c

sudo mjpg_streamer \
    -i "input_opencv.so -r 640x480 -d ${USB_CAMERA} --filter ./classification.tidl.so" \
    -o "output_http.so -p 8090 -w /usr/share/mjpg-streamer/www"
