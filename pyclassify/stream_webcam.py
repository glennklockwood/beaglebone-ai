#!/usr/bin/env python3
"""Streams video from webcam to http via Flask and OpenCV.

Demonstrates how to connect OpenCV to Flask to stream a webcam using Python.
Use this as a foundation for doing streaming image processing.
"""

import datetime

import flask
import cv2

app = flask.Flask(__name__)

def filter_image(frame):
    cv2.putText(
        img=frame,
        text=str(datetime.datetime.now()),
        org=(15, 60),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.0,
        color=(0, 255, 0),
        thickness=3)
    return frame


def stream_camera(camera):
    while True:
        ret, frame = camera.read()
        if ret:
            filter_image(frame)
            _, imstr = cv2.imencode(".jpg", frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                + bytes(imstr) + b'\r\n')

@app.route('/')
def video_feed():
    return flask.Response(
        stream_camera(cv2.VideoCapture("/dev/video1")),
        mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
