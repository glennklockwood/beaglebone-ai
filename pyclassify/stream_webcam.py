#!/usr/bin/env python3
"""Streams video from webcam to http via Flask and OpenCV.

Demonstrates how to connect OpenCV to Flask to stream a webcam using Python.
Use this as a foundation for doing streaming image processing.
"""

import flask
import cv2

app = flask.Flask(__name__)

def stream_camera(camera):
    while True:
        ret, frame = camera.read()
        if ret:
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
