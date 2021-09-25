#!/usr/bin/env python3

import flask
import cv2
import time

CAMERA = None

app = flask.Flask(__name__)

def gen():
    global CAMERA
    while True:
        if not CAMERA:
            print("initializing camera")
            CAMERA = cv2.VideoCapture("/dev/video1")
        ret, frame = CAMERA.read()
        if ret:
            _, imstr = cv2.imencode(".jpg", frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytes(imstr) + b'\r\n')

@app.route('/')
def video_feed():
    return flask.Response(
        gen(),
        mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

