#!/usr/bin/env python3

import io
import cv2

camera = cv2.VideoCapture("/dev/video1")
_, image = camera.read()
cv2.imwrite('cv2capture.jpg', image)
