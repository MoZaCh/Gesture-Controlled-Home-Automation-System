import numpy as np
import cv2

cap = cv2.VideoCapture(0) #

while True:
    ret, frame = cap.read()

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (thresh, binaryImage) = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY)

    cv2.imshow('frame', frame) #Showing the normal capture
    cv2.imshow('grey', grey) #Showing the capture in greyscale
    cv2.imshow('Black white image', binaryImage)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
