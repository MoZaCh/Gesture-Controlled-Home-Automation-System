import numpy as np
import cv2
import copy
import math
import threading
import time

# Env:
# OS: Windows 10
# Language: Python 3.6
# Package: OpenCV 2.4.13

# Appliances (Global variables)
lightStatus = False
fanStatus = False
lockStatus = False

# Timer
timerStatus = False

# Parameters
cap_region_x_begin = 0.5 # Stating point / Max width
cap_region_y_end = 0.8 # Starting point / Max width
threshold = 60 # Default binary threshold parameter
blurValue = 41 # Default gaussianblur parameter
bgSubThreshold = 50
learningRate = 0

# Variables
isBgCaptured = 0 # Integer used to check whether background is captured
triggerSwitch = False # Whilst True the keyboard simulator is enabled

# Array
gesture = [] # Array storing pattern of gesture that is performed and captured

# Font
font = cv2.FONT_HERSHEY_SIMPLEX

# Video capture (camera)
cap = cv2.VideoCapture(0)
cap.set(10, 200)
#cv2.nameWindow('trackbar')
#cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)


cap = cv2.VideoCapture(0) # Capture live video from the first camera device

while cap.isOpened():
    ret, frame = cap.read()
    #threshold = cv2.getTrackbarPos('trh1', 'trackbar')
    frame = cv2.bilateralFilter(frame, 5, 50, 100) # Smoothing filter
    cv2.rectangle(frame, 1) # The frame is flipped horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                  (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    cv2.imshow('original', frame)
    #cv2.putText(frame, 'H', (140, 250), font, .5, (255, 255, 255), 2, cv2.LINE_AA)

    # Main operation
    # if isBgCaptured == 1: # The section below will only run once background is captured
    #     img = removeBG(frame)
    #     img = img[0:int(cap_region_y_end * frame.shape[0]),
    #           int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]
    #     cv2.imshow('mask', img)

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (thresh, binaryImage) = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY)

    cv2.imshow('frame', frame) #Showing the normal capture
    cv2.imshow('grey', grey) #Showing the capture in greyscale
    cv2.imshow('Black white image', binaryImage)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
