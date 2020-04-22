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

def removeBG(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.unit8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


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
    if isBgCaptured == 1: # The section below will only run once background is captured
        img = removeBG(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]),
              int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]
        cv2.imshow('mask', img)

        # Convert the captured image into a binary image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        cv2.imshow('blur', blur)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        cv2.imshow('ori', thresh)

        # Get the contours
        thresh1 = copy.deepcopy(thresh)
        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        maxArea = -1

        if length > 0:
            for i in range(length): # Find the biggest contour (according to area)
                temp = contours[i]
                area = cv2.contourArea(temp)
                if area > maxArea:
                    maxArea = area
                    ci = i

            res = contours[ci]
            hull = cv2.convexHull(res)
            drawing = np.zeros(img.shape, np.unit8)
            cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

            #isFinishCal, cnt = calculateFingers(res, drawing)

            if triggerSwitch is True:
                if isFinishCal is True and cnt <= 2:
                    print(cnt)
                    # app('System Events').keystroke(' ') # Simulate pressing blank space
        cv2.imshow('output', drawing)

    # Keyboard Operation
    k = cv2.waitKey(10)
    if k == 27: # press ESC to exit
        cap.release()
        cv2.destroyAllWindows()
        break
    elif k == ord('b'): # If b is pressed the background is captured
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        isBgCaptured = 1
        print('** Background Captured **')
    elif k == ord('r'): # if r is pressed the background is reset
        bgModel = None
        triggerSwitch = False
        isBgCaptured = 0
        print('** Background Reset **')
    elif k == ord('n'):
        triggerSwitch = True
        print('** Trigger On **')







    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (thresh, binaryImage) = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY)

    cv2.imshow('frame', frame) #Showing the normal capture
    cv2.imshow('grey', grey) #Showing the capture in greyscale
    cv2.imshow('Black white image', binaryImage)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
