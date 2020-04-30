import cv2
import numpy as np
import copy
import math
import threading
import time

# Environment:
# OS    : Windows 10
# Python: 3.8.1
# OpenCV: 4.2.0

# Appliances
lightStatus = False
fanStatus = False
lockStatus = False

# Timer
timerStatus = False
ClockStatus = False
Timeleft = 5

# Parameters
cap_region_x_begin = 0.5  # Starting point / Maximum width
cap_region_y_end = 0.8  # Starting point / Maximum width
threshold = 60  # Default binary threshold setting
blurValue = 41  # Default gaussianblur setting
bgSubThreshold = 50
learningRate = 0

# Variables
isBgCaptured = 0  # Integer used to check whether background is captured
triggerSwitch = False  # Whilst True the keyboard simulator is enabled

# Array
gesture = []  # Array storing pattern of gesture that is performed and translated into an integer
font = cv2.FONT_HERSHEY_SIMPLEX # Declaring the font to display

def printThreshold(thr):
    print("** Changed threshold to " + str(thr) + " **")

def removeBG(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

def Smart():
    global ClockStatus
    global Timeleft
    ClockStatus = True
    while ClockStatus != 0:
        time.sleep(0.985)
        Timeleft -= 1
        if Timeleft == 0:
            if len(gesture) != 0:
                gesture.clear()
                print(gesture, "** Inactivity **")
            ClockStatus = False
            return

def threadtimer():
    global timerStatus
    timerStatus = True
    timeleft = 2
    while timeleft != 0:
        time.sleep(0.985)
        timeleft -= 1
        if timeleft == 0:
            gesture.clear()
            timerStatus = False
            return

def applianceOn():
    global lightStatus
    print(gesture)

    if(lightStatus==False):
        for i in range(0,len(gesture)):
            if gesture[i] == 0:
                j = i
                for j in range(j,len(gesture)):
                    if gesture[j] == 4:
                        print("** Light On **")
                        lightStatus = True
                        timer = threading.Thread(target=threadtimer, name='threadtimer')
                        timer.start()
                        print(lightStatus)
                        gesture.clear()
                        #print("** Wait 3 seconds before next gesture! **")
                        #time.sleep(1)
                        return True
    return False

def applianceOff():
    global lightStatus
    print(gesture)

    if (lightStatus == True):
        for i in range(0, len(gesture)):
            if gesture[i] == 4:
                j = i
                for j in range(j, len(gesture)):
                    if gesture[j] == 0:
                        print("** Light Off **")
                        lightStatus = False
                        timer = threading.Thread(target=threadtimer, name='threadtimer')
                        timer.start()
                        print(lightStatus)
                        gesture.clear()
                        #print("** Wait 3 seconds before next gesture! **")
                        #time.sleep(1)
                        return False
    return True

#--------------Fan turn on/off START---------------------#

# def applianceOn():
#     global fanStatus
#     print(gesture)
#
#     if(fanStatus==False):
#         for i in range(0,len(gesture)):
#             if gesture[i] == 0:
#                 j = i
#                 for j in range(j,len(gesture)):
#                     if gesture[j] == 3:
#                         print("** Fan On **")
#                         fanStatus = True
#                         timer = threading.Thread(target=threadtimer, name='threadtimer')
#                         timer.start()
#                         print(fanStatus)
#                         gesture.clear()
#                         #print("** Wait 3 seconds before next gesture! **")
#                         #time.sleep(1)
#                         return True
#     return False
#
# def applianceOff():
#     global fanStatus
#     print(gesture)
#
#     if (fanStatus == True):
#         for i in range(0, len(gesture)):
#             if gesture[i] == 3:
#                 j = i
#                 for j in range(j, len(gesture)):
#                     if gesture[j] == 0:
#                         print("** Fan Off **")
#                         fanStatus = False
#                         timer = threading.Thread(target=threadtimer, name='threadtimer')
#                         timer.start()
#                         print(lightStatus)
#                         gesture.clear()
#                         #print("** Wait 3 seconds before next gesture! **")
#                         #time.sleep(1)
#                         return False
#     return True

#--------------Fan turn on/off END---------------------#

#--------------Lock / Unlock START---------------------#
# def applianceOn():
#     global lockStatus
#     print(gesture)
#
#     if(lockStatus==False):
#         for i in range(0,len(gesture)):
#             if gesture[i] == 0:
#                 j = i
#                 for j in range(j,len(gesture)):
#                     if gesture[j] == 2:
#                         print("** Lock On **")
#                         lockStatus = True
#                         timer = threading.Thread(target=threadtimer, name='threadtimer')
#                         timer.start()
#                         print(lockStatus)
#                         gesture.clear()
#                         #print("** Wait 3 seconds before next gesture! **")
#                         #time.sleep(1)
#                         return True
#     return False
#
# def applianceOff():
#     global lockStatus
#     print(gesture)
#
#     if (lockStatus == True):
#         for i in range(0, len(gesture)):
#             if gesture[i] == 4:
#                 j = i
#                 for j in range(j, len(gesture)):
#                     if gesture[j] == 0:
#                         print("** Unlocked **")
#                         lockStatus = False
#                         timer = threading.Thread(target=threadtimer, name='threadtimer')
#                         timer.start()
#                         print(lockStatus)
#                         gesture.clear()
#                         #print("** Wait 3 seconds before next gesture! **")
#                         #time.sleep(1)
#                         return False
#     return True
#--------------Lock / Unlock END---------------------#

def calculateFingers(res, drawing):
    global lightStatus
    global timerStatus
    global ClockStatus
    global Timeleft

    #  Convexity defect
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):  # Avoid crashing

            cnt = 0 # Default finger count is 0
            for i in range(defects.shape[0]):  # Calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # Using the cosine theorem
                if angle <= math.pi / 2:  # If the calculated angle is < 90 degree, treat it space between 2 fingers
                    cnt += 1 # Increment the finger count by 1
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)
            # print(cnt)

            if len(gesture) == 0 or gesture[len(gesture)-1] != cnt and timerStatus == False:
                gesture.append(cnt)
                Timeleft = 5
                print(gesture)

                if ClockStatus == False:
                    clock = threading.Thread(target=Smart, name='threadclock')
                    clock.start()

                if len(gesture) >= 2 and lightStatus == False:
                    applianceOn()

                elif len(gesture) >= 2 and lightStatus == True:
                    applianceOff()

                if len(gesture) >= 4:
                    gesture.clear()

            cnt = 0 # Reset the finger count
            return True, cnt
    return False, 0


# Camera object to capture real-time video
cap = cv2.VideoCapture(0)
cap.set(10, 200)
cv2.namedWindow('trackbar')
cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)

while cap.isOpened():
    ret, frame = cap.read()
    threshold = cv2.getTrackbarPos('trh1', 'trackbar')
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # Smoothing filter
    frame = cv2.flip(frame, 1)  # Horizontally flip the frame
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                  (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    cv2.imshow('original', frame)
    cv2.putText(frame, 'H', (140, 250), font, .5, (255, 255, 255), 2, cv2.LINE_AA)

    #  Main program operations
    if isBgCaptured == 1:  # The section below will only run once the background has been captured
        img = removeBG(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]),
              int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]
        cv2.imshow('mask', img)
        #font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(img, 'Test', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        # Convert the captured image into a binary image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        cv2.imshow('blur', blur)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        cv2.imshow('ori', thresh)

        # Get the coutours
        thresh1 = copy.deepcopy(thresh)
        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        maxArea = -1
        if length > 0:
            for i in range(length):  # Find the biggest contour (according to area)
                temp = contours[i]
                area = cv2.contourArea(temp)
                if area > maxArea:
                    maxArea = area
                    ci = i

            res = contours[ci]
            hull = cv2.convexHull(res)
            drawing = np.zeros(img.shape, np.uint8)
            cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

            isFinishCal, cnt = calculateFingers(res, drawing)

            if triggerSwitch is True:
                if isFinishCal is True and cnt <= 2:
                    print(cnt)
                    # app('System Events').keystroke(' ')  # simulate pressing blank space

        cv2.imshow('output', drawing)

    # Program keyboard operations
    k = cv2.waitKey(10)
    if k == 27:  # Press 'ESC' to exit
        cap.release()
        cv2.destroyAllWindows()
        break
    elif k == ord('b'):  # Press 'b' to capture the background (background subtraction)
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        isBgCaptured = 1
        print('** Background Captured **')
    elif k == ord('r'):  # Press 'r' to reset the background
        bgModel = None
        triggerSwitch = False
        isBgCaptured = 0
        print('** Reset Background **')
    elif k == ord('n'):
        triggerSwitch = True
        print('** Trigger On **')
