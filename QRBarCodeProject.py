import cv2
import numpy as np
from pyzbar.pyzbar import decode
import pyautogui
import pygame

img1 = pyautogui.screenshot()
#img = cv2.imread('thuha.jpg')
#cap = cv2.VideoCapture(0)


#cap.set(3, 640)
#cap.set(4,480)

with open('Data.txt') as f:
    myDataList = f.read().splitlines()



while True:

    success, img = img1.read()
    for barcode in decode(img):
        myData = barcode.data.decode('utf-8')
        print(myData)

        if myData in myDataList:
            output='Completed!'
            mycolor=(0,255,0)
        else:
            output='Invalid Code!'
            mycolor=(0,0,255)



        pts = np.array([barcode.polygon],np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img,[pts],True,mycolor,3)
        pts2 = barcode.rect
        cv2.putText(img,output,(pts2[0],pts2[1]),cv2.FONT_HERSHEY_SIMPLEX, 0.9,mycolor,2)

    cv2.imshow('Output',img)
    cv2.waitKey(1)