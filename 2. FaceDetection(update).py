import cv2, os, sqlite3, pyautogui
import numpy as np
from PIL import Image

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read('recognizer\\trainningData.yml')

def getProfile(id):

    conn = sqlite3.connect('DataSet.db')
    query = "SELECT * FROM people WHERE ID=" + str(id)
    cursor = conn.execute(query)
    profile = None

    for row in cursor:
        profile = row

    conn.close()
    return profile

fontface = cv2.FONT_HERSHEY_SIMPLEX

while True:
    screenshot = pyautogui.screenshot()
    screenshot = np.array(screenshot)

    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    for(x, y, w, h) in faces:
        cv2.rectangle(screenshot, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id,conf = recognizer.predict(gray[y:y+h,x:x+w])

        if conf < 40:
            profile = getProfile(id)

            if profile != None:
                cv2.putText(screenshot, "" + str(profile[1]), (x + 10, y + h + 30), fontface, 1, (0, 255, 0), 2)
                print(str(profile[0]) + "-" + str(profile[1]))
        else:
            cv2.putText(screenshot, "Unknow", (x + 10, y + h + 30), fontface, 1, (0, 0, 255), 2)

    cv2.imshow('Face detection', screenshot)
    if cv2.waitKey(1) == ord('q'):
        break
