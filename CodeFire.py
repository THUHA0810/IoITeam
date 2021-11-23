import cv2
import numpy as np
import playsound

fire_repoted = 0
alarm_status = False
video = cv2.VideoCapture(0)

def play_audio():
    playsound.playsound("AlarmSound.mp3",True)

while True:
    ret, frame = video.read()
    frame = cv2.resize(frame,(1000,600))
    blur = cv2.GaussianBlur(frame,(15,15),0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lower = [18,50,50]
    upper = [35,255,255]
    lower = np.array(lower,dtype='uint8')
    upper = np.array(upper,dtype='uint8')
    
    mask = cv2.inRange(hsv,lower,upper)
    
    output = cv2.bitwise_and(frame,hsv,mask=mask)

    size = cv2.countNonZero(mask)
    if int(size)>15000:
        fire_repoted = fire_repoted + 1

        if fire_repoted >= 1:
            if alarm_status == False:
                print('''FIRE!!!''')                
                play_audio()
                alarm_status = True




    if ret == False:
        break
    
    cv2.imshow('output',output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
video.release()