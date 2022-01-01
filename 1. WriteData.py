import cv2, sqlite3, os
import numpy as np
from PIL import Image


def insertOrUpdate(id, name):
    conn = sqlite3.connect("DataSet.db")
    query = "SELECT * FROM people WHERE ID="+str(id)
    cusror = conn.execute(query)

    isRecordExist = 0

    for row in cusror:
        isRecordExist = 1

    if isRecordExist == 0:
        query = "INSERT INTO people(ID, Name) VALUES("+str(id)+ ",'"+str(name)+ "')"
    else:
        query = "UPDATE people SET Name='"+str(name)+"'WHERE ID="+ str(id)
    conn.execute(query)
    conn.commit()
    conn.close()

#Webcam
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

id = input('Enter your ID: ')
name = input('Enter your Name: ')
insertOrUpdate(id, name)

sampleNum = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
        if not os.path.exists('DataSet'):
            os.makedirs('DataSet')
        
        sampleNum += 1
        cv2.imwrite('DataSet/User.' + str(id) + '.' + str(sampleNum) + '.jpg', gray[y: y + h, x: x + w])
    
    cv2.imshow('frame', frame)
    cv2.waitKey(1)

    if sampleNum > 300:
        break

cap.release()
cv2.destroyAllWindows()



#Part3##########################################
recognizer = cv2.face.LBPHFaceRecognizer_create()
path = 'DataSet'
print("Waiting...")

def getImagesWithId(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

    faces = []
    IDs = []

    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImg, 'uint8')

        #print(faceNp)

        Id = int(imagePath.split('\\')[1].split('.')[1])

        faces.append(faceNp)
        IDs.append(Id)
        
        #cv2.imshow('trainning', faceNp)
        cv2.waitKey(10)

    return faces, IDs

faces, Ids = getImagesWithId(path)
recognizer.train(faces, np.array(Ids))

if not os.path.exists('recognizer'):
    os.makedirs('recognizer')

recognizer.save("recognizer\\trainningData.yml")
print("Complete")

#cv2.destroyAllWindows()