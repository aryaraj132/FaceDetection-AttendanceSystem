import cv2
import numpy as np
import os
from datetime import datetime

now = datetime.now()
date_time = now.strftime("%d/%m/%Y %H:%M:%S")
date = now.strftime("%d/%m/%Y")

recog = cv2.face.LBPHFaceRecognizer_create()
recog.read('trainer/trainer.yml')
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

font = cv2.FONT_HERSHEY_DUPLEX

rec = 0
id = 0
face_numbers = 5
names = [0,'Anand Raj','Ankit Raj', 'Usha Jha', 4, 5, 6, 'Aryan Raj', 8]
user_id = int(input("\nEnter your ID : "))

camera = cv2.VideoCapture(0)
camera.set(3, 1920)
camera.set(4, 1080)

minWidth = 0.001*camera.get(3)
minHeight = 0.001*camera.get(4)

while True:
    rtrn, image=camera.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale( 
        gray,
        scaleFactor = 1.3,
        minNeighbors = face_numbers,
        minSize = (int(minWidth), int(minHeight)),
       )
    for(x,y,w,h) in faces:
        id, match = recog.predict(gray[y:y+h,x:x+w])
        if id == user_id:
            rec = 1
            cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
            status = "Attandance Recorded"
            cv2.putText(image, str(status), (x,y+h+25), font, 1, (0,255,0), 1)
        else:
            cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)
            status = "Attandance Not Recorded"
            cv2.putText(image, str(status), (x,y+h+25), font, 1, (0,0,255), 1)
        if (match < 100):
            try:
                name = names[id]
            except:
                name = "Unknown"
            match = "  {0}%".format(round(100 - match))
        else:
            name = "unknown"
            match = "  {0}%".format(round(100 - match))
        cv2.putText(image, str(name), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(image, str(match), (x+5,y+h-5), font, 1, (255,255,0), 1)
    
    cv2.imshow('camera',image) 
    k = cv2.waitKey(10) & 0xff 
    if k == 27:
        break

print("\n [Info] Exiting Program")
if rec==1:
    try:
        f1 = open("Attendance/User_"+str(id)+".txt","r")
        data = f1.read()
        f1.close()
    except:
        data =''
    if date in data:
        print("\n Attendence already entered.")
    else:
        f = open("Attendance/User_"+str(id)+".txt","a+")
        f.write(date_time)
        f.write("\n")
        f.close()
camera.release()
cv2.destroyAllWindows()