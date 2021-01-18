import cv2
import os

camera = cv2.VideoCapture(0)
camera.set(3, 1920)
camera.set(4, 1080)

face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_id = input('\n Enter user ID : ')

print("\n [info] Initializing face capture.")

count = 0
while(True):
    rtrn, image=camera.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, 1.3, 5)

    for(x,y,w,h) in faces:
        cv2.rectangle(image, (x,y),(x+w,y+h),(255,0,0),2)
        count+=1

        cv2.imwrite("dataset/User_" + str(face_id) + '_' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', image)
    
    wait = cv2.waitKey(100) & 0xff
    if wait == 60:
        break
    elif count >=200:
        break

print("\n [Info] Exiting Program")

camera.release()
cv2.destroyAllWindows()