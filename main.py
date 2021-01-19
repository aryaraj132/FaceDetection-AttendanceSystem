from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
import cv2
import numpy as np
import os
from datetime import datetime
from PIL import Image

class AttendenceWindow(Screen):
    pass

class DatasetWindow(Screen):
    pass

class WindowManager(ScreenManager):
    pass

kv = Builder.load_file("my.kv")

class MainApp(App):
    def build(self):
        return kv
    Dir = os.path.dirname(os.path.realpath(__file__))
    def Attendence(self, userId, info):
        try:
            user_id = int(userId)
            now = datetime.now()
            date_time = now.strftime("%d/%m/%Y %H:%M:%S")
            date = now.strftime("%d/%m/%Y")

            recog = cv2.face.LBPHFaceRecognizer_create()
            recog.read(self.Dir + '/trainer/trainer.yml')
            face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            font = cv2.FONT_HERSHEY_DUPLEX

            rec = 0
            id = 0
            face_numbers = 5
            names = [0,'Anand Raj','Ankit Raj', 'Usha Jha', 4, 5, 6, 'Aryan Raj', 8]

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
            if rec==1:
                try:
                    f1 = open(self.Dir + "/Attendance/User_"+str(id)+".txt","r")
                    data = f1.read()
                    f1.close()
                except:
                    data =''
                if date in data:
                    info.text = "Attendence already entered."
                else:
                    f = open(self.Dir + "/Attendance/User_"+str(id)+".txt","a+")
                    f.write(date_time)
                    f.write("\n")
                    f.close()
                    info.text = "Attendence entered successfully."
            camera.release()
            cv2.destroyAllWindows()
        except:
            info.text = "Some error occured. Try again!"
    
    def dataset(self,face_id,snap,info):
        try:
            snap_amount = int(snap)
            camera = cv2.VideoCapture(0)
            camera.set(3, 1920)
            camera.set(4, 1080)

            face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if len(face_id)<=0:
                raise Exception("No ID provided")
            count = 0
            while(True):
                rtrn, image=camera.read()
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face.detectMultiScale(gray, 1.3, 5)

                for(x,y,w,h) in faces:
                    cv2.rectangle(image, (x,y),(x+w,y+h),(255,0,0),2)
                    count+=1

                    cv2.imwrite(self.Dir + "/dataset/User_" + str(face_id) + '_' + str(count) + ".jpg", gray[y:y+h,x:x+w])
                    cv2.imshow('image', image)
                
                wait = cv2.waitKey(10) & 0xff
                if wait == 27:
                    break
                elif count >=snap_amount:
                    break
            camera.release()
            cv2.destroyAllWindows()
            info.text = "Face included successfully. Please train the system."
        except:
            info.text = "Some error occured. Try again!"
    def getImage_Labels(self, dataset,face):
            imagesPath=[os.path.join(dataset,f) for f in os.listdir(dataset)]
            faceSamples = []
            ids = []
            for imagePath in imagesPath:
                PIL_img=Image.open(imagePath).convert('L')
                img_numpy = np.array(PIL_img, 'uint8')

                id=int(os.path.split(imagePath)[-1].split("_")[1])
                faces = face.detectMultiScale(img_numpy)

                for (x,y,w,h) in faces:
                    faceSamples.append(img_numpy[y:y+h,x:x+w])
                    ids.append(id)
            return faceSamples,ids
    def train(self,info):
        info.text = "Training Faces."
        dataset = self.Dir + '/dataset'
        recog = cv2.face.LBPHFaceRecognizer_create()
        face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        faces,ids=self.getImage_Labels(dataset,face)
        recog.train(faces, np.array(ids))

        recog.write(self.Dir + '/trainer/trainer.yml')

        info.text = str(len(np.unique(ids))) + " face trained."

if(__name__ == "__main__"):
    MainApp().run()