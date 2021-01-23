from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
import cv2
import numpy as np
import os
from datetime import datetime
from PIL import Image
from kivy.core.window import Window
import pandas as pd
Window.clearcolor = (.8, .8, .8, 1)

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
                            df = pd.read_csv(self.Dir + '/list/students.csv')
                            name = df.loc[df['id'] == id, 'name'].iloc[0]
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
                df = pd.read_csv(self.Dir + '/Attendance/Attendance.csv')
                coll = ['0']*len(df['id'])
                if date in df.columns:
                    if (int(df.loc[df['id'] == id, date].iloc[0]))==0:
                        df.loc[df['id'] == id, date]=1
                        df.to_csv(self.Dir + '/Attendance/Attendance.csv', index=False)
                        info.text = "Attendence entered successfully."
                    else:
                        info.text = "Attendence already exist."
                else:
                    df[date] = coll
                    df.loc[df['id'] == id, date]=1
                    df.to_csv(self.Dir + '/Attendance/Attendance.csv', index=False)
                    info.text = "Attendence entered successfully."
            camera.release()
            cv2.destroyAllWindows()
        except:
            info.text = "Some error occured. Try again!"
    
    def dataset(self,face_id,name,snap,info):
        try:
            snap_amount = int(snap)
            camera = cv2.VideoCapture(0)
            camera.set(3, 1920)
            camera.set(4, 1080)

            face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if len(face_id)<=0 or len(name)<=0 or snap_amount <=0:
                info.text = "All Fields Required"
            else:
                count = 0
                while(True):
                    rtrn, image=camera.read()
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = face.detectMultiScale(gray, 1.3, 5)

                    for(x,y,w,h) in faces:
                        cv2.rectangle(image, (x,y),(x+w,y+h),(255,0,0),2)
                        count+=1

                        cv2.imwrite(self.Dir + "/dataset/"+str(name)+"_" + str(face_id) + '_' + str(count) + ".jpg", gray[y:y+h,x:x+w])
                        cv2.imshow('image', image)
                    
                    wait = cv2.waitKey(10) & 0xff
                    if wait == 27:
                        break
                    elif count >=snap_amount:
                        break
                camera.release()
                cv2.destroyAllWindows()
                try:
                    exist = False
                    df = pd.read_csv(self.Dir + '/list/students.csv')
                    for i in range(len(df['id'])):
                        if df['id'].iloc[i] == int(face_id):
                            exist = True
                    if not exist:
                        df.loc[len(df.index)] = [int(face_id),name]
                        df.to_csv(self.Dir + '/list/students.csv', index=False)
                except Exception as e:
                    print(e)
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