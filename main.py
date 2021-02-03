import threading
from functools import partial
from kivy.clock import Clock
from kivy.graphics.texture import Texture
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
    running = False
    Dir = os.path.dirname(os.path.realpath(__file__))
    def build(self):
        self.icon = self.Dir + '/webcam.ico'
        self.title = 'Face Detection Attendance System'
        return kv
    def break_loop(self):
        self.running = False
    def startAttendence(self):
        threading.Thread(target=self.Attendence, daemon=True).start()
    def startTrain(self):
        threading.Thread(target=self.train, daemon=True).start()
    def startDataset(self):
        threading.Thread(target=self.dataset, daemon=True).start()
    def StudentList(self):
        os.startfile(self.Dir + '/list/students.csv')
    def AttendanceList(self):
        os.startfile(self.Dir + '/Attendance/Attendance.csv')
    def Attendence(self):
        self.running = True
        dataset_path = path = os.path.join(self.Dir, 'dataset') 
        if not (os.path.isdir(dataset_path)):
            os.mkdir(dataset_path)
        try:
            user_id = int(kv.get_screen('main').ids.user_id.text)
            now = datetime.now()
            date_time = now.strftime("%d/%m/%Y %H:%M:%S")
            date = now.strftime("%d/%m/%Y")
            eye = cv2.CascadeClassifier(self.Dir + '/haarcascade_eye.xml')
            recog = cv2.face.LBPHFaceRecognizer_create()
            recog.read(self.Dir + '/trainer/trainer.yml')
            face = cv2.CascadeClassifier(self.Dir + '/haarcascade_frontalface_default.xml')

            font = cv2.FONT_HERSHEY_DUPLEX

            rec = 0
            id = 0
            face_numbers = 5
            camera = cv2.VideoCapture(0)
            camera.set(3, 1920)
            camera.set(4, 1080)

            minWidth = 0.001*camera.get(3)
            minHeight = 0.001*camera.get(4)
            blink = 0
            is_eye = False
            while self.running:
                rtrn, image=camera.read()
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face.detectMultiScale( 
                    gray,
                    scaleFactor = 1.3,
                    minNeighbors = face_numbers,
                    minSize = (int(minWidth), int(minHeight)),
                )
                eyes = eye.detectMultiScale(image,scaleFactor = 1.2, minNeighbors = 5) 
                for (x, y, w, h) in eyes: 
                    cv2.rectangle(image, (x, y),  
                                (x + w, y + h), (255, 0, 0), 1)
                if len(eyes) >= 2:
                    is_eye = True
                    cv2.putText(image, "eye detected", (50,50), font, 1, (0,255,0), 1)
                if(len(faces)==0):
                    blink = 0
                if len(eyes) < 2:
                    blink+=1
                cv2.putText(image, "Blink(16+) : {}".format(blink), (1020,50), font, 1, (0,0,255), 2)
                for(x,y,w,h) in faces:
                    id, match = recog.predict(gray[y:y+h,x:x+w])
                    if (id == user_id) and (match < 35):
                        rec = 1
                        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
                        status = "Attandance Recorded"
                        cv2.putText(image, str(status), (x,y+h+25), font, 1, (0,255,0), 1)
                        try:
                            df = pd.read_csv(self.Dir + '/list/students.csv')
                            name = df.loc[df['id'] == id, 'name'].iloc[0]
                        except:
                            name = "Unknown"
                        match = "  {0}%".format(round(100 - match))
                    else:
                        rec = 0
                        cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)
                        status = "Attandance Not Recorded"
                        cv2.putText(image, str(status), (x,y+h+25), font, 1, (0,0,255), 1)
                        name = "unknown"
                        match = "  {0}%".format(round(100 - match))
                    cv2.putText(image, str(name), (x+5,y-5), font, 1, (255,255,255), 2)
                    cv2.putText(image, str(match), (x+5,y+h-5), font, 1, (255,255,0), 1)
                Clock.schedule_once(partial(self.display_frame, image))
                k = cv2.waitKey(1)
                if k == 27:
                    break
            if rec==1 and blink >15:
                df = pd.read_csv(self.Dir + '/Attendance/Attendance.csv')
                coll = ['0']*len(df['id'])
                if date in df.columns:
                    if (int(df.loc[df['id'] == id, date].iloc[0]))==0:
                        df.loc[df['id'] == id, date]=1
                        df.to_csv(self.Dir + '/Attendance/Attendance.csv', index=False)
                        kv.get_screen('main').ids.info.text = "Attendence entered successfully."
                    else:
                        kv.get_screen('main').ids.info.text = "Attendence already exist."
                else:
                    df[date] = coll
                    df.loc[df['id'] == id, date]=1
                    df.to_csv(self.Dir + '/Attendance/Attendance.csv', index=False)
                    kv.get_screen('main').ids.info.text = "Attendence entered successfully."
            else:
                kv.get_screen('main').ids.info.text = "Attendence not entered."
            camera.release()
            cv2.destroyAllWindows()
        except Exception as e:
            kv.get_screen('main').ids.info.text = "Some error occured. Try again!"
            print(e)
    def display_frame(self, frame, dt):
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(frame.tobytes(order=None), colorfmt='bgr', bufferfmt='ubyte')
        texture.flip_vertical()
        kv.get_screen('main').ids.vid.texture = texture
    def dataset(self):
        dataset_path = path = os.path.join(self.Dir, 'dataset') 
        if not (os.path.isdir(dataset_path)):
            os.mkdir(dataset_path)
        try:
            name = kv.get_screen('second').ids.user_name.text
            face_id = kv.get_screen('second').ids.user_id.text
            snap_amount = int(kv.get_screen('second').ids.snap.text)
            camera = cv2.VideoCapture(0)
            camera.set(3, 1920)
            camera.set(4, 1080)

            face = cv2.CascadeClassifier(self.Dir + '/haarcascade_frontalface_default.xml')
            if len(face_id)<=0 or len(name)<=0 or snap_amount <=0:
                kv.get_screen('second').ids.info.text = "All Fields Required"
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
                    df1 = pd.read_csv(self.Dir + '/Attendance/Attendance.csv')
                    for i in range(len(df1['id'])):
                        if df1['id'].iloc[i] == int(face_id):
                            exist = True
                    if not exist:
                        arr = [int(face_id),name]
                        arr = np.concatenate((arr,[0]*(len(df1.columns)-2)))
                        df1.loc[len(df1.index)] = arr
                        df1.to_csv(self.Dir + '/Attendance/Attendance.csv', index=False)
                except Exception as e:
                    print(e)
                kv.get_screen('second').ids.info.text = "Face included successfully. Please train the system."
        except:
            kv.get_screen('second').ids.info.text = "Some error occured. Try again!"
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
    def train(self):
        dataset_path = path = os.path.join(self.Dir, 'dataset') 
        if not (os.path.isdir(dataset_path)):
            os.mkdir(dataset_path)
        kv.get_screen('main').ids.info.text = "Training Faces."
        kv.get_screen('second').ids.info.text = "Training Faces."
        dataset = self.Dir + '/dataset'
        recog = cv2.face.LBPHFaceRecognizer_create()
        face = cv2.CascadeClassifier(self.Dir + '/haarcascade_frontalface_default.xml')

        faces,ids=self.getImage_Labels(dataset,face)
        recog.train(faces, np.array(ids))

        recog.write(self.Dir + '/trainer/trainer.yml')

        kv.get_screen('main').ids.info.text = str(len(np.unique(ids))) + " face trained."
        kv.get_screen('second').ids.info.text = str(len(np.unique(ids))) + " face trained."

if(__name__ == "__main__"):
    MainApp().run()