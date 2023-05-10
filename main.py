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
import sys
import subprocess
from datetime import datetime
from PIL import Image
from kivy.core.window import Window
import pandas as pd
from time import sleep
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
    msg_thread = None
    att_thread = None
    data_thread = None
    train_thread = None
    msg_clear = True
    msg_timer = 0
    def message_cleaner(self):
        while True:
            if not self.msg_clear:
                while self.msg_timer > 0:
                    sleep(0.25)
                    self.msg_timer -= 0.25
                kv.get_screen('main').ids.info.text = ""
                kv.get_screen('second').ids.info.text = ""
                self.msg_clear = True
    def show_message(self,message, screen="both"):
        if (self.msg_thread is None) or not(self.msg_thread.is_alive()):
            self.msg_thread = threading.Thread(target=self.message_cleaner, daemon=True)
            self.msg_thread.start()
        if screen=="both":
            kv.get_screen('main').ids.info.text = message
            kv.get_screen('second').ids.info.text = message
            self.msg_timer = 5
            self.msg_clear = False
        elif screen=="main":
            kv.get_screen('main').ids.info.text = message
            self.msg_timer = 5
            self.msg_clear = False
        elif screen=="second":
            kv.get_screen('second').ids.info.text = message
            self.msg_timer = 5
            self.msg_clear = False
        return
    def build(self):
        self.icon = self.Dir + '/webcam.ico'
        self.title = 'Face Detection Attendance System'
        return kv
    def break_loop(self):
        self.running = False
    def startAttendence(self):
        if self.att_thread is not None and self.att_thread.is_alive():
            return
        self.att_thread = threading.Thread(target=self.Attendence, daemon=True)
        self.att_thread.start()
    def startTrain(self):
        if self.train_thread is not None and self.train_thread.is_alive():
            return
        self.train_thread = threading.Thread(target=self.train, daemon=True)
        self.train_thread.start()
    def startDataset(self):
        if self.data_thread is not None and self.data_thread.is_alive():
            return
        self.data_thread = threading.Thread(target=self.dataset, daemon=True)
        self.data_thread.start()
    def UserList(self):
        users_file = os.path.join(self.Dir, 'list', 'users.csv')
        if not (os.path.exists(users_file)):
            self.show_message("Users file not found.")
            return
        try:
            if sys.platform == "win32":
                os.startfile(users_file)
            else:
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.call([opener, users_file])
        except Exception as e:
            print(e)
            
    def AttendanceList(self):
        attendance_file = os.path.join(self.Dir, 'Attendance', 'Attendance.csv')
        if not (os.path.exists(attendance_file)):
            self.show_message("Attendance file not found.")
            return
        try:
            if sys.platform == "win32":
                os.startfile(attendance_file)
            else:
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.call([opener, attendance_file])
        except Exception as e:
            print(e)
    def Attendence(self):
        self.running = True
        dataset_path = os.path.join(self.Dir, 'dataset') 
        if not (os.path.isdir(dataset_path)):
            os.mkdir(dataset_path)
        try:
            user_id = int(kv.get_screen('main').ids.user_id.text)
            now = datetime.now()
            date_time = now.strftime("%d/%m/%Y %H:%M:%S")
            date = now.strftime("%d/%m/%Y")
            eye = cv2.CascadeClassifier(self.Dir + '/haarcascade_eye.xml')
            recog = cv2.face.LBPHFaceRecognizer_create()
            try:
                recog.read(os.path.join(self.Dir, 'trainer', 'trainer.yml'))
            except:
                self.show_message("Training file not found. Please Train the model first.", "main")
                return
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
                            try:
                                df = pd.read_csv(os.path.join(self.Dir, 'list', 'users.csv'))
                            except FileNotFoundError:
                                self.show_message("Users file not found.", "main")
                                return
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
                try:
                    df = pd.read_csv(os.path.join(self.Dir, 'Attendance', 'Attendance.csv'))
                except FileNotFoundError:
                    self.show_message("Attendance file not found.", "main")
                    return
                coll = ['0']*len(df['id'])
                if date in df.columns:
                    if (int(df.loc[df['id'] == id, date].iloc[0]))==0:
                        df.loc[df['id'] == id, date]=1
                        df.to_csv(os.path.join(self.Dir, 'Attendance', 'Attendance.csv'), index=False)
                        self.show_message("Attendance Recorded Successfully.")
                    else:
                        self.show_message("Attendence already entered.")
                else:
                    df[date] = coll
                    df.loc[df['id'] == id, date]=1
                    df.to_csv(os.path.join(self.Dir, 'Attendance', 'Attendance.csv'), index=False)
                    self.show_message("Attendence entered successfully.")
            else:
                self.show_message("Attendence not entered.", "main")
            camera.release()
            cv2.destroyAllWindows()
            return
        except Exception as e:
            self.show_message('Some error occured. Try again!', 'main')
            print(e)
            return
    def display_frame(self, frame, dt):
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(frame.tobytes(order=None), colorfmt='bgr', bufferfmt='ubyte')
        texture.flip_vertical()
        kv.get_screen('main').ids.vid.texture = texture
    def dataset(self):
        dataset_path = os.path.join(self.Dir, 'dataset') 
        list_path = os.path.join(self.Dir, 'list') 
        attendance_path = os.path.join(self.Dir, 'Attendance')
        if not (os.path.isdir(dataset_path)):
            os.mkdir(dataset_path)
        if not (os.path.isdir(list_path)):
            os.mkdir(list_path)
        if not (os.path.isdir(attendance_path)):
            os.mkdir(attendance_path)
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
                    try:
                        df = pd.read_csv(os.path.join(list_path, 'users.csv'))
                    except FileNotFoundError:
                        df = pd.DataFrame(columns = ['id', 'name'])
                    for i in range(len(df['id'])):
                        if df['id'].iloc[i] == int(face_id):
                            exist = True
                    if not exist:
                        df.loc[len(df.index)] = [int(face_id),name]
                        df.to_csv(os.path.join(list_path, 'users.csv'), index=False)
                    try:
                        df1 = pd.read_csv(os.path.join(attendance_path, 'Attendance.csv'))
                    except FileNotFoundError:
                        df1 = pd.DataFrame(columns = ['id', 'name'])
                    for i in range(len(df1['id'])):
                        if df1['id'].iloc[i] == int(face_id):
                            exist = True
                    if not exist:
                        arr = [int(face_id),name]
                        arr = np.concatenate((arr,[0]*(len(df1.columns)-2)))
                        df1.loc[len(df1.index)] = arr
                        df1.to_csv(os.path.join(attendance_path, 'Attendance.csv'), index=False)
                except Exception as e:
                    print(e)
                    self.show_message(str(e), "second")
                    return
                self.show_message("Dataset Created Successfully. Please train the system.", "second")
                return
        except:
            self.show_message("Some error occured. Please try again.", "second")
            return
    def getImage_Labels(self, dataset,face):
            imagesPath=[os.path.join(dataset,f) for f in os.listdir(dataset)]
            faceSamples = []
            ids = []
            if len(imagesPath)<=0:
                return None, None
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
        dataset_path = os.path.join(self.Dir, 'dataset')
        trainer_path = os.path.join(self.Dir, 'trainer') 
        if not (os.path.isdir(dataset_path)):
            kv.get_screen('main').ids.info.text = "No Dataset available."
            kv.get_screen('second').ids.info.text = "No Dataset available."
            sleep(10)
            kv.get_screen('main').ids.info.text = ""
            kv.get_screen('second').ids.info.text = ""
        if not (os.path.isdir(trainer_path)):
            os.mkdir(trainer_path)
        kv.get_screen('main').ids.info.text = "Training Faces."
        kv.get_screen('second').ids.info.text = "Training Faces."
        sleep(10)
        kv.get_screen('main').ids.info.text = ""
        kv.get_screen('second').ids.info.text = ""

        try:
            recog = cv2.face.LBPHFaceRecognizer_create()
            face = cv2.CascadeClassifier(self.Dir + '/haarcascade_frontalface_default.xml')

            faces,ids=self.getImage_Labels(dataset_path,face)
            if faces is None or ids is None:
                self.show_message("No Dataset available")
                return
            recog.train(faces, np.array(ids))

            recog.write(os.path.join(trainer_path, 'trainer.yml'))
            self.show_message(str(len(np.unique(ids))) + " face trained.")
        except:
            self.show_message("Some error occured. Try again!")
            return
if(__name__ == "__main__"):
    MainApp().run()