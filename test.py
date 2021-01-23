import threading
from functools import partial
import cv2
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen


class MainScreen(Screen):
    pass


class Manager(ScreenManager):
    pass


kv = Builder.load_string('''
<MainScreen>:
    name: "Test"

    FloatLayout:
        Label:
            text: "Webcam from OpenCV?"
            pos_hint: {"x":0.0, "y":0.8}
            size_hint: 1.0, 0.2

        Image:
            # this is where the video will show
            # the id allows easy access
            id: vid
            size_hint: 1, 0.6
            allow_stretch: True  # allow the video image to be scaled
            keep_ratio: True  # keep the aspect ratio so people don't look squashed
            pos_hint: {'center_x':0.5, 'top':0.8}

        Button:
            text: 'Start Video'
            pos_hint: {"x":0.0, "y":0.1}
            size_hint: 1.0, 0.1
            font_size: 50
            on_release: app.start_video()
        Button:
            text: 'Stop Video'
            pos_hint: {"x":0.0, "y":0.0}
            size_hint: 1.0, 0.1
            font_size: 50
            on_release: app.stop_vid()
''')


class Main(App):
    def build(self):

        # start the camera access code on a separate thread
        # if this was done on the main thread, GUI would stop
        # daemon=True means kill this thread when app stops
        

        sm = ScreenManager()
        self.main_screen = MainScreen()
        sm.add_widget(self.main_screen)
        return sm
    def start_video(self):
        threading.Thread(target=self.doit, daemon=True).start()
    def doit(self):
        # this code is run in a separate thread
        self.do_vid = True  # flag to stop loop

        # make a window for use by cv2
        # flags allow resizing without regard to aspect ratio
        cv2.namedWindow('Hidden', cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)

        # resize the window to (0,0) to make it invisible
        cv2.resizeWindow('Hidden', 100, 100)
        cam = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_DUPLEX
        # start processing loop
        while (self.do_vid):
            ret, frame = cam.read()
            # ...
            # more code
            # ...

            # send this frame to the kivy Image Widget
            # Must use Clock.schedule_once to get this bit of code
            # to run back on the main thread (required for GUI operations)
            # the partial function just says to call the specified method with the provided argument (Clock adds a time argument)
            cv2.putText(frame, 'Hello', (10,10), font, 1, (255,255,255), 2)
            cv2.putText(frame, 'Hell', (10,20), font, 1, (255,255,0), 1)
            Clock.schedule_once(partial(self.display_frame, frame))

            cv2.imshow('Hidden', frame)
            cv2.waitKey(1)
        cam.release()
        cv2.destroyAllWindows()

    def stop_vid(self):
        # stop the video capture loop
        self.do_vid = False

    def display_frame(self, frame, dt):
        # display the current video frame in the kivy Image widget

        # create a Texture the correct size and format for the frame
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')

        # copy the frame data into the texture
        texture.blit_buffer(frame.tobytes(order=None), colorfmt='bgr', bufferfmt='ubyte')

        # flip the texture (otherwise the video is upside down
        texture.flip_vertical()

        # actually put the texture in the kivy Image widget
        self.main_screen.ids.vid.texture = texture

if __name__ == '__main__':
    Main().run()