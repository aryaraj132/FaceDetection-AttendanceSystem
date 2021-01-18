import cv2
import numpy as np
from PIL import Image
import os

dataset = 'dataset'
recog = cv2.face.LBPHFaceRecognizer_create()
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def getImage_Labels(dataset):
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

print("\n [Info] Training Faces...")

faces,ids=getImage_Labels(dataset)
recog.train(faces, np.array(ids))

recog.write('trainer/trainer.yml')

print("\n [Info] {0} faces trained.".format(len(np.unique(ids))))